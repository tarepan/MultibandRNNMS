import argparse
import os
import json

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import save_wav
from dataset import VocoderDataset
from model import Vocoder

from expdir import makeExpDirs
from torchaudio.functional import mu_law_decoding


def save_checkpoint(model, optimizer, scheduler, step, checkpoint_dir, ckpt:bool):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step}
    torch.save(checkpoint_state, checkpoint_dir/"model-latest.pt")
    if ckpt == True:
        torch.save(checkpoint_state, checkpoint_dir/f"model.ckpt-{step}.pt")
        print(f"Saved checkpoint #{step}")
    return


def train_fn(args, params):
    # Directory preparation
    exp_dir = makeExpDirs(args.results_dir, args.exp_name)

    # Automatic Mixed-Precision
    if args.optim != "no":
        import apex

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Vocoder(mel_channels=params["preprocessing"]["num_mels"],
                    conditioning_channels=params["vocoder"]["conditioning_channels"],
                    embedding_dim=params["vocoder"]["embedding_dim"],
                    rnn_channels=params["vocoder"]["rnn_channels"],
                    fc_channels=params["vocoder"]["fc_channels"],
                    bits=params["preprocessing"]["bits"],
                    hop_length=params["preprocessing"]["hop_length"],
                    nc=args.nc,
                    device=device
                    )
    model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=params["vocoder"]["learning_rate"])

    # Automatic Mixed-Precision
    if args.optim != "no":
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.optim)

    scheduler = optim.lr_scheduler.StepLR(optimizer, params["vocoder"]["schedule"]["step_size"], params["vocoder"]["schedule"]["gamma"])

    if args.resume is not None:
        print(f"Resume checkpoint from: {args.resume}:")
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    train_dataset = VocoderDataset(meta_file=os.path.join(args.data_dir, "train.txt"),
                                   sample_frames=params["vocoder"]["sample_frames"],
                                   audio_slice_frames=params["vocoder"]["audio_slice_frames"],
                                   hop_length=params["preprocessing"]["hop_length"],
                                   bits=params["preprocessing"]["bits"])

    train_dataloader = DataLoader(train_dataset, batch_size=params["vocoder"]["batch_size"],
                                  shuffle=True, num_workers=1,
                                  pin_memory=True)

    num_epochs = params["vocoder"]["num_steps"] // len(train_dataloader) + 1
    start_epoch = global_step // len(train_dataloader) + 1

    # Logger
    writer = SummaryWriter(exp_dir/"logs")

    # Add original utterance to TensorBoard
    if args.resume is None:
        with open(os.path.join(args.data_dir, "test.txt"), encoding="utf-8") as f:
            test_wavnpy_paths = [line.strip().split("|")[1] for line in f]
        for index, wavnpy_path in enumerate(test_wavnpy_paths):
            muraw_code = torch.from_numpy(np.load(wavnpy_path))
            wav_pth = mu_law_decoding(muraw_code, 2**params["preprocessing"]["bits"])
            writer.add_audio("orig", wav_pth, global_step=global_step, sample_rate=params["preprocessing"]["sample_rate"])
            break


    for epoch in range(start_epoch, num_epochs + 1):
        running_loss = 0
        
        for i, (audio, mels) in enumerate(tqdm(train_dataloader, leave=False), 1):
            audio, mels = audio.to(device), mels.to(device)

            output = model(audio[:, :-1], mels)
            loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
            optimizer.zero_grad()

            # Automatic Mixed-Precision
            if args.optim != "no":
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            average_loss = running_loss / i

            global_step += 1

            if global_step % args.save_step == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, exp_dir/"params", False)

            if global_step % params["vocoder"]["checkpoint_interval"] == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, exp_dir/"params", True)

            if global_step % params["vocoder"]["generation_interval"] == 0:
                with open(os.path.join(args.data_dir, "test.txt"), encoding="utf-8") as f:
                    test_mel_paths = [line.strip().split("|")[2] for line in f]

                for index, mel_path in enumerate(test_mel_paths):
                    utterance_id = os.path.basename(mel_path).split(".")[0]
                    # unsqueeze: insert in a batch
                    mel = torch.FloatTensor(np.load(mel_path)).unsqueeze(0).to(device)
                    output = np.asarray(model.generate(mel), dtype=np.float64)
                    path = exp_dir/"samples"/f"gen_{utterance_id}_model_steps_{global_step}.wav"
                    save_wav(str(path), output, params["preprocessing"]["sample_rate"])
                    if index == 0:
                        writer.add_audio("cnvt", torch.from_numpy(output), global_step=global_step, sample_rate=params["preprocessing"]["sample_rate"])
        # finish a epoch
        writer.add_scalar("NLL", average_loss, global_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument('--results-dir', type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--config-path", type=str, default="config.json")
    parser.add_argument('--optim', choices=["no", "O0", "O1", "O2", "O3"], default="O1")
    parser.add_argument('--nc', type=bool, default=False, help="True if no-conditioning")
    parser.add_argument('--save-step', type=int, default=10, help="save per this step")
    args = parser.parse_args()
    with open(args.config_path) as f:
        params = json.load(f)
    train_fn(args, params)
