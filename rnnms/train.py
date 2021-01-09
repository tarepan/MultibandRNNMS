from argparse import Namespace
from typing import Optional
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchaudio.functional import mu_law_decoding
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.datamodule import LightningDataModule

from .utils import save_wav
from .dataset import VocoderDataset
from .model import RNN_MS


def train_fn(args, params):
    train_dataset = VocoderDataset(meta_file=os.path.join(args.data_dir, "train.txt"),
                                   sample_frames=params["vocoder"]["sample_frames"],
                                   audio_slice_frames=params["vocoder"]["audio_slice_frames"],
                                   hop_length=params["preprocessing"]["hop_length"],
                                   bits=params["preprocessing"]["bits"])

    # -- matched with origin
    train_dataloader = DataLoader(train_dataset, batch_size=params["vocoder"]["batch_size"],
                                  shuffle=True, num_workers=1,
                                  pin_memory=True, drop_last=True)
    # --

    # -- looks good. can be transplanted.
    # Add original utterance to TensorBoard
    if args.resume is None:
        with open(os.path.join(args.data_dir, "test.txt"), encoding="utf-8") as f:
            test_wavnpy_paths = [line.strip().split("|")[1] for line in f]
        for index, wavnpy_path in enumerate(test_wavnpy_paths):
            muraw_code = torch.from_numpy(np.load(wavnpy_path))
            wav_pth = mu_law_decoding(muraw_code, 2**params["preprocessing"]["bits"])
            writer.add_audio("orig", wav_pth, global_step=global_step, sample_rate=params["preprocessing"]["sample_rate"])
            break
    # --

    # epoch loop
    for epoch in range(start_epoch, num_epochs + 1):
        running_loss = 0
        
        # step loop
        for i, (audio, mels) in enumerate(tqdm(train_dataloader, leave=False), 1):
            audio, mels = audio.to(device), mels.to(device)

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


def train(args: Namespace, datamodule: LightningDataModule) -> None:
    """Train RNN_MS on PyTorch-Lightning.
    """

    ckptAndLogging = CheckpointAndLogging(args.dir_root, args.name_exp, args.name_version)
    # setup
    gpus: int = 1 if torch.cuda.is_available() else 0  # single GPU or CPU
    model = RNN_MS()
    ckpt_cb = ModelCheckpoint(period=60, save_last=True, save_top_k=1, monitor="val_loss")
    trainer = pl.Trainer(
        gpus=gpus,
        auto_select_gpus=True,
        precision=32 if args.no_amp else 16,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.val_interval_epoch,
        # logging/checkpointing
        resume_from_checkpoint=ckptAndLogging.resume_from_checkpoint,
        default_root_dir=ckptAndLogging.default_root_dir,
        checkpoint_callback=ckpt_cb,
        logger=pl_loggers.TensorBoardLogger(
            ckptAndLogging.save_dir, ckptAndLogging.name, ckptAndLogging.version
        ),
        # reload_dataloaders_every_epoch=True,
        profiler=args.profiler,
        progress_bar_refresh_rate=30
    )

    # training
    trainer.fit(model, datamodule=datamodule)


class CheckpointAndLogging:
    """Generate path of checkpoint & logging.
    {dir_root}/
        {name_exp}/
            {name_version}/
                checkpoints/
                    {name_ckpt} # PyTorch-Lightning Checkpoint. Resume from here.
                hparams.yaml
                events.out.tfevents.{xxxxyyyyzzzz} # TensorBoard log file.
    """

    # [PL's Trainer](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#trainer-class-api)
    default_root_dir: Optional[str]
    resume_from_checkpoint: Optional[str]
    # [PL's TensorBoardLogger](https://pytorch-lightning.readthedocs.io/en/stable/logging.html#tensorboard)
    save_dir: str
    name: str
    version: str
    # [PL's ModelCheckpoint callback](https://pytorch-lightning.readthedocs.io/en/stable/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint)
    # inferred from above two

    def __init__(
        self,
        dir_root: str,
        name_exp: str = "default",
        name_version: str = "version_-1",
        name_ckpt: str = "last.ckpt",
    ) -> None:

        # ModelCheckpoint
        self.default_root_dir = dir_root
        self.resume_from_checkpoint = os.path.join(dir_root, name_exp, name_version, "checkpoints", name_ckpt)
        # TensorBoardLogger
        self.save_dir = dir_root
        self.name = name_exp
        self.version = name_version