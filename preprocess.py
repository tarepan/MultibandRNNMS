import argparse
import os
import numpy as np
import json
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from utils import load_wav, melspectrogram
import random
import glob
from itertools import chain
from torchaudio.functional import mu_law_encoding
import torch

# Basic strategy
# separate data and metadata
#   metadata contain whether they should be for train or test

def process_wav(dataset, wav_path, audio_path, mel_path, params):
    """Convert wav_path into speaker_id and internally save processed data in arg's pathes.
    """
    # auto resample based on params (internally, librosa)
    wav = load_wav(wav_path, sample_rate=params["preprocessing"]["sample_rate"])
    wav /= np.abs(wav).max() * 0.999
    mel = melspectrogram(wav, sample_rate=params["preprocessing"]["sample_rate"],
                         preemph=params["preprocessing"]["preemph"],
                         num_mels=params["preprocessing"]["num_mels"],
                         num_fft=params["preprocessing"]["num_fft"],
                         min_level_db=params["preprocessing"]["min_level_db"],
                         hop_length=params["preprocessing"]["hop_length"],
                         win_length=params["preprocessing"]["win_length"],
                         fmin=params["preprocessing"]["fmin"])

    length_diff = len(mel) * params["preprocessing"]["hop_length"] - len(wav)
    wav = np.pad(wav, (0, length_diff), "constant")

    pad = (params["vocoder"]["sample_frames"] - params["vocoder"]["audio_slice_frames"]) // 2
    mel = np.pad(mel, ((pad,), (0,)), "constant")
    wav = np.pad(wav, (pad * params["preprocessing"]["hop_length"],), "constant")
    wav = mu_law_encoding(torch.from_numpy(wav), mu=2 ** params["preprocessing"]["bits"])
    wav = wav.numpy()
    # speakerID acuisition
    speaker = get_speakerid(wav_path, dataset)

    # save processed data
    np.save(audio_path, wav)
    np.save(mel_path, mel)
    
    return speaker, audio_path, mel_path, len(mel)

def get_speakerid(wav_path: str, dataset: str) -> str:
    if dataset == "ZeroSpeech2019e":
        #    "path/to/speech.wav"
        # => ("path/to", "speech.wav")
        # => "speech.wav"
        # => ("speech", ".wav")
        # => "speech"
        # => <split with "_">
        # => first element of splits == speaker
        return os.path.splitext(os.path.split(wav_path)[-1])[0].split("_")[0]
    elif dataset == "JSUT":
        # JSUT contain only one actor.
        return "female"
    else:
        raise "dataset error"

def preprocess(dataset: str, wav_dirs, out_dir, num_workers, params):
    # directries generation
    ## /{out_dir}
    ##   /audio
    ##   /mels
    audio_out_dir = os.path.join(out_dir, "audio")
    mel_out_dir = os.path.join(out_dir, "mels")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(mel_out_dir, exist_ok=True)

    # parallel processing preparation
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    # directory list => all .wav under dir => flatten to all dirs === iter of all .wav path
    wav_paths = chain.from_iterable(glob.iglob(f"{dir}/**/*.wav", recursive=True) for dir in wav_dirs)
    for wav_path in wav_paths:
        # filename
        fid = os.path.basename(wav_path).replace(".wav", ".npy")
        audio_path = os.path.join(audio_out_dir, fid)
        mel_path = os.path.join(mel_out_dir, fid)
        # parallel processing registration
        futures.append(executor.submit(partial(process_wav, dataset, wav_path, audio_path, mel_path, params)))

    # [(speaker, audio_path, mel_path, len(mel))]
    metadata = [future.result() for future in tqdm(futures)]
    write_metadata(metadata, out_dir, params)


def write_metadata(metadata, out_dir, params):
    # [(speaker, audio_path, mel_path, len(mel))]
    # shuffle and divide dataset into test & train
    random.shuffle(metadata)
    test = metadata[-params["preprocessing"]["num_evaluation_utterances"]:]
    train = metadata[:-params["preprocessing"]["num_evaluation_utterances"]]

    with open(os.path.join(out_dir, "test.txt"), "w", encoding="utf-8") as f:
        for m in test:
            f.write("|".join([str(x) for x in m]) + "\n")

    with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8") as f:
        for m in train:
            f.write("|".join([str(x) for x in m]) + "\n")

    frames = sum([m[3] for m in metadata])
    frame_shift_ms = params["preprocessing"]["hop_length"] / params["preprocessing"]["sample_rate"]
    hours = frames * frame_shift_ms / 3600
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["ZeroSpeech2019e", "JSUT"], default="ZeroSpeech2019e")
    parser.add_argument("--output", default="data")
    parser.add_argument("--num-workers", type=int, default=cpu_count())
    parser.add_argument("--data-dir", type=str, default="./english")
    parser.add_argument("--config-path", type=str, default="config.json")
    args = parser.parse_args()
    with open(args.config_path) as f:
        params = json.load(f)
    
    if args.dataset == "ZeroSpeech2019e":
        wav_dirs = [os.path.join(args.data_dir, "train", "unit"), os.path.join(args.data_dir, "train", "voice")]
    if args.dataset == "JSUT":
        wav_dirs = [args.data_dir]

    preprocess(args.dataset, wav_dirs, args.output, args.num_workers, params)


if __name__ == "__main__":
    main()
