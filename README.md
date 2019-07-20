# Robust Universal Neural Vocoding
A PyTorch implementation of **RNN_MS** in "Jaime, et al. (2018) [Towards achieving robust universal neural vocoding](https://arxiv.org/abs/1811.06292)".
<!-- Audio samples can be found [here](https://bshall.github.io/UniversalVocoding/). -->


![network](network.png?raw=true "Robust Universal Neural Vocoding")

## Quick Start
1. Ensure you have PyTorch@>=1.1.

2. Clone the repo:
  ```
  git clone https://github.com/tarepan/UniversalVocoding.git
  cd ./UniversalVocoding
  ```
3. Install requirements:
  ```
  pipenv install
  pipenv install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git#egg=apex
  ```
4. Download and extract ZeroSpeech2019 TTS without the T English dataset:
  ```
  wget https://download.zerospeech.com/2019/english.tgz
  tar -xvzf english.tgz
  ```
5. Extract Mel spectrograms and preprocess audio:
  ```
  pipenv run python preprocess.py
  ```

6. Train the model:
  ```
  pipenv run python train.py
  ```
  
7. Generate:
  ```
  pipenv run python generate.py --checkpoint=/path/to/checkpoint.pt --wav-path=/path/to/wav.wav
  ```

## Pretrained Models
Pretrained weights for the 9-bit model are available in [original repository](https://github.com/bshall/UniversalVocoding/releases/tag/v0.1).

## Notable Differences from the Paper
1. Trained on 16kHz audio from 102 different speakers ([ZeroSpeech 2019: TTS without T](https://zerospeech.com/2019/) English dataset)
2. The model generates 9-bit mu-law audio (planning on training a 10-bit model soon)
3. Uses an embedding layer instead of one-hot encoding

## Knowledge from Original Repository
- training speed [issue#5](https://github.com/bshall/UniversalVocoding/issues/5)
  - intelligible samples by 20k steps
  - decent results by 60k-80k steps
  - no data of father step training
- input spectrogram [issue#4](https://github.com/bshall/UniversalVocoding/issues/4)
  - more "smoothed" spectrogram could be used
    - demo of VQ-VAE output (smoothed spec) => RNN_MS => .wav
- sensitivity to spectrogram shape [issue#3](https://github.com/bshall/UniversalVocoding/issues/3)
  - stable training regardless of shape
    - n_fft=1024 also work well
- other dataset [issue#2](https://github.com/bshall/UniversalVocoding/issues/2)
  - only ZeroSpeech2019, not yet (seems to be interested in other dataset?)

## Acknowlegements
- https://github.com/fatchord/WaveRNN

## Development notes
### PyTorch version control
control with pip/pipenv is so hard because installation target differ depending on evironments.  
windows needs
```
pip3 install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl
```
### apex install
pipenv (pipfile) cannot handle "pip install" options.  
apex needs options, so determined to manual install.    