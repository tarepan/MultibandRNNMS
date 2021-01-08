<div align="center">

# RNN_MS-PyTorch <!-- omit in toc -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]
[![Paper](http://img.shields.io/badge/paper-arxiv.1811.06292-B31B1B.svg)][paper]  

</div>

Reimplmentation of neural vocoder **"RNN_MS"** with PyTorch.

![network](network.png?raw=true "Robust Universal Neural Vocoding")

## Demo
[Audio sample page](https://tarepan.github.io/UniversalVocoding).  

## How to Use
### Quick training <!-- omit in toc -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]

### Train
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
  pipenv run python UniversalVocoding/train.py
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
4. Default automatic mixed-precision ON (2x speed-up)

## Informative Results
### Mixed-Precision
Google Colaboratory Tesla T4  
default configs  
(tag + settings)

* w/o apex : 2.04it/s (exp_woApex), 2.02it/s (exp_wApex + "no")  
* w/  apex : 3.76it/s (exp_wApex + "O1"), 2.30it/s (exp_wApex + "O2"), 3.68it/s (exp_wApex + "O3")

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

## Original paper
[![Paper](http://img.shields.io/badge/paper-arxiv.1811.06292-B31B1B.svg)][paper]  
<!-- https://arxiv2bibtex.org/?q=1811.06292&format=bibtex -->
```
@misc{1811.06292,
Author = {Jaime Lorenzo-Trueba and Thomas Drugman and Javier Latorre and Thomas Merritt and Bartosz Putrycz and Roberto Barra-Chicote and Alexis Moinet and Vatsal Aggarwal},
Title = {Towards achieving robust universal neural vocoding},
Year = {2018},
Eprint = {arXiv:1811.06292},
}
```

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

## Dependency Notes
### PyTorch version <!-- omit in toc -->
PyTorch version: PyTorch v1.6 is working (We checked with v1.6.0).  

For dependency resolution, we do **NOT** explicitly specify the compatible versions.  
PyTorch have several distributions for various environment (e.g. compatible CUDA version.)  
Unfortunately it make dependency version management complicated for dependency management system.  
In our case, the system `poetry` cannot handle cuda variant string (e.g. `torch>=1.6.0` cannot accept `1.6.0+cu101`.)  
In order to resolve this problem, we use `torch==*`, it is equal to no version specification.  
`Setup.py` could resolve this problem (e.g. `torchaudio`'s `setup.py`), but we will not bet our effort to this hacky method.  

[paper]:https://arxiv.org/abs/1811.06292
[notebook]:https://colab.research.google.com/github/tarepan/UniversalVocoding/blob/main/UniversalVocoding.ipynb