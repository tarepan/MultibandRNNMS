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

### Install

```bash
!pip install git+https://github.com/tarepan/UniversalVocoding#main -q
```

### Training
```bash
!python -m rnnms.main_train
```

For arguments, please check ./scyclonepytorch/args.py

## System details
### Model
- Encoder: 2-layer bidi-GRU (so that no time-directional compression)
- Upsampler: x200 time-directional latent upsampling with interpolation
- Decoder: Latent-conditional, embedded-auto-regressive generative RNN with 10-bit μ-law encoding

## Differences from the Paper

| property      |  paper           | this repo       |
|:--------------|:-----------------|:----------------|
| sampling rate | 24 kHz           |   16 kHz        |
| AR input      | one-hot          | embedding       |
| Dataset       | internal? 74 spk | LJSpeech, 1 spk |
| Presicion     |   -              | 32/16 Mixed     |

## Informative Results
### Mixed-Precision
Google Colaboratory Tesla T4  
default configs  

* w/o AMP : x.xxit/s
* w/  AMP : x.xxit/s

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
- https://github.com/bshall/UniversalVocoding

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
[notebook]:https://colab.research.google.com/github/tarepan/UniversalVocoding/blob/main/rnnms.ipynb