<div align="center">

# Multiband RNN_MS <!-- omit in toc -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]

</div>

Fast and Simple vocoder, ***Multiband RNN_MS***.

<!-- generated by [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) -->
- [Demo](#demo)
- [Quick training](#quick-training)
- [How to Use](#how-to-use)
- [System Details](#system-details)
- [Results](#results)
- [References](#references)

## Demo
<!-- [Audio sample page](https://tarepan.github.io/UniversalVocoding).   -->
ToDO: Link super great impressive high-quatity audio demo.  

## Quick Training
Jump to ☞ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook], then Run. That's all!  

## How to Use
### 1. Install <!-- omit in toc -->

```bash
# pip install "torch==1.10.0" -q      # Based on your environment (validated with v1.10)
# pip install "torchaudio==0.10.0" -q # Based on your environment
pip install git+https://github.com/tarepan/MultibandRNNMS
```

### 2. Data & Preprocessing <!-- omit in toc -->
"Batteries Included".  
RNNMS transparently download corpus and preprocess it for you 😉  

### 3. Train <!-- omit in toc -->
```bash
python -m mbrnnms.main_train
```

For arguments, check [./mbrnnms/config.py](https://github.com/tarepan/MultibandRNNMS/blob/main/mbrnnms/config.py)  

### Advanced: Other datasets <!-- omit in toc -->
You can switch dataset with arguments.  
All [`speechcorpusy`](https://github.com/tarepan/speechcorpusy)'s preset corpuses are supported.  

```bash
# LJSpeech corpus
python -m mbrnnms.main_train data.data_name=LJ
```

### Advanced: Custom dataset <!-- omit in toc -->
Copy [`mbrnnms.main_train`] and replace DataModule.  

```python
    # datamodule = LJSpeechDataModule(batch_size, ...)
    datamodule = YourSuperCoolDataModule(batch_size, ...)
    # That's all!
```

[`mbrnnms.main_train`]:https://github.com/tarepan/MultibandRNNMS/blob/main/mbrnnms/main_train.py

## System Details
### Model <!-- omit in toc -->
- PreNet: GRU
- Upsampler: time-directional nearest interpolation
- Decoder: Embedding-auto-regressive generative RNN with 10-bit μ-law encoding

## Results
### Output Sample <!-- omit in toc -->
[Demo](#demo)

### Performance <!-- omit in toc -->
X [iter/sec] @ NVIDIA T4 on Google Colaboratory (AMP+, num_workers=8)  

It takes about Ydays for full training.  

## References
### Acknowlegements <!-- omit in toc -->
- [![Paper](http://img.shields.io/badge/paper-arxiv.1811.06292-B31B1B.svg)][paper]: Basic vocoder concept came from this paper.
- [bshall/UniversalVocoding]: Model and hyperparams are derived from this repository. All codes are re-written.


[paper]:https://arxiv.org/abs/1811.06292
[notebook]:https://colab.research.google.com/github/tarepan/MultibandRNNMS/blob/main/mbrnnms.ipynb
[bshall/UniversalVocoding]:https://github.com/bshall/UniversalVocoding
