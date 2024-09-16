# WaveODE

An Efficient and High-Fidelity Diffusion Vocoder. See audio demo at https://jbjwzzhcds.github.io/WaveODE/

## Basic Usage

First modify the paths for `training`, `distillation` and `inference` in `params.py` as needed. 

```
cd src

# Extracting Mel-spectrograms
python dataset.py -i your_audio_path -o your_mel_saving_path

# Training
python train.py

# Distillation
python distillation.py

# Inference
python inference.py
```

## Package Requirements

WaveODE has been tested on the following requirements, and it should work fine with the latest version as well.

```
torch           2.0.1+cu118
torchaudio      2.0.2+cu118
torchdiffeq     0.2.3
tqdm            4.66.4
```

## References
- To be updated
