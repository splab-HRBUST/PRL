# PRL-VITS: Posterior Representation Learning for Text-to-Speech

<p align="center">
  <img src="resources/rae_vits_svg_native_shapes.svg" alt="PRL-VITS architecture overview" width="900" />
</p>

<p align="center">
  A VITS-based TTS framework that strengthens <b>training-time posterior representation learning</b>
  with frozen HuBERT acoustic evidence, posterior fusion, and reconstruction regularization.
</p>

## Overview

PRL-VITS extends baseline VITS by injecting **frozen HuBERT acoustic evidence** into the posterior branch during training. The injected SSL feature is projected into the posterior hidden space, fused with spectrogram-derived posterior evidence, and constrained by an auxiliary Mel reconstruction objective.

The extra branch is **training-only**. Inference still follows the original text-conditioned VITS pathway, so no extra inference cost is introduced.

## First Innovation Point

### HuBERT-Guided Posterior Acoustic Injection

The first key innovation of PRL-VITS is to enhance posterior-side acoustic modeling with **training-only self-supervised speech evidence**.

In baseline VITS, posterior statistics are estimated mainly from spectrogram-side posterior features. In PRL-VITS, frozen HuBERT representations from target speech are additionally injected into the posterior path through three steps:

1. **Acoustic projection**
   - HuBERT features are projected from the SSL feature space to the posterior hidden-channel space.
   - Implemented as `ssl_proj` in [models.py](models.py).

2. **Posterior fusion**
   - The projected HuBERT feature is concatenated with the spectrogram-derived posterior feature.
   - A lightweight `1x1` convolution compresses the fused feature back to the original posterior width.
   - Implemented as `fusion` in [models.py](models.py).

3. **Training-only removal at inference**
   - The HuBERT path is used only during training.
   - Inference keeps the original VITS text-to-waveform route unchanged.

This design makes the posterior encoder learn from both:

- low-level spectral evidence from the target spectrogram,
- high-level acoustic and phonetic regularities encoded by HuBERT.

## Method Highlights

- Posterior-side acoustic enhancement rather than decoder-side feature injection.
- Frozen HuBERT teacher for robust acoustic supervision.
- Auxiliary reconstruction regularization via a Mel decoder on the projected SSL representation.
- No inference overhead compared with baseline VITS.
- Support for both single-speaker and multi-speaker training.

## Code Mapping

Key PRL-VITS components are implemented in:

- [models.py](models.py)
  - `ssl_proj`: posterior-side SSL projection
  - `fusion`: posterior fusion
  - `rae_decoder`: auxiliary reconstruction decoder
- [data_utils.py](data_utils.py)
  - cached HuBERT feature loading
  - temporal interpolation to target spectrogram length
- [extract_hubert_features.py](extract_hubert_features.py)
  - offline HuBERT feature extraction
- [train.py](train.py)
  - single-speaker training
- [train_ms.py](train_ms.py)
  - multi-speaker training

## Repository Structure

```text
PRL/
├─ configs/
├─ filelists/
├─ monotonic_align/
├─ resources/
├─ text/
├─ data_utils.py
├─ extract_hubert_features.py
├─ models.py
├─ train.py
├─ train_ms.py
└─ inference.py
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For some environments you may also need:

```bash
apt-get install espeak
```

### 2. Build monotonic alignment search

```bash
cd monotonic_align
python setup.py build_ext --inplace
```

### 3. Prepare HuBERT features

```bash
python extract_hubert_features.py \
  --filelists filelists/ljs_audio_text_train_filelist.txt \
              filelists/ljs_audio_text_val_filelist.txt \
              filelists/ljs_audio_text_test_filelist.txt
```

### 4. Training

Single-speaker:

```bash
python train.py -c configs/ljs_base.json -m ljs_base
```

Multi-speaker:

```bash
python train_ms.py -c configs/vctk_base.json -m vctk_base
```

## Notes

- HuBERT features are loaded from cached tensors instead of being extracted online during training.
- If SSL feature length does not match spectrogram length, the loader interpolates the feature sequence to the target frame length.
- Missing SSL features will raise an explicit file-not-found error.

## Inference

See [inference.py](inference.py).

## Acknowledgement

This codebase is built on top of the original VITS implementation and adapts it for posterior representation learning experiments with HuBERT-guided acoustic evidence.
