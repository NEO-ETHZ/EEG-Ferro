# EEG-Ferro

Code for the forthcoming paper:

> **Personalized Spiking Neural Networks with Ferroelectric Synapses for EEG Signal Processing**
>
> Nikhil Garg, Anxiong Song, Niklas Plessnig, Nathan Savoia, and Laura Bégon-Lours
>
> Integrated Systems Laboratory, ETH Zürich

Paper link: **coming soon**.

## Overview

We train a spiking neural network (SNN) to classify left-hand vs. right-hand motor imagery from EEG signals, targeting deployment on neuromorphic hardware with ferroelectric FET (FeFET) synaptic devices. The pipeline consists of four steps:

1. **Floating-point pretraining** — Train the SNN on pooled multi-subject data using standard backpropagation.
2. **On-device training** — Retrain from scratch using a device-aware optimizer that models FeFET conductance updates via a beta-distribution kernel.
3. **On-device finetuning** — Load pretrained weights, quantise and add device noise, then finetune with the device model.
4. **Subject-specific transfer learning** — Adapt the model to individual subjects using only their own data (online, batch_size=1).

## Repository Layout

```text
dataset/              EEGMMIDB preprocessing, CV splits, and motor imagery dataset loaders
src/                  Reusable training, model, and device code
  models/             SNN architecture
  training/           Training, on-device learning, transfer learning, re-tuning
  devices/            Quantization
scripts/              Thin command-line entry points
configs/              Experiment parameters
results/              Figures, postprocessed CSVs, logs, checkpoints
notebooks/            Analysis and plotting notebooks
```

## Paper-to-Code Map

| Paper section | Repository location |
|---|---|
| Motor imagery dataset | `dataset/` |
| SNN architecture | `src/models/` |
| Floating point baseline | `src/training/train_floating_point.py` |
| On-device learning | `src/training/train_on_device.py` |
| Subject-specific transfer learning | `src/training/SSTL.py` |
| Quantization/noise | `src/devices/quantization.py` |
| Re-tuning | `src/training/retuning.py` |
| Figures/results | `results/` and `notebooks/` |


## Core Files

| File | Role |
|---|---|
| `src/models/snn.py` | CUBA-LIF spiking CNN architecture |
| `src/training/train_floating_point.py` | Floating-point baseline training |
| `src/training/train_on_device.py` | Device-aware on-device learning |
| `src/training/SSTL.py` | Subject-specific transfer learning |
| `src/training/eval_pretrain.py` | Evaluating per-subject accuracy of pretrained model|
| `src/training/retuning.py` | Low-overhead device-aware re-tuning |
| `src/devices/quantization.py` | Quantization and programming-noise utilities |
| `configs/config.py` | Experiment arguments and defaults |

## Results and Notebooks

Generated SVG figures are under `results/figures/`. Postprocessed CSVs are under `results/postprocessing/`, and raw experiment outputs/checkpoints are under `snn_experiment_results`.

Plotting and analysis notebooks are grouped under `notebooks/figures/`, Notebook for analysing MAC count is under `notebooks/MAC/`

## SNN Architecture

The network processes 2D spatial EEG maps (10×11 grid, 160 timesteps) through:

| Layer | Type | Output Shape | Description |
|-------|------|--------------|-------------|
| Conv1 | Conv2d + CUBA-LIF | 64 × 8 × 9 | 3×3 convolution |
| Conv2 | Conv2d + CUBA-LIF | 128 × 6 × 7 | 3×3 convolution |
| Pool  | AvgPool2d | 128 × 3 × 3 | 2×2 average pooling |
| Conv3 | Conv2d + CUBA-LIF | 256 × 1 × 1 | 3×3 convolution |
| TC1   | Temporal Conv + LIF | 256 | Kernel=3, aggregates 3 timesteps |
| R1    | Recurrent LIF | 256 | Adaptive threshold |
| FC1   | Linear + LIF | 256 | With dropout |
| FC2   | Linear (readout) | 2 | Weighted sum over timesteps |

All hidden layers use **CUBA-LIF** (Current-Based Leaky Integrate-and-Fire) neurons with learnable per-neuron current and voltage decay constants and rectangular surrogate gradients for backpropagation.

## Key Parameters

### Neuron Parameters (Eq. 2–5)

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `CDECAY` | β | 0.1 | Current decay constant (Eq. 2) |
| `VDECAY` | γ | 0.1 | Voltage decay constant (Eq. 3) |
| `VTH` | v_th | 0.1 | Spike threshold (Eq. 4) |
| `GRAD_WIN` | g | 0.3 | Surrogate gradient window width (Eq. 5) |
| `TH_AMP` | — | 0.01 | Adaptive threshold increment |
| `TH_DECAY` | — | 0.1 | Adaptive threshold decay |
| `BASE_TH` | — | 0.01 | Baseline threshold |

### Device Model Parameters (config.py, Eq. 14–15)

| Argument | Symbol | Default | Description |
|----------|--------|---------|-------------|
| `--dut` | — | dut2 | Device-under-test (dut1–dut5) |
| `--thr` | ε | 0.025 | Weight update threshold as fraction of range (Eq. 15) |
| `--thr_asym` | ε_asym | 1.0 | LTP/LTD threshold asymmetry (Eq. 15) |
| `--x` | 1/η | 4.0 | Noise scaling; η = 1/x is the RSD (Sec. 3.4) |
| `--q` | q | 0 | Quantization levels (0 = full precision) |
| `--A_scale` | — | 1.0 | Global conductance update scaling |
| `--A_asym` | — | 1.0 | LTP/LTD amplitude asymmetry |
| `--rsd_c2c` | σ_c2c | 0.0 | Cycle-to-cycle variation (RSD on A) |

### Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 1e-4 | Pretraining learning rate |
| `--finetune_lr` | 2e-4 | Finetuning learning rate |
| `--pretrain_epochs` | 20 | Number of pretraining epochs |
| `--retuning_epochs` | 4 | Number of re-tuning epochs |
| `--pretraining_batch_size` | 64 | Batch size for pretraining |
| `--retuning_batch_size` | 64 | Batch size for re-tuning |
| `--finetuning_batch_size` | 1 | Batch size for SSTL |
| `--finetune_epochs` | 5 | Number of SSTL epochs |
| `--seed` | 5 | Random seed |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) into a directory named "data":

```bash
mkdir -p eegmmidb/physionet
# Download using your preferred method, e.g.:
# wget -r -np https://physionet.org/files/eegmmidb/1.0.0/
```

### 3. Preprocess the data

```bash
python dataset/utility.py
```

This preprocesses and saves normalised, sliced EEG data as pickle files under `eegmmidb_slice_norm/`.

## Reproducing Paper Results

Run the steps in order:

```bash

# Floating point pretraining
python src/training/train_floating_point.py

# On-device training with FeFET model
bash scripts/run_train_on_device.sh

# Finetune pretrained model after simulated weight transfer
bash scripts/run_retuning.sh

# Subject-specific transfer learning (SSTL)
bash scripts/run_SSTL.sh

# Evaluating per-subject accuracy of pretrained model on held-out subject (for calculating delta accuracy after SSTL)
python src/training/eval_pretrain.py
```

Each script saves CSV log files and `.pth` model checkpoints to the `results/` directory.

## Dataset

- **Name**: PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)
- **Subjects**: 109 (6 excluded due to data quality: 88, 89, 92, 100, 104, 106)
- **Task**: Left-hand vs. right-hand motor imagery
- **Sampling rate**: 160 Hz
- **EEG channels**: 64, mapped to a 10×11 spatial grid
- **Epoch length**: 160 samples (1 second)

## Citation

```bibtex
@article{garg2026personalized,
  title   = {Personalized Spiking Neural Networks with Ferroelectric Synapses for EEG Signal Processing},
  author  = {Garg, Nikhil and Song, Anxiong and Plessnig, Niklas and Savoia, Nathan and B{\'e}gon-Lours, Laura},
  journal = {APL Machine Learning},
  year    = {2026}
}
```
