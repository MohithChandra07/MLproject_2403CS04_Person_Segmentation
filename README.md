# Binary Person Segmentation on Penn-Fudan Pedestrian Dataset

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework for binary pedestrian semantic segmentation built from scratch using PyTorch on the **Penn-Fudan Pedestrian Dataset**. This repository documents the complete systematic engineering and experimental lifecycle across 10 controlled phases—from data engineering and custom U-Net design to hyperparameter optimization, loss weighting, capacity ablation, and final test evaluation.

---

## 📌 Project Overview

Pedestrian segmentation plays a fundamental role in autonomous driving, smart surveillance, and robotics. Unlike generic semantic segmentation datasets with tens of thousands of samples, the Penn-Fudan dataset poses unique challenges: high instance occlusion, limb ambiguities, diverse pedestrian scales, and a constrained sample size (170 images).

This project investigates how to achieve high-accuracy, robust binary segmentation without pre-trained backbones through rigorous architectural design, data augmentation, learning rate scheduling, and compound objective optimization.

### Key Highlights
- **From-Scratch U-Net Architecture**: 4-level encoder-decoder with double convolutional blocks, skip connections, and transpose convolutions (~7.76M parameters).
- **Controlled Experimental Progression**: 10 distinct phases systematically isolating data augmentation, learning rate, compound loss weighting, and architectural capacity.
- **Strict Data Hygiene**: Fixed 70/15/15 split (118 train, 25 val, 27 test). Test set was strictly isolated and evaluated only once at the conclusion of all tuning.
- **Final Independent Test Performance**:
  - **Global Pixel IoU**: **0.7513**
  - **F1-Score / Dice**: **0.8580**
  - **Precision**: **0.9023**
  - **Recall**: **0.8179**

---

## 📊 Experimental Results

### Phase-by-Phase Progression (Validation Metrics)

| Model / Experiment | Configuration | Val IoU | Val F1 | Precision | Recall | Parameters |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline U-Net** | LR 0.001, BCE 1.0 + Dice 1.0, No Aug | 0.7410 | 0.8512 | 0.8421 | 0.8606 | 7.76M |
| **Augmented U-Net** | + Flip (p=0.5) + Scale (0.9–1.1) | 0.7635 | 0.8659 | 0.8610 | 0.8708 | 7.76M |
| **LR Optimized** | LR 0.0005 (best among 0.001, 0.0005, 0.0001) | 0.7669 | 0.8681 | 0.8715 | 0.8647 | 7.76M |
| **Loss Optimized (Best)** | **1.5 BCE + 0.5 Dice**, LR 0.0005 | **0.7739** | **0.8725** | **0.8841** | **0.8612** | 7.76M |
| Capacity Ablation (Small) | Base channels = 16 | 0.7554 | 0.8607 | 0.8650 | 0.8564 | 1.94M |
| Capacity Ablation (Large) | Base channels = 48 | 0.7682 | 0.8689 | 0.8740 | 0.8638 | 17.47M |

### Final Test Evaluation (Held-Out Test Set, 27 Images)

```
Test Split Evaluation (27 untouched images):
--------------------------------------------------
Global Pixel IoU:          0.7513
Global Pixel F1-Score:     0.8580
Precision:                 0.9023
Recall:                    0.8179
Mean Per-Image IoU:        0.7328 ± 0.0914
Mean Per-Image F1:         0.8406 ± 0.0768
Test BCE Loss:             0.0469
Test Dice Loss:            0.1558
```

---

## 📑 Project Deliverables

| Deliverable | Description | Format |
|---|---|---|
| **College Project Report** | Comprehensive 13-page academic report adhering strictly to curriculum guidelines | [`PennFudan_Pedestrian_Segmentation_Report.pdf`](PennFudan_Pedestrian_Segmentation_Report.pdf) |
| **Presentation Poster (PDF)** | 16:9 high-resolution academic presentation poster | [`PennFudan_Pedestrian_Segmentation_Poster.pdf`](PennFudan_Pedestrian_Segmentation_Poster.pdf) |
| **Presentation Poster (PPTX)** | Fully editable 16:9 widescreen PowerPoint presentation | [`PennFudan_Pedestrian_Segmentation_Poster.pptx`](PennFudan_Pedestrian_Segmentation_Poster.pptx) |
| **Presentation Poster (Canva)** | Interactive online design workspace on Canva | [Canva Design Link](https://www.canva.com/design/DAHbKYEydMI/gIWss2omihcDXmBd4BL0QA/edit) |
| **Draw.io Architecture Diagram** | Raw multi-page editable vector file for Draw.io / Diagrams.net | [`architecture.drawio`](architecture.drawio) |
| **Referred Research Papers** | Curated collection of 14 published reference papers (PDFs) | [`referred_papers/`](referred_papers/) |
| **Overleaf LaTeX Package** | Ready-to-compile Overleaf bundle (compiles in <2 seconds) | [`report_overleaf.zip`](report_overleaf.zip) / [`report/`](report/) |
| **Complete Submission Folder** | Institutional submission package formatted to guideline naming | [`MLproject_2403CS04_Person_Segmentation/`](MLproject_2403CS04_Person_Segmentation/) |

---

## 📁 Repository Structure

```
.
├── MLproject_2403CS04_Person_Segmentation/  # Submission package (MLproject_rollno_topic)
│   ├── Code/                      # Source code, configs, logs, requirements
│   ├── Draw.io/                   # Raw Draw.io editable vector diagrams
│   ├── Poster/                    # Presentation poster (PDF and PPTX)
│   ├── Referred_Papers/           # 14 primary referenced research papers (PDFs)
│   └── Report/                    # Final college report PDF and Overleaf bundle
├── configs/                       # Configuration files
│   ├── config.yaml                # Master hyperparameter configuration
│   └── splits.json                # Fixed train (118), val (25), test (27) splits
├── experiments/                   # Experiment training logs and ablation CSVs
│   ├── augmented/                 # Data augmentation experiment logs
│   ├── lr_0005/ & lr_0001/        # Learning rate exploration logs
│   ├── loss_05_15/ & loss_15_05/  # Loss weight exploration logs
│   └── architecture_16/ & 48/     # Architecture capacity ablation logs
├── referred_papers/               # All 14 original research papers referenced
├── report/                        # LaTeX report source files
│   ├── main.tex                   # Standalone master college report LaTeX
│   ├── poster.tex                 # Beamer poster LaTeX source
│   ├── references.bib             # Complete bibliography
│   └── figures/                   # Optimized figures and result plots
├── results/                       # Final quantitative and visual outputs
│   ├── metrics/                   # Metrics JSON/CSV files across experiments
│   ├── plots/                     # Visual diagnostic plots
│   └── predictions/               # Test prediction overlays across best/median/worst cases
├── src/                           # Core PyTorch source code
│   ├── __init__.py
│   ├── augmentations.py           # Synchronized image-mask spatial transforms
│   ├── dataset.py                 # PyTorch Dataset and DataLoader pipelines
│   ├── evaluate.py                # Comprehensive metric evaluation routines
│   ├── losses.py                  # Compound BCE + Soft Dice loss
│   ├── metrics.py                 # IoU, F1, Precision, Recall computation
│   ├── model.py                   # Custom U-Net architecture from scratch
│   ├── preprocessing.py           # Image normalization and mask binarization
│   ├── train.py                   # Training loop with validation & early stopping
│   └── utils.py                   # Checkpointing, reproducibility, and logging
├── architecture.drawio            # Raw editable Draw.io multi-diagram file
├── PennFudan_Pedestrian_Segmentation_Report.pdf  # Final 13-page college project report
├── PennFudan_Pedestrian_Segmentation_Poster.pdf  # Final 16:9 poster (PDF)
├── PennFudan_Pedestrian_Segmentation_Poster.pptx # Editable 16:9 poster (PPTX)
├── report_overleaf.zip            # Overleaf-ready LaTeX zip bundle
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/MohithChandra07/Person-Segmentation.git
cd Person-Segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Setup
Download and extract the Penn-Fudan Pedestrian Dataset into `data/PennFudanPed`:
```bash
# Expected structure:
# data/PennFudanPed/
#   ├── PNGImages/
#   └── PedMasks/
```

### 3. Training
Train the optimal U-Net configuration (base_channels=32, lr=0.0005, BCE weight=1.5, Dice weight=0.5):
```bash
python -m src.train --name loss_15_05 --lr 0.0005 --bce_weight 1.5 --dice_weight 0.5 --augmentation flip_zoom
```

### 4. Evaluation
Evaluate a trained model checkpoint on the test set:
```bash
python -m src.evaluate --checkpoint checkpoints/loss_15_05_best.pt --split test
```

---

## 🔬 Architectural Details

```
Input: [B, 3, 256, 256] (RGB, Normalized)
│
├── Enc 1: DoubleConv(3 -> 32)      ──[Skip 1: 32]──┐
│   └── MaxPool2d (128x128)                         │
├── Enc 2: DoubleConv(32 -> 64)     ──[Skip 2: 64]──┼─┐
│   └── MaxPool2d (64x64)                           │ │
├── Enc 3: DoubleConv(64 -> 128)    ──[Skip 3: 128]─┼─┼─┐
│   └── MaxPool2d (32x32)                           │ │ │
├── Enc 4: DoubleConv(128 -> 256)   ──[Skip 4: 256]─┼─┼─┼─┐
│   └── MaxPool2d (16x16)                           │ │ │ │
├── Bottleneck: DoubleConv(256 -> 512)              │ │ │ │
│                                                   │ │ │ │
├── Dec 4: ConvTranspose + Concat + Conv(512 -> 256)◄─────┘
├── Dec 3: ConvTranspose + Concat + Conv(256 -> 128)◄───┘
├── Dec 2: ConvTranspose + Concat + Conv(128 -> 64) ◄─┘
└── Dec 1: ConvTranspose + Concat + Conv(64 -> 32)  ◄─┘
    └── Final Conv 1x1 (32 -> 1) -> Logits
Output: [B, 1, 256, 256] -> Sigmoid -> Binary Mask (Threshold = 0.5)
```

---

## 📜 Citation & References
1. O. Ronneberger, P. Fischer, and T. Brox, *"U-Net: Convolutional Networks for Biomedical Image Segmentation,"* in MICCAI, 2015.
2. L. Wang, J. Shi, G. Song, and I. Shen, *"Object Detection Combining Recognition and Segmentation,"* in ACCV, 2007 (Penn-Fudan Dataset).
3. F. Milletari, N. Navab, and S. Ahmadi, *"V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation,"* in 3DV, 2016.

---

## 👤 Author
**Mohith Chandra**  
GitHub: [@MohithChandra07](https://github.com/MohithChandra07)  
Contact: [mohithg404@gmail.com](mailto:mohithg404@gmail.com)
