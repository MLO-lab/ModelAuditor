# ModelAuditor

An autonomous LLM agent that audits and improves the reliability of clinical AI vision models. ModelAuditor converses with practitioners to understand their deployment context, selects task-specific evaluation metrics (guided by [MetricsReloaded](https://www.nature.com/articles/s41592-023-02151-z)), simulates clinically relevant distribution shifts, and generates interpretable reports with actionable mitigation strategies. The system uses a multi-agent debate mechanism for robust metric and shift selection. The computational backend (metrics and distribution shifts) lives in the companion repository [ModelAuditorCore](https://github.com/MLO-lab/ModelAuditorCore).

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Dataset Documentation](#dataset-documentation)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)

---

## Installation

### Prerequisites

- **Python 3.10** (tested extensively with 3.10.15)
- An API key for one of the supported LLM providers:
  - [Anthropic](https://console.anthropic.com/) (default: Claude 4.5 Sonnet)
  - [OpenAI](https://platform.openai.com/) (validated: GPT-5.2)

### 1. Create a virtual environment

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv --python 3.10
source .venv/bin/activate
```

Or with conda:

```bash
conda create -n modelauditor python=3.10 -y
conda activate modelauditor
```

Or with venv:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```



### 2. Install ModelAuditor

Using uv (recommended — handles all dependency issues automatically):

```bash
uv pip install "setuptools<81"
uv sync
```

Using pip:

```bash
pip install "setuptools<81"
pip install --no-build-isolation "MetricsReloaded @ git+https://github.com/csudre/MetricsReloaded.git"
pip install -e .
```

> **Why the extra steps?** ModelAuditor depends on [MetricsReloaded](https://github.com/csudre/MetricsReloaded). MetricsReloaded's build requires `setuptools<81` (for `pkg_resources`). With uv, `setuptools<81` must be installed in the venv before `uv sync` so it is available during the no-build-isolation build. With pip, MetricsReloaded must be installed explicitly with `--no-build-isolation` first.

For medical imaging datasets (Camelyon17 via WILDS, MedMNIST, SIIM-ISIC models):

```bash
uv sync --extra medical
# or
pip install -e ".[medical]"
```

### 3. Set up API keys

```bash
# For the default Claude 4.5 Sonnet backbone
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# If using an OpenAI model instead
export OPENAI_API_KEY="your-openai-api-key"
```

### 4. Verify the installation

```bash
python main.py --model resnet18 --dataset CIFAR10 --weights examples/cifar10/cifar10.pth --subset 20
```

This runs a minimal audit on the bundled CIFAR-10 toy model (ResNet-18). The agent will ask you to describe the task — respond with something like *"A 10-class natural image classifier"* and follow the prompts. If you see the final audit report, the installation is working.

---

## Quick Start

This example audits a pretrained ResNet-50 on a dermatology dataset from Vienna (HAM10000). The audit uses a bundled 50-image demo validation subset (25 benign keratosis + 25 melanoma from the Vienna clinic). No dataset download required.

### 1. Download a pretrained model

```bash
pip3 install huggingface_hub
hf download lukaskuhndkfz/ModelAuditor ham10000_resnet50_1_224.pt --local-dir models
```

### 2. Run the audit

```bash
python main.py \
  --model resnet50 \
  --dataset ham10000 \
  --weights models/ham10000_resnet50_1_224.pt 
```

The agent auto-detects the bundled demo images in `examples/ham10000/`. To use the full dataset instead, download it from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and set it up following the [HAM10000 section](#ham10000-dermatology) under Reproducing Paper Results.

The agent will:
1. Ask you to describe your clinical task (e.g., *"Binary classification of skin lesions into benign keratosis (bkl) and melanoma (mel), trained on dermatoscopic images from Vienna"*)
2. Ask follow-up questions about deployment context and failure modes
3. Select metrics (e.g., Sensitivity, Specificity, AUROC, ECE) and distribution shifts (e.g., BrightnessShift, GaussianNoise, Rotation) via multi-agent debate
4. Run the perturbation-evaluation audit
5. Generate a structured report with findings and torchvision v2 augmentation recommendations

### Expected output (abbreviated)
The expected output depends on how you answer the agent's questions. If you tell the Auditor the model is intended to be deployed in other clinics across Europe and Australia and that you expect to see fewer melanoma than benign lesions, the final audit will look like this: 

```
╭─ Final Audit Report ─────────────────────────────────────────────────────╮
│ ## 0. Clinical Summary                                                   │
│ The model is NOT reliable for clinical use in its current state. When     │
│ images have adjusted brightness, altered contrast, or undergo standard   │
│ JPEG compression, the model fails to detect melanoma in nearly all       │
│ cases — 0% sensitivity ...                                               │
│                                                                          │
│ ## 1. Executive Summary                                                  │
│ Severe brittleness to common image perturbations. Critical failure       │
│ under brightness/contrast (0% sensitivity), JPEG compression (0%),       │
│ and Gaussian blur (68-72%). Robust to noise (96%) ...      │
│                                                                          │
│ ## 2. Detailed Findings                                                  │
│ - BrightnessShift: 0% sensitivity across all severity levels             │
│ - ContrastShift: 0% sensitivity, AUROC drops to 0.60                     │
│ - JPEGCompression: 0% sensitivity even at quality=90                     │
│ - GaussianBlur: sensitivity drops to 68%, missing 1 in 3 melanomas       │
│ - GaussianNoise: 92-96% sensitivity, AUROC 0.98-0.99 (robust) ...       │
│                                                                          │
│ ## 4. Actionable Recommendations                                         │
│ ```python                                                                │
│ v2.RandomApply([v2.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),  │
│ v2.RandomApply([v2.JPEG(quality=(60, 95))], p=0.5),                      │
│ v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1,2.0))], p=0.3)│
│ ```                                                                      │
╰──────────────────────────────────────────────────────────────────────────╯
```

The full conversation is saved to `results/conversation_<timestamp>.txt`.

---

## Reproducing Paper Results

### Data and model setup

Two utility scripts automate dataset downloading and split preparation. Check which datasets and models you already have:

```bash
python data/download_datasets.py --list
```

**Download datasets and pretrained models:**

```bash
# Download all auto-downloadable datasets (CIFAR-10, Camelyon17) and pretrained models
python data/download_datasets.py --all

# Or download individually
python data/download_datasets.py --dataset camelyon17
python data/download_datasets.py --models

# HAM10000 requires manual download first, then pass the zip directory
python data/download_datasets.py --dataset ham10000 --ham10000-zip-dir ~/Downloads
```

**Generate CSV split manifests** (splits used in the paper):

```bash
# All datasets at once
python data/prepare_splits.py --all

# Or individually
python data/prepare_splits.py --dataset ham10000
python data/prepare_splits.py --dataset chexpert
python data/prepare_splits.py --dataset camelyon17
python data/prepare_splits.py --dataset polypgen
```

This writes CSV manifests to `data/splits/` with filepath and label columns for each split (e.g., `ham10000_vidir_modern.csv`, `ham10000_rosendahl.csv`, `chexpert_eval5k.csv`). See [Dataset Documentation](#dataset-documentation) for full preprocessing and split details.

---

### SIIM-ISIC Audit (Pretrained Checkpoint)

**Task:** Melanoma classification using a pretrained EfficientNet ensemble.

**Prerequisites:** Install the medical extras (provides `geffnet`):

```bash
uv sync --extra medical
# or: pip install -e ".[medical]"
```

**Download the public ensemble checkpoints** from [Zenodo](https://zenodo.org/records/10049217) and place them in `models/isic/`:

```bash
mkdir -p models/isic
curl -L -o models/isic/9c_b7ns_1e_224_ext_15ep_best_fold0.pth \
  "https://zenodo.org/api/records/10049217/files/9c_b7ns_1e_224_ext_15ep_best_fold0.pth/content"
curl -L -o models/isic/9c_b6ns_1e_224_ext_15ep_best_fold1.pth \
  "https://zenodo.org/api/records/10049217/files/9c_b6ns_1e_224_ext_15ep_best_fold1.pth/content"
curl -L -o models/isic/9c_b5ns_1e_224_ext_15ep_best_fold2.pth \
  "https://zenodo.org/api/records/10049217/files/9c_b5ns_1e_224_ext_15ep_best_fold2.pth/content"
```

**Prepare the ISIC evaluation data:** Organize images from the [ISIC Archive](https://challenge.isic-archive.com/data) as an ImageFolder:

```
data/isic/train/
├── benign/
│   └── *.jpg
└── malignant/
    └── *.jpg
```

**Run the audit:**

```bash
python main.py \
  --model siim-isic \
  --dataset isic \
  --weights models/isic/9c_b7ns_1e_224_ext_15ep_best_fold0.pth \
  --subset 200
```

### DeepDerm Audit (Pretrained Checkpoint)

**Task:** Skin condition classification using the DeepDerm Inception V3 model.

**Download the public model checkpoint** from [Zenodo](https://zenodo.org/records/10049217):

```bash
curl -L -o models/deepderm_isic.pth \
  "https://zenodo.org/api/records/10049217/files/deepderm_isic.pth/content"
```

```bash
python main.py \
  --model deepderm \
  --dataset ham10000 \
  --weights models/deepderm_isic.pth \
  --subset 200
```

---

### Audits of models trainend for the ModelAuditor publication
All pretrained models are available on [HuggingFace](https://huggingface.co/lukaskuhndkfz/ModelAuditor). Download them all at once:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
for name in ['camelyon17', 'chexpert', 'ham10000']:
    hf_hub_download('lukaskuhndkfz/ModelAuditor', f'{name}_resnet50_1_224.pt', local_dir='models')
"
```

Or individually:

```bash
huggingface-cli download lukaskuhndkfz/ModelAuditor camelyon17_resnet50_1_224.pt --local-dir models
huggingface-cli download lukaskuhndkfz/ModelAuditor chexpert_resnet50_1_224.pt --local-dir models
huggingface-cli download lukaskuhndkfz/ModelAuditor ham10000_resnet50_1_224.pt --local-dir models
```

---

### Camelyon17 (Histopathology)

**Task:** Binary patch-level metastasis detection.

**Dataset:** Download via the [WILDS](https://wilds.stanford.edu/datasets/) package:

```bash
pip install wilds
python -c "from wilds import get_dataset; get_dataset(dataset='camelyon17', download=True)"
```

**Split (matches paper):** The WILDS split is used, but only 2 of the 3 training hospitals are used for training. The 3rd training hospital serves as the ID evaluation set. The held-out 4th hospital is the OOD test set.

**Run the audit:**

```bash
python main.py \
  --model resnet50 \
  --dataset camelyon17 \
  --weights models/camelyon17_resnet50_1_224.pt \
  --subset 200
```


---

### CheXpert → ChestX-ray14 (Chest Radiography)

**Task:** Multi-label classification of 5 pathologies (Atelectasis, Consolidation, Cardiomegaly, Pleural Effusion, Edema) from frontal AP chest X-rays.

**Dataset (ID):** Download [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) and extract into `data/chexpert/`. The training set is used with frontal AP images filtered. Uncertain labels (-1) are mapped to 0.

**Dataset (OOD):** [ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) serves as the OOD test set.

**Run the audit:**

```bash
python main.py \
  --model resnet50 \
  --dataset chexpert \
  --weights models/chexpert_resnet50_1_224.pt \
  --subset 200
```

---

### HAM10000 (Dermatology)

**Task:** Binary classification of skin lesions (benign keratosis vs. melanoma).

**Dataset:** Download from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T). See [Quick Start](#quick-start) for setup.

**Split (matches paper):** Partitioned by clinic of origin — Vienna (`vidir_modern`) for training/ID evaluation, Australia (`rosendahl`) for OOD testing.

**Run the audit:**

```bash
python main.py \
  --model resnet50 \
  --dataset ham10000 \
  --weights models/ham10000_resnet50_1_224.pt \
  --subset 200
```

---

### PolypGen (Colonoscopy — Segmentation)

**Task:** Polyp segmentation (pixel-level masks).

**Dataset:** [PolypGen](https://www.synapse.org/Synapse:syn26376615/wiki/613312) multi-center dataset. Place center data under `data/data_C{1..5}/` with subdirectories `images_C{N}/` and `masks_C{N}/`.

**Split (matches paper):** Train on C1–C4, OOD test on C5 (John Radcliffe Hospital, Oxford).

**Architecture:** U-Net

**Run the audit:**

```bash
python main.py \
  --task segmentation \
  --model unet \
  --dataset polyp-c5 \
  --weights models/unet_polyp.pth \
  --subset 100
```

### PolypGen (Colonoscopy — Detection)

**Task:** Polyp detection (bounding boxes).

**Dataset:** Same PolypGen data, with bounding box annotations in `bbox_C{N}/` subdirectories.

**Architecture:** Faster R-CNN (MobileNet V3 backbone)

**Run the audit:**

```bash
python main.py \
  --task detection \
  --model fasterrcnn \
  --dataset polyp-c5-detection \
  --weights models/fasterrcnn_polyp.pth \
  --subset 100
```

---

### Model training
Scripts for model training are available on [Hugging Face](https://huggingface.co/lukaskuhndkfz/ModelAuditor)

## Dataset Documentation

[DATASETS.md](DATASETS.md) contains comprehensive documentation for every dataset used in the paper (Camelyon17, CheXpert, ChestX-ray14, HAM10000, PolypGen, SIIM-ISIC, Fitzpatrick17k), including:

- Download URLs and access requirements
- Expected directory structures
- Exact preprocessing pipelines (resize, normalization, label encoding)
- Split definitions with code
- Class distributions per split
- Known issues and quirks

---

## Configuration

### Switching the LLM backbone

Edit the `MODEL` constant at the top of `main.py` (line 36):

```python
# Anthropic (default)
MODEL = "anthropic/claude-sonnet-4-5-20250929"

# OpenAI
MODEL = "gpt-5.2-2025-12-11"
```

The `Agent` class auto-detects the provider from the model string prefix. Models prefixed with `anthropic/` use the Anthropic API; all others use the OpenAI API. No other code changes are needed.


### Usage

```bash
python main.py --model <architecture> --dataset <dataset> --weights <path> [options]
```

| Argument         | Required | Default          | Description                                      |
| ---------------- | -------- | ---------------- | ------------------------------------------------ |
| `--model`        | yes      | `resnet18`       | Model architecture (see below)                   |
| `--dataset`      | yes      | `CIFAR10`        | Dataset name or path (see below)                 |
| `--weights`      | yes      | —                | Path to model weights (`.pth` or `.onnx`)        |
| `--task`         | no       | `classification` | `classification`, `segmentation`, or `detection` |
| `--subset`       | no       | `50`             | Number of samples to evaluate                    |
| `--device`       | no       | auto             | `cpu`, `cuda`, or `mps`                          |
| `--single-agent` | no       | —                | Use single agent instead of multi-agent debate   |
| `--no-debate`    | no       | —                | Hide debate details from output                  |

**Supported models (`--model`):**
- Any torchvision classification model: `resnet18`, `resnet50`, `resnet152`, `vgg19`, `vit_b_16`, etc.
- Segmentation: `unet`
- Detection: `fasterrcnn`
- Pretrained checkpoints: `siim-isic`, `deepderm`
- Any ONNX model: `other` (with a `.onnx` weights file)

**Supported datasets (`--dataset`):**
- Built-in: `CIFAR10`, `CIFAR100`, `camelyon17`, `chexpert`, `ham10000`, `isic`, `kvasir-seg`, `polyp-c5`, `polyp-c5-detection`
- MedMNIST: `bloodmnist`, `dermamnist`, `octmnist`, `pneumoniamnist`, `retinamnist`, `chestmnist`
- **Custom datasets:** pass a folder path organized as an [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html):
  ```
  my_dataset/
  ├── class_a/
  │   ├── img001.jpg
  │   └── img002.jpg
  └── class_b/
      ├── img003.jpg
      └── img004.jpg
  ```

All images are resized to 224 × 224 and normalized with ImageNet statistics before evaluation.

### Audit workflow (three phases)

1. **Phase 1 — Metric & Shift Selection:** The agent conducts a Q&A with the practitioner, then two debate agents (Proposer + Critic) and a Moderator reach consensus on which metrics and shifts to evaluate, using XML-tagged structured output.
2. **Phase 2 — Iterative Audit:** The auditor runs perturbation-evaluation cycles. An interpretation agent analyzes results and may request additional rounds with new metrics/shifts (up to `MAX_AUDIT_ROUNDS`).
3. **Phase 3 — Report Generation:** A report agent synthesizes all results into a 5-section structured report (Executive Summary, Detailed Findings, Real-World Impact, Actionable Recommendations, Test Conditions Summary) with torchvision v2 code snippets.

---

## Citation

If you use ModelAuditor in your research, please cite:

```bibtex
@article{kuhn2025modelauditor,
  title={An autonomous agent for auditing and improving the reliability of clinical AI models},
  author={Kuhn, Lukas and Buettner, Florian},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. See [License.md](License.md) for details.
