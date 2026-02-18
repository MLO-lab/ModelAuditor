# Dataset Documentation

This document describes every dataset used in the ModelAuditor paper, including download
instructions, directory layout, preprocessing pipelines, exact split definitions, class
distributions, and known issues.

For automated setup, see:
- [`data/download_datasets.py`](data/download_datasets.py) — downloads datasets where programmatic access is available
- [`data/prepare_splits.py`](data/prepare_splits.py) — produces CSV manifests with the exact splits used in the paper

---

## Table of Contents

- [Camelyon17](#camelyon17)
- [CheXpert](#chexpert)
- [ChestX-ray14](#chestx-ray14)
- [HAM10000](#ham10000)
- [PolypGen](#polypgen)
- [SIIM-ISIC](#siim-isic)
- [Fitzpatrick17k](#fitzpatrick17k)
- [Preprocessing Reference](#preprocessing-reference)

---

## Camelyon17

**Task:** Binary patch-level metastasis detection (tumor vs. normal) in histopathology whole-slide images.

### Download

Available through the [WILDS](https://wilds.stanford.edu/datasets/) benchmark package.
Requires ~10 GB disk space.

```bash
pip install wilds
python -c "from wilds import get_dataset; get_dataset(dataset='camelyon17', download=True)"
```

The WILDS package downloads and caches the dataset automatically. Default location:
`~/.wilds/camelyon17_v1.0/`.

### Directory structure

```
~/.wilds/camelyon17_v1.0/
├── metadata.csv
└── patches/
    ├── patient_XXX_node_Y/
    │   ├── patch_*.png
    │   └── ...
    └── ...
```

### Preprocessing pipeline

| Step      | Detail                                                                   |
| --------- | ------------------------------------------------------------------------ |
| Resize    | 224 × 224 (`transforms.Resize(224)`)                                     |
| To tensor | `transforms.ToTensor()`                                                  |
| Normalize | `mean=[0.5], std=[0.5]` (single-channel style applied to all 3 channels) |

```python
transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

**Label encoding:** Binary integer — 0 = normal, 1 = tumor.

The WILDS loader returns `(image, label, metadata)` tuples. ModelAuditor wraps this with
`Camelyon17Wrapper` (see `main.py:629–642`) to return standard `(image, label)` pairs.

### Split definition


| Split         | Hospitals                                | Usage in paper                 |
| ------------- | ---------------------------------------- | ------------------------------ |
| Train         | 2 of the 3 WILDS training hospitals      | Model training                 |
| ID evaluation | 3rd WILDS training hospital              | In-distribution validation     |
| OOD test      | WILDS validation hospital (4th hospital) | Out-of-distribution evaluation |


In the ModelAuditor CLI, `--dataset camelyon17` loads the WILDS **validation** subset
(`dataset.get_subset('val')`) for auditing:

```python
dataset = get_dataset(dataset="camelyon17", download=True)
test_data = dataset.get_subset('val')
```


---

## CheXpert

**Task:** Multi-label classification of 5 chest pathologies from frontal AP radiographs.

### Download

1. Register and download from the [CheXpert competition page](https://stanfordmlgroup.github.io/competitions/chexpert/).
2. Download `CheXpert-v1.0-small.zip` (~11 GB).
3. Extract into `data/chexpert/`.

Access requires agreeing to the Stanford data use agreement.

### Directory structure

```
data/chexpert/
├── train.csv                    # Training set metadata
├── valid.csv                    # Validation set metadata
├── train/
│   ├── patient00001/
│   │   └── study1/
│   │       └── view1_frontal.jpg
│   └── ...
└── valid/
    └── ...
```

### Preprocessing pipeline

| Step      | Detail                               |
| --------- | ------------------------------------ |
| Resize    | 224 × 224 (`transforms.Resize(224)`) |
| To tensor | `transforms.ToTensor()`              |
| Normalize | `mean=[0.5], std=[0.5]`              |

```python
transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

**Filtering (applied during loading):**
1. Keep only **frontal** images: `df['Frontal/Lateral'] == 'Frontal'`
2. Keep only **AP** projection: `df['AP/PA'] == 'AP'`

**Label encoding:**
- **5 pathologies extracted:** Atelectasis, Consolidation, Cardiomegaly, Pleural Effusion, Edema
- NaN values mapped to 0 (absent)
- Uncertain labels (-1) mapped to 0 (absent)
- Each label is binary float (0.0 or 1.0)
- Output shape per sample: `torch.float32` tensor of length 5

```python
df = df.replace(np.nan, 0.0)
df = df.replace(-1.0, 0.0)
labels = df[['Atelectasis', 'Consolidation', 'Cardiomegaly', 'Pleural Effusion', 'Edema']].to_numpy()
```

### Split definition

- **Training / ID evaluation:** The CheXpert **training set** is used. For auditing, the last
  5,000 samples (after frontal AP filtering) are used as the evaluation subset:
  ```python
  test_data = Subset(dataset, range(len(dataset) - 5000, len(dataset)))
  ```
- **OOD test:** ChestX-ray14 (see [below](#chestx-ray14)).


---

## ChestX-ray14

**Task:** Out-of-distribution test set for models trained on CheXpert.

### Download

Download from [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) (~42 GB).
Requires a Kaggle account.

### Directory structure

```
data/chestxray14/
├── Data_Entry_2017.csv
└── images/
    ├── 00000001_000.png
    ├── 00000001_001.png
    └── ... (~112,000 images)
```

### Preprocessing pipeline

Same as CheXpert:

| Step      | Detail                  |
| --------- | ----------------------- |
| Resize    | 224 × 224               |
| To tensor | `transforms.ToTensor()` |
| Normalize | `mean=[0.5], std=[0.5]` |

**Label encoding:** Multi-label, using the subset of pathologies shared with CheXpert. Labels
are extracted from `Data_Entry_2017.csv` and encoded as binary vectors.

### Split definition

The entire ChestX-ray14 dataset serves as the OOD test set for CheXpert-trained models. No
train/val split is created from this dataset.

---

## HAM10000

**Task:** Binary classification of skin lesions — benign keratosis (bkl) vs. melanoma (mel).

### Download

Download from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) (~2.4 GB total):
- `HAM10000_images_part_1.zip`
- `HAM10000_images_part_2.zip`
- `HAM10000_metadata` (download as CSV, rename to `HAM10000_metadata.csv`)

No access agreement required; publicly available.

### Directory structure (after running `download_datasets.py --dataset ham10000`)

```
data/ham10000/
├── HAM10000_metadata.csv       # Original metadata
├── ISIC_0024306.jpg            # Raw images (from both zips)
├── ISIC_0024307.jpg
├── ...                         # (~10,015 images total)
├── vidir_modern/               # Vienna clinic subset (ID)
│   ├── bkl/  (475 images)     #   Benign keratosis
│   └── mel/  (680 images)     #   Melanoma
└── rosendahl/                  # Australia clinic subset (OOD)
    ├── bkl/  (490 images)     #   Benign keratosis
    └── mel/  (342 images)     #   Melanoma
```

### Setup commands

```bash
python data/download_datasets.py --dataset ham10000 --ham10000-zip-dir /path/to/downloads
```

### Preprocessing pipeline

| Step      | Detail                                                            |
| --------- | ----------------------------------------------------------------- |
| Resize    | 224 × 224 (`transforms.Resize(224)`)                              |
| To tensor | `transforms.ToTensor()`                                           |
| Normalize | ImageNet: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` |

```python
transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Label encoding:** Binary integer — `bkl` = 0 (benign), `mel` = 1 (malignant).

### Split definition

The split is **clinic-based**, using the `dataset` column in `HAM10000_metadata.csv`:

| Split           | Metadata filter             | Clinic               | Usage                                 |
| --------------- | --------------------------- | -------------------- | ------------------------------------- |
| Train / ID eval | `dataset == 'vidir_modern'` | Vienna, Austria      | Training + in-distribution evaluation |
| OOD test        | `dataset == 'rosendahl'`    | Rosendahl, Australia | Out-of-distribution evaluation        |



The `download_datasets.py` script reads the metadata, filters by `dataset` and `dx` columns, and
copies images into the directory structure shown above.


---

## PolypGen

**Task:** Polyp segmentation (pixel-level masks) and polyp detection (bounding boxes) in
colonoscopy images.

### Download

Download from [Synapse](https://www.synapse.org/Synapse:syn26376615/wiki/613312).
Requires a Synapse account and acceptance of the data use agreement.

Download the per-center data archives for C1 through C5 (and optionally C6).

### Directory structure

```
data/
├── data_C1/
│   ├── images_C1/              # Colonoscopy frames
│   │   ├── C1_000001.jpg
│   │   └── ...
│   ├── masks_C1/               # Segmentation masks
│   │   ├── C1_000001_mask.jpg
│   │   └── ...
│   └── bbox_C1/                # Bounding box annotations
│       ├── C1_000001_mask.txt
│       └── ...
├── data_C2/
│   ├── images_C2/
│   ├── masks_C2/
│   └── bbox_C2/
├── data_C3/ ...
├── data_C4/ ...
└── data_C5/                    # OOD test center (John Radcliffe Hospital, Oxford)
    ├── images_C5/
    ├── masks_C5/
    └── bbox_C5/
```

### Preprocessing pipeline

**Segmentation (U-Net):**

| Step            | Detail                                                            |
| --------------- | ----------------------------------------------------------------- |
| Image resize    | 128 × 128                                                         |
| Image normalize | ImageNet: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` |
| Mask resize     | 128 × 128                                                         |
| Mask binarize   | `(mask > 0.5).float()`                                            |

```python
img_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
mask_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
# After loading: mask = (mask > 0.5).float()
```

**Detection (Faster R-CNN with MobileNet V3):**

| Step            | Detail                                                   |
| --------------- | -------------------------------------------------------- |
| Image resize    | 320 × 320                                                |
| Image normalize | `transforms.ToTensor()` only (no explicit normalization) |
| Bbox scaling    | Bounding boxes scaled by resize ratio                    |

```python
det_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])
```

**Bounding box format:** Text files with one line per polyp:
```
polyp x_min y_min x_max y_max
```

Bounding boxes are scaled proportionally when images are resized. The detection loader outputs
Faster R-CNN-compatible targets:
```python
target = {
    "boxes": tensor([[x1, y1, x2, y2], ...]),  # float32
    "labels": tensor([1, 1, ...]),               # int64, all class 1 (polyp)
    "image_id": tensor([idx]),
    "area": tensor([...]),                       # (x2-x1) * (y2-y1)
    "iscrowd": tensor([0, 0, ...])
}
```

### Split definition

| Split    | Centers                              | Usage                          |
| -------- | ------------------------------------ | ------------------------------ |
| Train    | C1, C2, C3, C4                       | Model training                 |
| OOD test | C5 (John Radcliffe Hospital, Oxford) | Out-of-distribution evaluation |

### Mask naming convention

For an image file `{name}.jpg`, the corresponding mask is `{name}_mask.jpg` and the
bounding box file is `{name}_mask.txt`. The loader handles an edge case where image stems
end with `]` by trying an alternative mask path without the trailing bracket.



---

## SIIM-ISIC

**Task:** Melanoma classification using a pretrained EfficientNet ensemble.

### Download

Download the pretrained checkpoint from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10049216).

The ISIC evaluation dataset should be organized as an ImageFolder:

```
data/isic/
└── train/
    ├── benign/
    │   ├── ISIC_0000000.jpg
    │   └── ...
    └── malignant/
        ├── ISIC_0000001.jpg
        └── ...
```

ISIC image data can be obtained from the [ISIC Archive](https://challenge.isic-archive.com/data).

### Preprocessing pipeline

| Step      | Detail                                                                                                    |
| --------- | --------------------------------------------------------------------------------------------------------- |
| Resize    | 224 × 224                                                                                                 |
| To tensor | `transforms.ToTensor()`                                                                                   |
| Normalize | None at load time; the model internally rescales from (-1,1) to (0,1) then applies ImageNet normalization |

```python
# External transform (applied by data loader)
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Internal to SIIMISICClassifier.forward():
#   rescaled = x * 0.5 + 0.5        # (-1,1) → (0,1)
#   rescaled = normalize(rescaled)   # ImageNet mean/std
```

**Model architecture:** Ensemble of 3 EfficientNets (B7-NS, B6-NS, B5-NS) with 8-fold
test-time augmentation (horizontal flip, vertical flip, transpose, and combinations). The
ensemble averages melanoma probabilities across models and augmentations.

**Label encoding:** Binary — 0 = benign, 1 = malignant (melanoma). The model output is the
melanoma probability at index 6 of the EfficientNet output vector.

### Split definition

The pretrained checkpoint was trained externally (SIIM-ISIC Melanoma Classification competition).
ModelAuditor uses it as-is for auditing — no retraining is performed. The ISIC evaluation
dataset is used for the perturbation audit.

### Known issues

- The `SIIMISICClassifier` wrapper expects input in (-1,1) range but the external transform
  produces (0,1) range via `ToTensor()`. The model's internal rescaling (`x*0.5+0.5`) is
  designed for (-1,1) input; with (0,1) input the effective range becomes (0.5, 1.0). Ensure
  input is in the expected range if modifying transforms.
- The model uses stochastic test-time augmentation (`random.choice(self.augment_list)`),
  so predictions are non-deterministic across runs.

---

## Fitzpatrick17k

**Task:** Out-of-distribution validation set for the SIIM-ISIC melanoma classifier.

### Download

The Fitzpatrick17k dataset was published by Groh et al. (2021). Access instructions:
- Paper: [Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset](https://arxiv.org/abs/2104.09957)
- Dataset: Available upon request from the authors or via the [GitHub repository](https://github.com/mattgroh/fitzpatrick17k)

### Directory structure

```
data/fitzpatrick17k/
├── fitzpatrick17k.csv           # Metadata with Fitzpatrick skin type labels
└── images/
    ├── image_000001.jpg
    └── ...
```

### Preprocessing pipeline

Same ImageNet-standard pipeline as HAM10000:

| Step      | Detail                                                            |
| --------- | ----------------------------------------------------------------- |
| Resize    | 224 × 224                                                         |
| To tensor | `transforms.ToTensor()`                                           |
| Normalize | ImageNet: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` |

### Split definition

Used as an OOD test set for the SIIM-ISIC-trained melanoma classifier. The full dataset is
used for evaluation (no train/val split created from Fitzpatrick17k).



