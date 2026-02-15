# ModelAuditor Agent

AI-powered model auditing agent with multi-agent debate for robust evaluation of machine learning models.

## Setup

This repository has been tested extensively with Python 3.10.15. Typical install time via uv is less than a minute.

### Using uv (recommended)
```bash
uv sync
uv run python main.py --model resnet50 --dataset CIFAR10 --weights path/to/weights.pth
```

### Using pip
```bash
pip install -e .
python main.py --model resnet50 --dataset CIFAR10 --weights path/to/weights.pth
```

### Medical AI dependencies (optional)
```bash
uv sync --extra medical  # or pip install -e ".[medical]"
```

## Data and Model Downloads

### Pre-trained Models (HuggingFace)

Download pre-trained ResNet50 models from [HuggingFace](https://huggingface.co/lukaskuhndkfz/ModelAuditor):

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download all models
python -c "from huggingface_hub import hf_hub_download; [hf_hub_download('lukaskuhndkfz/ModelAuditor', f'{name}_resnet50_1_224.pt', local_dir='models') for name in ['camelyon17', 'chexpert', 'ham10000']]"
```

Or download individually:
```bash
# Camelyon17 (pathology)
huggingface-cli download lukaskuhndkfz/ModelAuditor camelyon17_resnet50_1_224.pt --local-dir models

# CheXpert (chest X-ray)
huggingface-cli download lukaskuhndkfz/ModelAuditor chexpert_resnet50_1_224.pt --local-dir models

# HAM10000 (dermatology)
huggingface-cli download lukaskuhndkfz/ModelAuditor ham10000_resnet50_1_224.pt --local-dir models
```

### HAM10000 Dataset

1. **Download the dataset** from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T):
   - `HAM10000_images_part_1.zip`
   - `HAM10000_images_part_2.zip`
   - `HAM10000_metadata.tab` as CSV

2. **Extract and organize**:
```bash
# Create data directory
mkdir -p data/ham10000

# Extract images (both parts) into data/ham10000/
unzip HAM10000_images_part_1.zip -d data/ham10000/
unzip HAM10000_images_part_2.zip -d data/ham10000/

# Copy metadata
cp HAM10000_metadata.csv data/ham10000/
```

3. **Split into vidir_modern and rosendahl subsets**:
```bash
python setup_ham10000.py
```

This creates the following structure:
```
data/ham10000/
├── vidir_modern/
│   ├── bkl/  (475 images)
│   └── mel/  (680 images)
└── rosendahl/
    ├── bkl/  (490 images)
    └── mel/  (342 images)
```

## Usage

### General Usage
```bash
python main.py --model resnet50 --dataset CIFAR10 --weights models/model.pth
```

### Medical Models
```bash
# ISIC skin lesion classification
python main.py --model siim-isic --dataset isic --weights models/isic/model.pth

# HAM10000 dataset
python main.py --model deepderm --dataset ham10000 --weights models/ham10000.pth
```

### Medical Example (DeepDerm)

The auditor can be tested with the DeepDerm classifier which can be downloaded [here](https://zenodo.org/records/10049217). All that is needed is a valid Anthropic API Key as can be seen below (see section 'Environment Variables') as well as a test dataset such as [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

```bash
python main.py --model deepderm --dataset ham10000 --weights models/deepderm_isic.pth
```

Alternatively, use our pre-trained ResNet50 model from HuggingFace (see "Data and Model Downloads" above):
```bash
python main.py --model resnet50 --dataset ham10000 --weights models/ham10000_resnet50_1_224.pt
```

Expected runtime varies depending on user response speed and subset time but should take less than 10 minutes in total.

### Toy Example (CIFAR10)

We also prepared a small toy model, trained on CIFAR10 so the Auditor can be evaluated on natural images. All that is needed is a valid Anthropic API Key as can be seen below (see section 'Environment Variables').

```bash
python main.py --model resnet18 --dataset CIFAR10 --weights examples/cifar10/cifar10.pth
```

Expected runtime varies depending on user response speed and subset time but should take less than 10 minutes in total.


### Options
- `--subset N`: Use N samples for faster evaluation
- `--no-debate`: Disable multi-agent debate
- `--single-agent`: Use single agent instead of multi-agent debate
- `--device`: Specify device (cpu, cuda, mps)

## Environment Variables

Set your API keys:
```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"  # if using non-Anthropic models
```

## Project Structure

- `main.py` - Interactive model auditor with multi-agent debate
- `testbench.py` - Automated evaluation script
- `utils/agent.py` - Multi-agent conversation system
- `architectures/` - Custom model architectures
- `prompts/` - System prompts for different evaluation phases
- `models/` - Pre-trained model weights
- `results/` - Evaluation results and conversation logs