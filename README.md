# ModelAuditor Agent

AI-powered model auditing agent with multi-agent debate for robust evaluation of machine learning models.

## Setup

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

## Usage

### Basic Usage
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