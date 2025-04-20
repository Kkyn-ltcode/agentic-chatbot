# Vietnamese Language Model Fine-tuning

This project contains scripts and tools for fine-tuning Vietnamese language models using LoRA (Low-Rank Adaptation) on MacBook Pro M1.

## Project Structure
.
├── config/                # Configuration files
├── data/                  # Data directory
│   ├── raw/               # Raw data
│   └── processed/         # Processed data
├── models/                # Model directory
│   └── cache/             # Model cache
├── scripts/               # Scripts for data processing and training
└── README.md              # This file

## Scripts

- `scripts/prepare_vietnamese_data.py`: Prepares Vietnamese data for fine-tuning
- `scripts/finetune_vietnamese_llm_m1.py`: Fine-tunes a Vietnamese LLM on MacBook Pro M1
- `scripts/finetune_vietnamese_llm_a100.py`: Fine-tunes a Vietnamese LLM on A100 GPUs

## Usage

### Download a model

```bash
python scripts/finetune_vietnamese_llm_m1.py \
    --model_size small \
    --train_data_path data/processed/train.json \
    --eval_data_path data/processed/eval.json \
    --cache_dir models/cache \
    --download_only