"""
Fine-tuning script for Vietnamese LLM with LoRA.
"""

import os
import logging
from typing import Dict, List, Optional, Union

import torch
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import load_dataset, Dataset
import wandb

from config.model_config import ModelConfig, DEFAULT_CONFIG
from src.model.model_loader import load_base_model, prepare_model_for_training

logger = logging.getLogger(__name__)


def format_chat_dataset(examples, tokenizer, max_length=2048):
    """
    Format chat dataset into prompt-response pairs.
    
    Args:
        examples: Dataset examples
        tokenizer: Model tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Formatted examples
    """
    # This is a simplified example - adjust based on your dataset format
    prompts = []
    for messages in examples["messages"]:
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt += f"Người dùng: {content}\n"
            elif role == "assistant":
                prompt += f"Trợ lý: {content}\n"
            elif role == "system":
                prompt += f"Hệ thống: {content}\n"
        prompts.append(prompt)
    
    # Tokenize
    tokenized = tokenizer(
        prompts, 
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # Set labels equal to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def prepare_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 2048,
    split: str = "train",
):
    """
    Prepare dataset for fine-tuning.
    
    Args:
        data_path: Path to the dataset
        tokenizer: Model tokenizer
        max_length: Maximum sequence length
        split: Dataset split to use
        
    Returns:
        Processed dataset
    """
    logger.info(f"Loading dataset from {data_path}")
    
    # Load dataset - adjust format based on your data
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path, split=split)
    else:
        dataset = load_dataset(data_path, split=split)
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Process the dataset
    processed_dataset = dataset.map(
        lambda examples: format_chat_dataset(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return processed_dataset


def train(
    config: ModelConfig = DEFAULT_CONFIG,
    train_data_path: str = "/Users/nguyen/Documents/Work/agentic/data/processed/train.json",
    eval_data_path: Optional[str] = "/Users/nguyen/Documents/Work/agentic/data/processed/eval.json",
    output_dir: str = "/Users/nguyen/Documents/Work/agentic/models/vietnamese-llm-lora",
):
    """
    Fine-tune the Vietnamese LLM with LoRA.
    
    Args:
        config: Model configuration
        train_data_path: Path to training data
        eval_data_path: Path to evaluation data
        output_dir: Directory to save the model
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize wandb if enabled
    if config.training_config.use_wandb:
        wandb.init(project="vietnamese-llm-lora")
    
    # Load model and tokenizer
    model, tokenizer = load_base_model(config)
    
    # Prepare model for LoRA fine-tuning
    model = prepare_model_for_training(model, config)
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_data_path, tokenizer, config.training_config.max_seq_length)
    eval_dataset = None
    if eval_data_path:
        eval_dataset = prepare_dataset(eval_data_path, tokenizer, config.training_config.max_seq_length, split="validation")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.training_config.learning_rate,
        num_train_epochs=config.training_config.num_train_epochs,
        per_device_train_batch_size=config.training_config.per_device_train_batch_size,
        per_device_eval_batch_size=config.training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training_config.gradient_accumulation_steps,
        gradient_checkpointing=config.training_config.gradient_checkpointing,
        max_grad_norm=config.training_config.max_grad_norm,
        weight_decay=config.training_config.weight_decay,
        warmup_ratio=config.training_config.warmup_ratio,
        lr_scheduler_type=config.training_config.lr_scheduler_type,
        evaluation_strategy=config.training_config.evaluation_strategy if eval_dataset else "no",
        eval_steps=config.training_config.eval_steps if eval_dataset else None,
        save_strategy=config.training_config.save_strategy,
        save_steps=config.training_config.save_steps,
        logging_steps=config.training_config.logging_steps,
        report_to="wandb" if config.training_config.use_wandb else "none",
        fp16=True,
        bf16=False,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training completed")
    
    return model, tokenizer