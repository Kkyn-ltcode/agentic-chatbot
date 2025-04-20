#!/usr/bin/env python
"""
Script to fine-tune a Vietnamese language model using LoRA.
Optimized for MacBook Pro M1.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a Vietnamese language model")
    
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="vinai/PhoGPT-7B5",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    
    # Add a new argument for model size selection
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "medium", "large"],
        default="small",
        help="Size of the model to use: small (1-2B), medium (3-4B), or large (7B+)",
    )
    
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the training data JSON file",
    )
    
    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help="Path to the evaluation data JSON file",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/nguyen/Documents/Work/agentic/models/vietnamese-llm-lora-m1",
        help="Directory to save the fine-tuned model",
    )
    
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device during training",
    )
    
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size per device during evaluation",
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA attention dimension",
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter",
    )
    
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability",
    )
    
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization",
    )
    
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume from last checkpoint",
    )
    
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Only download the model without fine-tuning",
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/Users/nguyen/Documents/Work/agentic/models/cache",
        help="Directory to cache the downloaded model",
    )
    
    return parser.parse_args()


def prepare_datasets(train_path, eval_path, tokenizer, max_length):
    """Prepare datasets for training and evaluation."""
    # Load datasets
    data_files = {
        "train": train_path,
        "validation": eval_path,
    }
    
    raw_datasets = load_dataset("json", data_files=data_files)
    
    # Tokenization function
    def tokenize_function(examples):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        
        for conversation in examples["messages"]:
            # Format the conversation
            formatted_text = ""
            for message in conversation:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    formatted_text += f"<|system|>\n{content}\n"
                elif role == "user":
                    formatted_text += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    formatted_text += f"<|assistant|>\n{content}\n"
            
            # Tokenize
            tokenized = tokenizer(
                formatted_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            all_input_ids.append(tokenized["input_ids"][0])
            all_attention_mask.append(tokenized["attention_mask"][0])
            all_labels.append(tokenized["input_ids"][0].clone())
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }
    
    # Apply tokenization
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        batch_size=16,  # Smaller batch size for M1
        num_proc=1,     # Use single process to avoid memory issues on M1
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing datasets",
    )
    
    return tokenized_datasets


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create cache directory if it doesn't exist
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Select model based on size
    if args.model_name_or_path == "vinai/PhoGPT-7B5" and args.model_size != "large":
        if args.model_size == "small":
            # Smaller Vietnamese models
            model_name = "NlpHUST/gpt-neo-1.3B-vietnamese"  # or "NlpHUST/gpt-neo-1.3B-vietnamese"
            logger.info(f"Using smaller model: {model_name}")
            args.model_name_or_path = model_name
        elif args.model_size == "medium":
            model_name = "vinai/PhoGPT-4B"
            logger.info(f"Using medium-sized model: {model_name}")
            args.model_name_or_path = model_name
    
    # Check if MPS is available (for M1 Macs)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    
    # Add special tokens if they don't exist
    special_tokens = ["<|system|>", "<|user|>", "<|assistant|>"]
    special_tokens_dict = {}
    
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "[PAD]"
    
    if any(token not in tokenizer.get_vocab() for token in special_tokens):
        special_tokens_dict["additional_special_tokens"] = special_tokens
    
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
    
    # Prepare quantization config
    quantization_config = None
    if args.use_4bit:
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,  # Use float32 for M1
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.use_8bit:
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load model
    logger.info(f"Loading model from {args.model_name_or_path}")
    logger.info(f"Caching model to {args.cache_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for M1
        cache_dir=args.cache_dir,
    )
    
    # If download_only flag is set, save the model and exit
    if args.download_only:
        download_dir = os.path.join(args.cache_dir, "downloaded_model")
        os.makedirs(download_dir, exist_ok=True)
        logger.info(f"Saving downloaded model to {download_dir}")
        model.save_pretrained(download_dir)
        tokenizer.save_pretrained(download_dir)
        logger.info("Model and tokenizer downloaded and saved successfully")
        return
    
    # Prepare model for k-bit training if using quantization
    if args.use_4bit or args.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    logger.info("Configuring LoRA")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Fewer target modules for M1
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare datasets
    logger.info("Preparing datasets")
    tokenized_datasets = prepare_datasets(
        args.train_data_path,
        args.eval_data_path,
        tokenizer,
        args.max_seq_length,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="longest",
        return_tensors="pt",
    )
    
    # Check for last checkpoint
    last_checkpoint = None
    if args.resume_from_checkpoint:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and os.path.exists(args.output_dir):
            logger.warning(
                f"Output directory ({args.output_dir}) exists but no checkpoint found. "
                "Training will start from scratch."
            )
        elif last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training from {last_checkpoint}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=False,  # Disable fp16 for M1
        bf16=False,  # Disable bf16 for M1
        optim="adamw_torch",  # Use standard optimizer for M1
        report_to="none",  # Disable reporting to save memory
        ddp_find_unused_parameters=False,
        use_mps_device=torch.backends.mps.is_available(),  # Enable MPS for M1
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("Training completed")


if __name__ == "__main__":
    main()