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
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
try:
    load_dotenv()
except ImportError:
    pass

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
        default=258,
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
    
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for experiment tracking",
    )
    
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="vietnamese-llm-finetuning",
        help="Weights & Biases project name",
    )
    
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type",
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for the learning rate scheduler",
    )
    
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of evaluation calls with no improvement after which training will be stopped",
    )
    
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Use mixed precision training (only enable if your device supports it)",
    )
    
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Limit the total amount of checkpoints. Deletes the older checkpoints.",
    )
    
    parser.add_argument(
        "--validation_split_percentage",
        type=int,
        default=None,
        help="If set, will use this percentage of training data as validation if no validation file is provided",
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
            assistant_positions = []  # Track where assistant responses begin
            
            for message in conversation:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    formatted_text += f"<|system|>\n{content}\n"
                elif role == "user":
                    formatted_text += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    # Mark the position where assistant response begins
                    assistant_positions.append(len(formatted_text))
                    formatted_text += f"<|assistant|>\n{content}\n"
            
            # Tokenize
            tokenized = tokenizer(
                formatted_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt",
            )
            
            # Handle padding manually to ensure consistent sizes
            input_ids = tokenized["input_ids"][0]
            attention_mask = tokenized["attention_mask"][0]
            
            # Create labels: -100 for tokens we don't want to predict (system/user messages)
            # and actual token IDs for tokens we want to predict (assistant responses)
            labels = torch.ones_like(input_ids) * -100  # Initialize all as -100 (ignored in loss)
            
            # Find token positions for assistant responses
            for pos in assistant_positions:
                # Convert character position to token position (approximate)
                token_pos = tokenizer(formatted_text[:pos], return_tensors="pt")["input_ids"].shape[1]
                if token_pos < len(labels):
                    # Set all tokens after this position to their actual values (for prediction)
                    labels[token_pos:] = input_ids[token_pos:]
            
            # Pad or truncate to exactly max_length
            if len(input_ids) < max_length:
                # Pad
                padding_length = max_length - len(input_ids)
                input_ids = torch.cat([input_ids, torch.ones(padding_length, dtype=torch.long) * tokenizer.pad_token_id])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
                labels = torch.cat([labels, torch.ones(padding_length, dtype=torch.long) * -100])  # Pad labels with -100
            elif len(input_ids) > max_length:
                # Truncate
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]
            
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
        
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
            model_name = "vinai/phobert-base"  # or "NlpHUST/gpt-neo-1.3B-vietnamese"
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
    
    # Determine target modules based on model architecture
    target_modules = []
    model_type = model.config.model_type if hasattr(model.config, "model_type") else ""
    
    if "gpt" in model_type.lower() or "phogpt" in args.model_name_or_path.lower():
        # For GPT-like models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        logger.info(f"Using target modules for GPT architecture: {target_modules}")
    elif "bert" in model_type.lower() or "phobert" in args.model_name_or_path.lower():
        # For BERT-like models
        target_modules = ["query", "key", "value", "output.dense"]
        logger.info(f"Using target modules for BERT architecture: {target_modules}")
    else:
        # Fallback to a more general approach
        # Try to find attention modules by inspecting model
        for name, _ in model.named_modules():
            if any(keyword in name for keyword in ["attention", "self", "query", "key", "value"]):
                if name.endswith(("query", "key", "value", "dense")):
                    target_modules.append(name)
        
        if not target_modules:
            # If no modules found, use a default set that works for many models
            target_modules = ["query", "key", "value", "dense"]
        
        logger.info(f"Using automatically detected target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM if "gpt" in model_type.lower() else TaskType.SEQ_CLS,
        target_modules=target_modules,
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Make sure trainable parameters require gradients
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad_(True)
    
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
    
    # Setup wandb if requested
    if args.use_wandb:
        try:
            import wandb
            wandb_available = True
            
            # Initialize wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            logger.info("Weights & Biases initialized successfully")
        except ImportError:
            wandb_available = False
            logger.warning("Weights & Biases not available. Install with: pip install wandb")
    else:
        wandb_available = False
    
    # Training arguments with improved settings
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        eval_strategy="epoch",  # Changed from eval_strategy to evaluation_strategy
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=args.mixed_precision == "fp16",
        bf16=args.mixed_precision == "bf16",
        optim="adamw_torch",
        report_to="wandb" if wandb_available else "none",
        ddp_find_unused_parameters=False,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        # Add a specific run_name to avoid the warning
        run_name=args.wandb_run_name or f"vietnamese-llm-{args.model_size}-{os.path.basename(args.output_dir)}",
    )
    
    # Add early stopping callback
    callbacks = []
    if args.early_stopping_patience > 0:
        from transformers.trainer_callback import EarlyStoppingCallback
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )
    
    # Initialize Trainer with callbacks and label_names
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        # Add label_names to fix the warning
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
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
    
    # Clean up wandb
    if wandb_available:
        wandb.finish()
    
    logger.info("Training completed")


if __name__ == "__main__":
    main()