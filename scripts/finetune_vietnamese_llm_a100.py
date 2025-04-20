#!/usr/bin/env python
"""
Script to fine-tune a Vietnamese language model using LoRA.
Optimized for 8xA100 NVIDIA GPUs.
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
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

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
        default="/Users/nguyen/Documents/Work/agentic/models/vietnamese-llm-lora",
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
        default=8,
        help="Batch size per device during training",
    )
    
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size per device during evaluation",
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA attention dimension",
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
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
        "--deepspeed",
        type=str,
        default=None,
        help="Path to deepspeed config file",
    )
    
    parser.add_argument(
        "--fsdp",
        type=str,
        default=None,
        choices=["full_shard", "shard_grad_op", "offload"],
        help="Fully Sharded Data Parallelism configuration",
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
        batch_size=64,  # Larger batch size for A100s
        num_proc=16,    # Use multiple processes for A100s
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing datasets",
    )
    
    return tokenized_datasets


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. This script is optimized for 8xA100 GPUs.")
    else:
        logger.info(f"Found {torch.cuda.device_count()} CUDA devices")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
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
            bnb_4bit_compute_dtype=torch.float16,
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto",  # Will distribute across multiple GPUs
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for A100s
    )
    
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
    
    # Configure FSDP if requested
    fsdp_config = None
    if args.fsdp:
        logger.info(f"Using FSDP with configuration: {args.fsdp}")
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )
        fsdp_config = {
            "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
            "xla": False,
            "fsdp_offload_params": args.fsdp == "offload",
        }
    
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
        # DeepSpeed optimizations
        fp16=args.deepspeed is not None,  # Let DeepSpeed config handle this
        bf16=args.deepspeed is not None and "bf16" in open(args.deepspeed).read(),
        optim="adamw_torch_fused" if args.deepspeed is None else "adamw_torch",  # Compatible with DeepSpeed
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed,
        # Disable FSDP if using DeepSpeed
        fsdp=None if args.deepspeed else args.fsdp,
        fsdp_config=None if args.deepspeed else fsdp_config,
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