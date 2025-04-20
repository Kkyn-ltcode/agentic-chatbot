"""
Utilities for loading and preparing the Vietnamese LLM model.
"""

import os
import logging
from typing import Dict, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

from config.model_config import ModelConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def load_base_model(config: ModelConfig = DEFAULT_CONFIG):
    """
    Load the base Vietnamese LLM model with optional quantization.
    
    Args:
        config: Model configuration
        
    Returns:
        model: The loaded model
        tokenizer: The model's tokenizer
    """
    logger.info(f"Loading model: {config.model_name_or_path}")
    
    # Configure quantization if needed
    quantization_config = None
    if config.quantization == "4bit" and config.load_in_4bit:
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif config.quantization == "8bit" and config.load_in_8bit:
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=quantization_config,
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
        use_auth_token=config.use_auth_token,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2" if config.use_flash_attention else "eager",
    )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        use_auth_token=config.use_auth_token,
    )
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def prepare_model_for_training(
    model, 
    config: ModelConfig = DEFAULT_CONFIG,
):
    """
    Prepare the model for LoRA fine-tuning.
    
    Args:
        model: The base model to prepare
        config: Model configuration
        
    Returns:
        model: The prepared model with LoRA adapters
    """
    if not config.use_lora:
        logger.info("LoRA not enabled, returning the base model")
        return model
    
    logger.info("Preparing model for LoRA fine-tuning")
    
    # Prepare model for k-bit training if using quantization
    if config.load_in_8bit or config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=config.training_config.gradient_checkpointing
        )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_config.r,
        lora_alpha=config.lora_config.lora_alpha,
        lora_dropout=config.lora_config.lora_dropout,
        bias=config.lora_config.bias,
        task_type=config.lora_config.task_type,
        target_modules=config.lora_config.target_modules,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    return model


def load_model_for_inference(
    model_path: str,
    device: str = "cuda",
    load_in_8bit: bool = False,
    load_in_4bit: bool = True,
):
    """
    Load a fine-tuned model for inference.
    
    Args:
        model_path: Path to the fine-tuned model
        device: Device to load the model on
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
        
    Returns:
        model: The loaded model
        tokenizer: The model's tokenizer
    """
    logger.info(f"Loading fine-tuned model from {model_path}")
    
    # Configure quantization
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else device,
        torch_dtype=torch.float16,
    )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer