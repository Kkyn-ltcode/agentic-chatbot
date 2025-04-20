"""
Inference utilities for the Vietnamese LLM.
"""

import logging
from typing import Dict, List, Optional, Union

import torch
from transformers import GenerationConfig

from src.model.model_loader import load_model_for_inference

logger = logging.getLogger(__name__)


class VietnamLLMInference:
    """Inference class for Vietnamese LLM."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ):
        """
        Initialize the inference class.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to load the model on
            load_in_8bit: Whether to load in 8-bit precision
            load_in_4bit: Whether to load in 4-bit precision
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty parameter
        """
        self.model, self.tokenizer = load_model_for_inference(
            model_path=model_path,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        
        # Set generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        logger.info(f"Initialized inference with model from {model_path}")
    
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into a prompt for the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt += f"Hệ thống: {content}\n"
            elif role == "user":
                prompt += f"Người dùng: {content}\n"
            elif role == "assistant":
                prompt += f"Trợ lý: {content}\n"
        
        # Add the assistant prefix for the response
        prompt += "Trợ lý: "
        
        return prompt
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate a response for the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            generation_config: Optional custom generation config
            
        Returns:
            Generated response text
        """
        # Format the prompt
        prompt = self.format_prompt(messages)
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config or self.generation_config,
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        response = generated_text[len(prompt):]
        
        return response