"""
Configuration for the Vietnamese LLM model and fine-tuning.
"""

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 8  # Rank of the update matrices
    lora_alpha: int = 16  # Alpha parameter for LoRA scaling
    lora_dropout: float = 0.05  # Dropout probability for LoRA layers
    bias: str = "none"  # Bias type
    task_type: str = "CAUSAL_LM"  # Task type
    target_modules: List[str] = None  # Which modules to apply LoRA to


@dataclass
class TrainingConfig:
    """Configuration for training the model."""
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    logging_steps: int = 10
    max_seq_length: int = 2048
    packing: bool = False
    use_wandb: bool = True


@dataclass
class ModelConfig:
    """Configuration for the model."""
    model_name_or_path: str = "vinai/PhoGPT-7B5"  # or "bkai-foundation-models/vinallama-7b-chat"
    use_lora: bool = True
    lora_config: LoRAConfig = LoRAConfig()
    training_config: TrainingConfig = TrainingConfig()
    quantization: str = "4bit"  # "4bit", "8bit", or None
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    use_flash_attention: bool = True
    trust_remote_code: bool = True
    use_auth_token: bool = False
    device_map: str = "auto"
    

# Default configuration
DEFAULT_CONFIG = ModelConfig(
    model_name_or_path="vinai/PhoGPT-7B5",
    lora_config=LoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"]
    ),
    training_config=TrainingConfig(
        learning_rate=2e-4,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8
    )
)