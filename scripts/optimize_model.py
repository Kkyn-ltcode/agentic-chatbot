import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

# Path to your fine-tuned model
model_path = "/Users/nguyen/Documents/Work/agentic/models/vietnamese-llm-lora-a100"
output_path = "/Users/nguyen/Documents/Work/agentic/models/vietnamese-llm-optimized"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Convert to ONNX format for faster inference
ort_model = ORTModelForCausalLM.from_pretrained(
    model_path,
    export=True,
    provider="CUDAExecutionProvider"
)

# Save the optimized model
ort_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"Model optimized and saved to {output_path}")