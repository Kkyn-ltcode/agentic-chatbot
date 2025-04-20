import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Models to compare
models = [
    "/Users/nguyen/Documents/Work/agentic/models/vietnamese-llm-lora-a100",
    "vinai/PhoGPT-7B5",
    # Add other models for comparison
]

# Test prompts
test_prompts = [
    "<|user|>\nXin chào, bạn là ai?",
    "<|user|>\nCho tôi biết về lịch sử Việt Nam.",
    "<|user|>\nLàm thế nào để nấu phở?",
    # Add more test prompts
]

results = []

for model_name in models:
    print(f"Testing model: {model_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    for prompt in test_prompts:
        # Measure generation time
        start_time = time.time()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        generation_time = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            "model": model_name,
            "prompt": prompt,
            "response": response,
            "generation_time": generation_time
        })

# Create a DataFrame and save results
df = pd.DataFrame(results)
df.to_csv("/Users/nguyen/Documents/Work/agentic/benchmark_results.csv", index=False)
print("Benchmark completed and results saved.")