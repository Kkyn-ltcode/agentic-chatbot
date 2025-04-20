import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
model_path = "/Users/nguyen/Documents/Work/agentic/models/vietnamese-llm-lora-a100"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

def chat(message, history):
    # Format the prompt with history
    prompt = "<|system|>\nBạn là trợ lý AI hữu ích.\n"
    
    # Add conversation history
    for user_msg, assistant_msg in history:
        prompt += f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}\n"
    
    # Add the current message
    prompt += f"<|user|>\n{message}\n<|assistant|>\n"
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Decode the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response from the last turn
    response_parts = full_response.split("<|assistant|>\n")
    assistant_response = response_parts[-1].strip()
    
    return assistant_response

# Create the Gradio interface
demo = gr.ChatInterface(
    chat,
    title="Vietnamese Language Model Demo",
    description="Chat with the fine-tuned Vietnamese language model",
    theme="soft",
    examples=[
        "Xin chào, bạn là ai?",
        "Cho tôi biết về lịch sử Việt Nam.",
        "Làm thế nào để nấu phở?",
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)