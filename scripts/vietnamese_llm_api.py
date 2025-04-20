from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

app = FastAPI(title="Vietnamese LLM API")

# Load the model
model_path = "/Users/nguyen/Documents/Work/agentic/models/vietnamese-llm-lora-a100"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

class ChatRequest(BaseModel):
    system_prompt: str = "Bạn là trợ lý AI hữu ích."
    user_message: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Format the prompt
        prompt = f"<|system|>\n{request.system_prompt}\n<|user|>\n{request.user_message}\n<|assistant|>\n"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )
        
        # Decode and return the response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        assistant_response = full_response.split("<|assistant|>\n")[-1].strip()
        
        return ChatResponse(response=assistant_response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)