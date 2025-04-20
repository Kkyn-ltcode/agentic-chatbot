#!/usr/bin/env python
"""
Gradio demo for the Vietnamese LLM.
"""

import argparse
import logging
import os
import sys

import gradio as gr

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.inference import VietnamLLMInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a Gradio demo for the Vietnamese LLM")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model",
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )
    
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization",
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a publicly shareable link for the demo",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Initialize inference
    inference = VietnamLLMInference(
        model_path=args.model_path,
        load_in_8bit=args.use_8bit,
        load_in_4bit=args.use_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    # Initialize chat history
    system_message = {"role": "system", "content": "Bạn là một trợ lý AI thông minh, hữu ích và thân thiện. Bạn trả lời bằng tiếng Việt."}
    
    def predict(message, history):
        """Generate a response for the given message and history."""
        messages = [system_message]
        
        # Convert Gradio history to our format
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:  # Skip None values
                messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add the current message
        messages.append({"role": "user", "content": message})
        
        # Generate response
        response = inference.generate(messages)
        
        return response
    
    # Create Gradio interface
    demo = gr.ChatInterface(
        fn=predict,
        title="Vietnamese LLM Demo",
        description="Chat with a Vietnamese language model fine-tuned with LoRA.",
        examples=[
            "Xin chào, bạn là ai?",
            "Hãy giải thích về trí tuệ nhân tạo bằng tiếng Việt.",
            "Viết một bài thơ ngắn về Hà Nội.",
            "Làm thế nào để nấu phở?",
        ],
        theme="soft",
    )
    
    # Launch the demo
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()