#!/usr/bin/env python
"""
Script to run inference with a fine-tuned Vietnamese LLM.
"""

import argparse
import logging
import os
import sys
import json

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
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Vietnamese LLM")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model",
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to a JSON file containing messages for inference",
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
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
    
    if args.input_file:
        # Run inference on input file
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        messages = data.get("messages", [])
        response = inference.generate(messages)
        
        print(f"Response: {response}")
    
    elif args.interactive:
        # Run in interactive mode
        print("Interactive mode. Type 'exit' to quit.")
        
        messages = [
            {"role": "system", "content": "Bạn là một trợ lý AI thông minh, hữu ích và thân thiện. Bạn trả lời bằng tiếng Việt."}
        ]
        
        while True:
            user_input = input("\nUser: ")
            if user_input.lower() == "exit":
                break
            
            messages.append({"role": "user", "content": user_input})
            
            response = inference.generate(messages)
            print(f"\nAssistant: {response}")
            
            messages.append({"role": "assistant", "content": response})
    
    else:
        logger.error("Either --input_file or --interactive must be specified")
        sys.exit(1)


if __name__ == "__main__":
    main()