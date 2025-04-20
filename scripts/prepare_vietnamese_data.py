#!/usr/bin/env python
"""
Script to prepare Vietnamese data for fine-tuning.
"""

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare Vietnamese data for fine-tuning")
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input data file",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/nguyen/Documents/Work/agentic/data/processed",
        help="Directory to save the processed data",
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data to use for training (rest for evaluation)",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["sharegpt", "openchat", "custom"],
        default="openchat",
        help="Format of the input data",
    )
    
    return parser.parse_args()


def convert_to_standard_format(data: List[Dict], format_type: str) -> List[Dict]:
    """
    Convert data to a standard format for fine-tuning.
    
    Args:
        data: Input data
        format_type: Format of the input data
        
    Returns:
        Standardized data
    """
    standardized_data = []
    
    if format_type == "sharegpt":
        for item in data:
            conversations = item.get("conversations", [])
            messages = []
            
            # Add system message if it exists
            if "system" in item:
                messages.append({"role": "system", "content": item["system"]})
            
            # Add conversations
            for conv in conversations:
                role = "assistant" if conv.get("from") == "gpt" else "user"
                messages.append({"role": role, "content": conv.get("value", "")})
            
            if messages:
                standardized_data.append({"messages": messages})
    
    elif format_type == "openchat":
        for item in data:
            messages = []
            
            # Add system message if it exists
            if "system" in item:
                messages.append({"role": "system", "content": item["system"]})
            
            # Add user message
            if "query" in item:
                messages.append({"role": "user", "content": item["query"]})
            
            # Add assistant message
            if "output" in item:
                messages.append({"role": "assistant", "content": item["output"]})
            
            if len(messages) >= 2:  # At least user and assistant messages
                standardized_data.append({"messages": messages})
    
    elif format_type == "custom":
        # Assuming custom format already has "messages" field with role/content structure
        for item in data:
            if "messages" in item and isinstance(item["messages"], list):
                valid = True
                for msg in item["messages"]:
                    if not (isinstance(msg, dict) and "role" in msg and "content" in msg):
                        valid = False
                        break
                
                if valid:
                    standardized_data.append(item)
    
    return standardized_data


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input data
    logger.info(f"Loading data from {args.input_file}")
    
    # Check file size
    file_size = os.path.getsize(args.input_file) / (1024 * 1024)  # Size in MB
    logger.info(f"File size: {file_size:.2f} MB")
    
    # Use streaming approach for large files
    if file_size > 100:  # If file is larger than 100MB
        logger.info("Large file detected, using streaming approach")
        try:
            # Try using a line-by-line JSON parser for more resilience
            logger.info("Using line-by-line JSON parsing")
            standardized_data = []
            count = 0
            batch = []
            
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Handle JSONL format (one JSON object per line)
                    try:
                        item = json.loads(line)
                        batch.append(item)
                        count += 1
                        
                        # Process in batches to save memory
                        if len(batch) >= 1000:
                            batch_standardized = convert_to_standard_format(batch, args.format)
                            standardized_data.extend(batch_standardized)
                            batch = []
                            logger.info(f"Processed {count} items")
                    except json.JSONDecodeError:
                        # Try to fix common JSON errors in the line
                        try:
                            # Check if this might be part of an array with missing commas
                            if line.startswith('{') and line.endswith('}'):
                                item = json.loads(line)
                                batch.append(item)
                                count += 1
                            elif line.startswith('[') and line.endswith(']'):
                                # This might be the entire array
                                items = json.loads(line)
                                batch.extend(items)
                                count += len(items)
                            logger.info(f"Fixed and processed line {count}")
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping malformed JSON line: {line[:50]}...")
            
            # Process any remaining items
            if batch:
                batch_standardized = convert_to_standard_format(batch, args.format)
                standardized_data.extend(batch_standardized)
                logger.info(f"Processed {count} items total")
                
        except Exception as e:
            logger.warning(f"Line-by-line parsing failed: {str(e)}")
            logger.info("Trying to fix the JSON file...")
            
            # Try to fix the JSON file by reading it as text and correcting common issues
            try:
                with open(args.input_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to determine if it's an array or object at the root
                content = content.strip()
                if content.startswith('[') and content.endswith(']'):
                    # It's an array, try to parse each item separately
                    logger.info("Detected JSON array, attempting to parse items individually")
                    
                    # Simple approach to extract objects from array
                    standardized_data = []
                    item_start = 0
                    bracket_count = 0
                    in_string = False
                    escape_next = False
                    items = []
                    
                    for i, char in enumerate(content):
                        if escape_next:
                            escape_next = False
                            continue
                            
                        if char == '\\':
                            escape_next = True
                        elif char == '"' and not escape_next:
                            in_string = not in_string
                        elif not in_string:
                            if char == '{':
                                if bracket_count == 0:
                                    item_start = i
                                bracket_count += 1
                            elif char == '}':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    try:
                                        item_json = content[item_start:i+1]
                                        item = json.loads(item_json)
                                        items.append(item)
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse item: {item_json[:50]}...")
                    
                    logger.info(f"Extracted {len(items)} items from array")
                    standardized_data = convert_to_standard_format(items, args.format)
                else:
                    # Try to load the entire file with some error handling
                    logger.warning("Falling back to standard JSON loading (may cause memory issues)")
                    with open(args.input_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    standardized_data = convert_to_standard_format(data, args.format)
            
            except Exception as e:
                logger.warning(f"JSON fixing failed: {str(e)}")
                logger.error("Could not parse the JSON file. Please check the file format.")
                sys.exit(1)
    else:
        # Standard approach for smaller files
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert to standard format
        logger.info(f"Converting data from {args.format} format to standard format")
        standardized_data = convert_to_standard_format(data, args.format)
    
    # Shuffle data
    random.shuffle(standardized_data)
    
    # Split into train and eval
    split_idx = int(len(standardized_data) * args.train_ratio)
    train_data = standardized_data[:split_idx]
    eval_data = standardized_data[split_idx:]
    
    # Save processed data
    train_output_path = os.path.join(args.output_dir, "train.json")
    eval_output_path = os.path.join(args.output_dir, "eval.json")
    
    logger.info(f"Saving {len(train_data)} training examples to {train_output_path}")
    with open(train_output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saving {len(eval_data)} evaluation examples to {eval_output_path}")
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    logger.info("Data preparation completed")


if __name__ == "__main__":
    main()