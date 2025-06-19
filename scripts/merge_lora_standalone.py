#!/usr/bin/env python3
"""
Standalone script to merge LoRA adapter with base model.
This script uses minimal imports to avoid environment conflicts.
"""

import argparse
import json
import os
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set environment variables to prevent problematic imports
os.environ['TRANSFORMERS_NO_SKLEARN'] = '1'
os.environ['SKLEARN_ENABLE_ARRAY_API'] = '0'
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_extensions'

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--adapter_path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output_path", required=True, help="Path to save merged model")
    args = parser.parse_args()
    
    try:
        print("Loading configuration...")
        
        # Load adapter config
        adapter_config_path = os.path.join(args.adapter_path, "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model = adapter_config.get("base_model_name_or_path")
        print(f"Base model: {base_model}")
        
        # Import after setting environment variables
        print("Importing libraries...")
        import torch
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("Loading adapter configuration...")
        config = PeftConfig.from_pretrained(args.adapter_path)
        
        print("Loading base model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": device} if device == "cuda" else None,
            trust_remote_code=True
        )
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True
        )
        
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        
        print("Merging LoRA adapter...")
        merged_model = model.merge_and_unload()
        
        print("Saving merged model...")
        os.makedirs(args.output_path, exist_ok=True)
        merged_model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        
        print('LoRA merge completed successfully')
        
    except Exception as e:
        print(f"Error during LoRA merge: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()