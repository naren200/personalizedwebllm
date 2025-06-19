#!/usr/bin/env python3
"""
Convert fine-tuned Qwen model to MLC format for WebLLM deployment.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import yaml

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load WebLLM configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def run_command(cmd: list, description: str) -> bool:
    """Run a shell command and return success status."""
    logger.info(f"{description}...")
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False


def check_and_merge_lora(model_path: str, temp_dir: str) -> str:
    """Check if model is LoRA adapter and merge with base model if needed."""
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        logger.info("Detected LoRA adapter, merging with base model...")
        
        # Load adapter config to get base model
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("Base model not found in adapter config")
        
        logger.info(f"Base model: {base_model}")
        
        # Merge LoRA with base model
        merged_path = os.path.join(temp_dir, "merged_model")
        os.makedirs(merged_path, exist_ok=True)
        
        # Use standalone merge script to avoid environment issues
        merge_script = os.path.join(os.path.dirname(__file__), "merge_lora_standalone.py")
        merge_cmd = ["python", merge_script, "--adapter_path", model_path, "--output_path", merged_path]
        
        if not run_command(merge_cmd, "Merging LoRA adapter with base model"):
            raise RuntimeError("Failed to merge LoRA adapter")
        
        return merged_path
    
    return model_path


def convert_weights(model_path: str, output_dir: str, quantization: str = "q4f16_1") -> bool:
    """Convert model weights to MLC format."""
    cmd = [
        "mlc_llm", "convert_weight",
        model_path,
        "--quantization", quantization,
        "--device", "cpu",  # Use CPU to avoid CUDA compilation issues
        "-o", output_dir
    ]
    
    return run_command(cmd, f"Converting weights with {quantization} quantization")


def generate_config(
    model_path: str, 
    output_dir: str, 
    quantization: str = "q4f16_1",
    conv_template: str = "qwen2",
    context_window: int = 2048,
    prefill_chunk_size: int = 512
) -> bool:
    """Generate MLC chat config and process tokenizers."""
    cmd = [
        "mlc_llm", "gen_config",
        model_path,
        "--quantization", quantization,
        "--conv-template", conv_template,
        "--context-window-size", str(context_window),
        "--prefill-chunk-size", str(prefill_chunk_size),
        "-o", output_dir
    ]
    
    return run_command(cmd, "Generating MLC chat config")


def compile_model_library(
    config_path: str, 
    output_path: str, 
    device: str = "webgpu",
    prefill_chunk_size: Optional[int] = None
) -> bool:
    """Compile model library for WebGPU."""
    cmd = [
        "mlc_llm", "compile",
        config_path,
        "--device", device,
        "-o", output_path
    ]
    
    if prefill_chunk_size:
        cmd.extend(["--overrides", f"prefill_chunk_size={prefill_chunk_size}"])
    
    return run_command(cmd, f"Compiling model library for {device}")


def create_webllm_config(
    model_id: str,
    model_path: str,
    model_lib_path: str,
    output_path: str,
    quantization: str = "q4f16_1",
    conv_template: str = "qwen2",
    context_window: int = 2048,
    prefill_chunk_size: int = 512
) -> bool:
    """Create WebLLM configuration file."""
    config = {
        "model_list": [
            {
                "model": model_path,
                "model_id": model_id,
                "model_lib": model_lib_path,
                "required_features": ["shader-f16"],
                "overrides": {
                    "context_window_size": context_window,
                    "prefill_chunk_size": prefill_chunk_size,
                    "conv_template": conv_template
                }
            }
        ]
    }
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✓ WebLLM config created at {output_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create WebLLM config: {e}")
        return False


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert fine-tuned model to WebLLM format")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--output_dir", required=True, help="Output directory for WebLLM model")
    parser.add_argument("--model_id", help="Model ID for WebLLM (auto-generated if not provided)")
    parser.add_argument("--quantization", default="q4f16_1", help="Quantization method")
    parser.add_argument("--conv_template", default="qwen2", help="Conversation template")
    parser.add_argument("--context_window", type=int, default=2048, help="Context window size")
    parser.add_argument("--prefill_chunk_size", type=int, default=512, help="Prefill chunk size")
    parser.add_argument("--config", default="config/webllm_config.yaml", help="Config file path")
    parser.add_argument("--skip_compile", action="store_true", help="Skip library compilation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with config values if available
    quantization = config.get("quantization", args.quantization)
    conv_template = config.get("conv_template", args.conv_template)
    context_window = config.get("context_window_size", args.context_window)
    prefill_chunk_size = config.get("prefill_chunk_size", args.prefill_chunk_size)
    
    # Generate model ID if not provided
    if not args.model_id:
        model_name = Path(args.model_path).name
        args.model_id = f"{model_name}-{quantization}-MLC"
    
    logger.info(f"Converting model: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model ID: {args.model_id}")
    logger.info(f"Quantization: {quantization}")
    
    # Create output directories
    weights_dir = os.path.join(args.output_dir, "weights")
    libs_dir = os.path.join(args.output_dir, "libs")
    temp_dir = os.path.join(args.output_dir, "temp")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(libs_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    success = True
    actual_model_path = args.model_path
    
    try:
        # Step 0: Check for LoRA and merge if needed
        logger.info("=== Step 0: Checking model format ===")
        actual_model_path = check_and_merge_lora(args.model_path, temp_dir)
        if actual_model_path != args.model_path:
            logger.info(f"Using merged model at: {actual_model_path}")
        
        # Step 1: Convert weights
        logger.info("=== Step 1: Converting model weights ===")
        if not convert_weights(actual_model_path, weights_dir, quantization):
            logger.error("Weight conversion failed")
            success = False
    except Exception as e:
        logger.error(f"Error during model preparation: {e}")
        success = False
    
    # Step 2: Generate config
    logger.info("=== Step 2: Generating MLC config ===")
    if success and not generate_config(
        actual_model_path, 
        weights_dir, 
        quantization, 
        conv_template,
        context_window,
        prefill_chunk_size
    ):
        logger.error("Config generation failed")
        success = False
    
    # Step 3: Compile model library (optional)
    if success and not args.skip_compile:
        logger.info("=== Step 3: Compiling model library ===")
        config_path = os.path.join(weights_dir, "mlc-chat-config.json")
        lib_path = os.path.join(libs_dir, f"{args.model_id}-webgpu.wasm")
        
        if not compile_model_library(config_path, lib_path, "webgpu", prefill_chunk_size):
            logger.error("Library compilation failed")
            success = False
    
    # Step 4: Create WebLLM configuration
    if success:
        logger.info("=== Step 4: Creating WebLLM configuration ===")
        webllm_config_path = os.path.join(args.output_dir, "webllm_config.json")
        model_url = f"https://huggingface.co/your-username/{args.model_id}"
        lib_url = f"https://github.com/your-username/your-repo/releases/download/v1.0/{args.model_id}-webgpu.wasm"
        
        if not create_webllm_config(
            args.model_id,
            model_url,
            lib_url,
            webllm_config_path,
            quantization,
            conv_template,
            context_window,
            prefill_chunk_size
        ):
            success = False
    
    # Step 5: Create deployment instructions
    if success:
        logger.info("=== Step 5: Creating deployment instructions ===")
        instructions_path = os.path.join(args.output_dir, "DEPLOYMENT.md")
        
        instructions = f"""# Deployment Instructions for {args.model_id}

## Files Generated

### Model Weights (Upload to HuggingFace)
- Location: `{weights_dir}/`
- Contains: Model weights, tokenizer, and config files
- Upload command: `python scripts/deploy_hf.py --model_path {weights_dir} --hf_repo your-username/{args.model_id}`

### Model Library (Upload to GitHub Releases)
- Location: `{libs_dir}/{args.model_id}-webgpu.wasm`
- Upload to: GitHub releases or CDN
- Size: Check file size for CDN limitations

### WebLLM Configuration
- Location: `{webllm_config_path}`
- Use this configuration in your web applications

## Next Steps

1. **Upload model weights to HuggingFace:**
   ```bash
   python scripts/deploy_hf.py \\
       --model_path {weights_dir} \\
       --hf_repo your-username/{args.model_id}
   ```

2. **Upload model library to GitHub:**
   - Create a release on your GitHub repository
   - Upload the `.wasm` file from `{libs_dir}/`
   - Update the `model_lib` URL in your WebLLM config

3. **Test in browser:**
   ```bash
   cd examples/web-demo
   npm install
   npm run dev
   ```

## Model Configuration

- **Model ID:** `{args.model_id}`
- **Quantization:** `{quantization}`
- **Conversation Template:** `{conv_template}`
- **Context Window:** `{context_window}`
- **Prefill Chunk Size:** `{prefill_chunk_size}`

## WebLLM Integration

Use the generated `webllm_config.json` in your web applications:

```javascript
import {{ CreateMLCEngine }} from "@mlc-ai/web-llm";

const appConfig = {{
  // Your generated config here
}};

const engine = await CreateMLCEngine("{args.model_id}", {{ appConfig }});
```
"""
        
        try:
            with open(instructions_path, 'w') as f:
                f.write(instructions)
            logger.info(f"✓ Deployment instructions created at {instructions_path}")
        except Exception as e:
            logger.error(f"✗ Failed to create deployment instructions: {e}")
    
    # Cleanup temporary files
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir: {e}")
    
    if success:
        logger.info("🎉 Conversion completed successfully!")
        logger.info(f"Model files are ready in: {args.output_dir}")
        logger.info("Check DEPLOYMENT.md for next steps")
    else:
        logger.error("❌ Conversion failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()