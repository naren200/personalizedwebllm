#!/usr/bin/env python3
"""
Deploy fine-tuned model to HuggingFace Hub.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, Repository, create_repo, upload_folder
from huggingface_hub.utils import HfHubHTTPError

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_model_card(
    model_id: str,
    base_model: str,
    dataset_info: str,
    model_description: str,
    quantization: str = "q4f16_1"
) -> str:
    """Create a model card for the HuggingFace model."""
    return f"""---
license: mit
base_model: {base_model}
tags:
- qwen
- elevator-pitch
- webllm
- fine-tuned
- conversational
language:
- en
pipeline_tag: text-generation
---

# {model_id}

This is a fine-tuned Qwen model optimized for elevator pitch conversations and deployable with WebLLM for in-browser inference.

## Model Details

- **Base Model:** {base_model}
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Quantization:** {quantization}
- **Target Use Case:** Elevator pitch conversations
- **Deployment:** WebLLM compatible for in-browser inference

## Model Description

{model_description}

## Dataset Information

{dataset_info}

## Usage

### With WebLLM (Browser)

```javascript
import {{ CreateMLCEngine }} from "@mlc-ai/web-llm";

const appConfig = {{
  model_list: [
    {{
      model: "https://huggingface.co/{model_id}",
      model_id: "{model_id}",
      model_lib: "https://github.com/your-username/your-repo/releases/download/v1.0/{model_id}-webgpu.wasm",
      required_features: ["shader-f16"]
    }}
  ]
}};

const engine = await CreateMLCEngine("{model_id}", {{ appConfig }});

// Start conversation
const response = await engine.chat.completions.create({{
  messages: [
    {{ role: "user", content: "Tell me about yourself" }}
  ]
}});

console.log(response.choices[0].message.content);
```

### With Transformers (Python)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = "{base_model}"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)

# Load fine-tuned weights
model = PeftModel.from_pretrained(model, "{model_id}")

# Generate response
messages = [
    {{"role": "user", "content": "What's your elevator pitch?"}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

- **Training Framework:** Transformers + PEFT
- **Training Method:** LoRA fine-tuning
- **Base Model:** {base_model}
- **Quantization:** 4-bit quantization for efficient inference

## WebLLM Deployment

This model is specifically optimized for WebLLM deployment:

1. **Quantized weights** for efficient browser inference
2. **Compiled model library** for WebGPU acceleration
3. **Optimized context window** for elevator pitch conversations
4. **Browser-compatible format** with no server dependencies

## Performance

- **Model Size:** ~400MB (quantized)
- **Inference Speed:** Real-time in modern browsers
- **Memory Usage:** ~1GB GPU memory (WebGPU)
- **Compatibility:** Chrome 113+, Edge 113+, Firefox (experimental)

## Limitations

- Optimized specifically for elevator pitch conversations
- May not perform well on other conversational tasks
- Requires WebGPU-compatible browser for optimal performance
- Context window limited to 2048 tokens

## Ethical Considerations

- Model responses should be reviewed for accuracy
- Not suitable for professional advice or critical decisions
- May reflect biases present in training data
- Intended for demonstration and educational purposes

## Citation

If you use this model, please cite:

```bibtex
@misc{{{model_id.lower().replace("-", "_")},
  title={{{model_id}: Fine-tuned Qwen for Elevator Pitch Conversations}},
  author={{Your Name}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/{model_id}}}}},
}}
```

## License

This model is released under the MIT License. See LICENSE for details.
"""


def deploy_to_huggingface(
    model_path: str,
    hf_repo: str,
    hf_token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload fine-tuned model",
    base_model: str = "Qwen/Qwen-0.5B-Chat",
    model_description: str = "Fine-tuned Qwen model for elevator pitch conversations",
    dataset_info: str = "Custom elevator pitch dataset"
) -> bool:
    """Deploy model to HuggingFace Hub."""
    try:
        api = HfApi(token=hf_token)
        
        # Create repository if it doesn't exist
        logger.info(f"Creating repository: {hf_repo}")
        try:
            create_repo(
                repo_id=hf_repo,
                token=hf_token,
                private=private,
                exist_ok=True
            )
            logger.info(f"✓ Repository created/verified: {hf_repo}")
        except HfHubHTTPError as e:
            if "already exists" in str(e):
                logger.info(f"✓ Repository already exists: {hf_repo}")
            else:
                raise e
        
        # Create model card
        logger.info("Creating model card...")
        model_card_content = create_model_card(
            model_id=hf_repo,
            base_model=base_model,
            dataset_info=dataset_info,
            model_description=model_description
        )
        
        # Save model card to model directory
        model_card_path = os.path.join(model_path, "README.md")
        with open(model_card_path, 'w') as f:
            f.write(model_card_content)
        logger.info(f"✓ Model card created: {model_card_path}")
        
        # Upload entire folder
        logger.info(f"Uploading model from {model_path} to {hf_repo}...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=hf_repo,
            commit_message=commit_message,
            token=hf_token
        )
        
        logger.info(f"🎉 Model successfully deployed to: https://huggingface.co/{hf_repo}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        return False


def validate_model_directory(model_path: str) -> bool:
    """Validate that the model directory contains necessary files."""
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    optional_files = [
        "mlc-chat-config.json",
        "ndarray-cache.json",
        "params_shard_*.bin"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    logger.info("✓ Model directory validation passed")
    return True


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy model to HuggingFace Hub")
    parser.add_argument("--model_path", required=True, help="Path to the model directory")
    parser.add_argument("--hf_repo", required=True, help="HuggingFace repository name (username/repo)")
    parser.add_argument("--hf_token", help="HuggingFace access token (or use HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--commit_message", default="Upload fine-tuned model", help="Commit message")
    parser.add_argument("--base_model", default="Qwen/Qwen-0.5B-Chat", help="Base model name")
    parser.add_argument("--model_description", help="Model description for README")
    parser.add_argument("--dataset_info", help="Dataset information for README")
    parser.add_argument("--force", action="store_true", help="Skip validation checks")
    
    args = parser.parse_args()
    
    # Get HuggingFace token
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HuggingFace token not provided. Use --hf_token or set HF_TOKEN environment variable")
        sys.exit(1)
    
    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    # Validate model directory
    if not args.force and not validate_model_directory(args.model_path):
        logger.error("Model directory validation failed. Use --force to skip validation.")
        sys.exit(1)
    
    # Set default descriptions
    model_description = args.model_description or "Fine-tuned Qwen model optimized for elevator pitch conversations and WebLLM deployment."
    dataset_info = args.dataset_info or "Custom dataset containing elevator pitch conversations and professional introductions."
    
    logger.info(f"Deploying model to HuggingFace Hub...")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Repository: {args.hf_repo}")
    logger.info(f"Private: {args.private}")
    
    # Deploy to HuggingFace
    success = deploy_to_huggingface(
        model_path=args.model_path,
        hf_repo=args.hf_repo,
        hf_token=hf_token,
        private=args.private,
        commit_message=args.commit_message,
        base_model=args.base_model,
        model_description=model_description,
        dataset_info=dataset_info
    )
    
    if success:
        logger.info("🎉 Deployment completed successfully!")
        logger.info(f"Model available at: https://huggingface.co/{args.hf_repo}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Update your WebLLM config with the new model URL")
        logger.info("2. Upload the compiled model library to GitHub releases")
        logger.info("3. Test the model in your web application")
    else:
        logger.error("❌ Deployment failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()