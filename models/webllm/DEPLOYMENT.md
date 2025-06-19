# Deployment Instructions for finetuned-q4f16_1-MLC

## Files Generated

### Model Weights (Upload to HuggingFace)
- Location: `models/webllm/weights/`
- Contains: Model weights, tokenizer, and config files
- Upload command: `python scripts/deploy_hf.py --model_path models/webllm/weights --hf_repo your-username/finetuned-q4f16_1-MLC`

### Model Library (Upload to GitHub Releases)
- Location: `models/webllm/libs/finetuned-q4f16_1-MLC-webgpu.wasm`
- Upload to: GitHub releases or CDN
- Size: Check file size for CDN limitations

### WebLLM Configuration
- Location: `models/webllm/webllm_config.json`
- Use this configuration in your web applications

## Next Steps

1. **Upload model weights to HuggingFace:**
   ```bash
   python scripts/deploy_hf.py \
       --model_path models/webllm/weights \
       --hf_repo your-username/finetuned-q4f16_1-MLC
   ```

2. **Upload model library to GitHub:**
   - Create a release on your GitHub repository
   - Upload the `.wasm` file from `models/webllm/libs/`
   - Update the `model_lib` URL in your WebLLM config

3. **Test in browser:**
   ```bash
   cd examples/web-demo
   npm install
   npm run dev
   ```

## Model Configuration

- **Model ID:** `finetuned-q4f16_1-MLC`
- **Quantization:** `q4f16_1`
- **Conversation Template:** `qwen2`
- **Context Window:** `2048`
- **Prefill Chunk Size:** `512`

## WebLLM Integration

Use the generated `webllm_config.json` in your web applications:

```javascript
import { CreateMLCEngine } from "@mlc-ai/web-llm";

const appConfig = {
  // Your generated config here
};

const engine = await CreateMLCEngine("finetuned-q4f16_1-MLC", { appConfig });
```
