# Personalized WebLLM - Qwen 2.5 Fine-tuning for Elevator Pitches

Complete toolkit for fine-tuning Qwen 2.5 models on custom datasets and deploying them with WebLLM for in-browser inference.

## Features

- **Fine-tune Qwen 2.5** on your custom elevator pitch dataset
- **Convert to WebLLM** format for in-browser inference
- **Deploy to HuggingFace** for easy distribution
- **Ready-to-use web examples** for your websites
- **Automated scripts** for the entire pipeline

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU with CUDA 11.8+ (for fine-tuning and MLC-LLM)
- Node.js 18+ (for web examples)
- Git LFS
- WebGPU-compatible browser (Chrome 113+)
- **Emscripten SDK** (for WebGPU compilation)
- **~4GB disk space** (for MLC-LLM source build)

## 🛠️ Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/personalizedwebllm.git
cd personalizedwebllm

# Install Python dependencies
pip install -r requirements.txt

# Install MLC-LLM for basic functionality (Linux with CUDA 12.2)
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu122 mlc-ai-nightly-cu122

# For WebGPU compilation, install from source (REQUIRED)
# 1. Install Emscripten SDK
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
cd ..

# 2. Clone and build MLC-LLM from source
git clone https://github.com/mlc-ai/mlc-llm.git mlc-llm-source
cd mlc-llm-source
./web/prep_emcc_deps.sh

# 3. Build WebAssembly runtime libraries
cd web
make
cd ../..

# 4. Set environment variables (add to ~/.bashrc for persistence)
export MLC_LLM_SOURCE_DIR=$(pwd)/mlc-llm-source
export PATH="$(pwd)/emsdk:$(pwd)/emsdk/upstream/emscripten:$PATH"

# If you are having conda env - Copy runtime libraries to expected locations
cp mlc-llm-source/web/dist/wasm/mlc_wasm_runtime.bc ~/.conda/envs/$(conda info --envs | grep '*' | awk '{print $1}')/lib/python*/site-packages/tvm/wasm_runtime.bc
cp mlc-llm-source/web/dist/wasm/mlc_wasm_runtime.bc ~/.conda/envs/$(conda info --envs | grep '*' | awk '{print $1}')/lib/python*/site-packages/tvm/tvmjs_support.bc
cp mlc-llm-source/web/dist/wasm/mlc_wasm_runtime.bc ~/.conda/envs/$(conda info --envs | grep '*' | awk '{print $1}')/lib/python*/site-packages/tvm/webgpu_runtime.bc

# Verify installation
mlc_llm --help
emcc --version
```

### 2. Prepare Your Dataset

Create your elevator pitch dataset in the `data/` directory:

```bash
python scripts/prepare_dataset.py --input data/naren_elevator_pitch.json --output data/processed/
```

### 3. Fine-tune Qwen 2.5

```bash
python scripts/finetune_qwen.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_path data/processed/ \
    --output_dir models/finetuned/
```

### 4. Convert to WebLLM Format

```bash
# IMPORTANT: Set environment variables for WebGPU compilation
export PATH="$(pwd)/emsdk:$(pwd)/emsdk/upstream/emscripten:$PATH"
export MLC_LLM_SOURCE_DIR=$(pwd)/mlc-llm-source

# Convert to WebLLM format (now with working WebGPU compilation!)
python scripts/convert_to_mlc.py \
    --model_path models/finetuned \
    --output_dir models/webllm \
    --quantization q4f16_1

# Expected successful output:
# 🎉 Conversion completed successfully!
# Model files are ready in: models/webllm
# Check DEPLOYMENT.md for next steps
```

**Note**: If you get "Cannot find library: mlc_wasm_runtime.bc" error, you forgot to set the environment variables above!

### 5. Verify Successful Conversion

```bash
# Check that all required files were generated
ls -la models/webllm/libs/          # Should contain *.wasm files
ls -la models/webllm/weights/       # Should contain quantized model weights
cat models/webllm/DEPLOYMENT.md     # Contains deployment instructions

# Expected files:
# models/webllm/libs/finetuned-q4f16_1-MLC-webgpu.wasm
# models/webllm/weights/mlc-chat-config.json
# models/webllm/weights/ndarray-cache.json
# models/webllm/weights/params_shard_*.bin
# models/webllm/webllm_config.json
```

### 6. Deploy to HuggingFace

```bash
python scripts/deploy_hf.py \
    --model_path models/webllm/ \
    --hf_repo your-username/your-model-name
```

### 7. Test in Browser

```bash
cd examples/web-demo
npm install
npm run dev
```

## 📁 Repository Structure

```
personalizedwebllm/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config/                   # Configuration files
│   ├── training_config.yaml
│   └── webllm_config.json
├── scripts/                  # Automation scripts
│   ├── prepare_dataset.py
│   ├── finetune_qwen.py
│   ├── convert_to_mlc.py
│   └── deploy_hf.py
├── data/                     # Dataset directory
│   ├── example_dataset.json
│   └── processed/
├── models/                   # Model outputs
│   ├── finetuned/
│   └── webllm/
└── examples/                 # Web integration examples
    ├── web-demo/
    ├── react-integration/
    └── vanilla-js/
```

## 📊 Dataset Format

Your dataset should be in JSON format with conversation pairs:

```json
[
  {
    "input": "Tell me about yourself",
    "output": "I'm a software engineer with 5 years of experience in web development..."
  },
  {
    "input": "What's your elevator pitch?",
    "output": "I specialize in building scalable web applications using modern frameworks..."
  }
]
```

## 🔧 Configuration

### Training Configuration (`config/training_config.yaml`)

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  max_length: 2048

training:
  batch_size: 4
  learning_rate: 2e-4
  num_epochs: 3
  gradient_accumulation_steps: 8

lora:
  r: 16
  alpha: 32
  dropout: 0.1
```

### WebLLM Configuration (`config/webllm_config.json`)

```json
{
  "model_id": "PersonalizedQwen-0.6B-Chat-q4f16_1-MLC",
  "quantization": "q4f16_1",
  "conv_template": "qwen",
  "context_window_size": 2048,
  "prefill_chunk_size": 512
}
```

## 🌐 Web Integration Examples

### Basic HTML Integration

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.46/lib/bundle.min.js"></script>
</head>
<body>
    <div id="chat-container"></div>
    <script src="js/elevator-pitch-bot.js"></script>
</body>
</html>
```

### React Integration

```jsx
import { CreateMLCEngine } from "@mlc-ai/web-llm";
import { useElevatorPitchBot } from "./hooks/useElevatorPitchBot";

export function ElevatorPitchChat() {
  const { messages, sendMessage, isLoading } = useElevatorPitchBot();
  
  return (
    <div className="chat-interface">
      {/* Chat UI implementation */}
    </div>
  );
}
```

## 📚 Detailed Instructions

### Fine-tuning Process

1. **Data Preparation**: Format your elevator pitch conversations
2. **Model Loading**: Load Qwen 2.5 base model
3. **LoRA Training**: Fine-tune using Low-Rank Adaptation
4. **Model Saving**: Save the fine-tuned weights

### WebLLM Conversion

1. **Weight Conversion**: Convert PyTorch weights to MLC format
2. **Quantization**: Apply q4f16_1 quantization for web deployment
3. **Library Compilation**: Build WebGPU-compatible model library (✅ **Now Working!**)
4. **Config Generation**: Create WebLLM configuration files

**✅ WebGPU Compilation Status**: Fully functional with source installation!

### Deployment

1. **HuggingFace Upload**: Push model weights to HF Hub
2. **GitHub Release**: Upload compiled libraries
3. **CDN Distribution**: Make models available via CDN
4. **Version Management**: Tag releases for stability

## 🧪 Testing Your Model

### Quick Python Test

```bash
# Test the fine-tuned model with sample prompts
python test_model.py
```

### Browser Testing

```bash
# Vanilla JS example
cd examples/vanilla-js
python -m http.server 8000
# Open http://localhost:8000

# React example
cd examples/react-integration
npm install
npm start
```

### Training Results

**Current Training Performance:**
- Dataset: 20 elevator pitch examples
- Training steps: 3 (limited by small dataset)
- Model: Qwen2.5-0.5B-Instruct with LoRA
- Trainable parameters: 2.16M (0.44% of total)

**To improve training:**
```bash
# Add more training data or increase training steps
python scripts/finetune_qwen.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_path data/processed/ \
    --output_dir models/finetuned/ \
    --max_steps 100 \
    --learning_rate 5e-4
```

## 🔍 Troubleshooting

### Common Issues

**Limited Training Results**
```bash
# Issue: Model not learning from training data
# Solution: Increase training data or steps
# Current: 20 examples → 3 training steps
# Recommended: 100+ examples or --max_steps 100
```

**GPU Memory Issues**
```bash
# Reduce batch size in config
batch_size: 2
gradient_accumulation_steps: 16
```

**WebGPU Compatibility**
```bash
# Check browser support
chrome://gpu/
# Enable WebGPU flags if needed
```

**MLC-LLM Installation Issues**
```bash
# For Linux with CUDA 12.2:
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu122 mlc-ai-nightly-cu122

# For other configurations, check the official installation guide:
# https://llm.mlc.ai/docs/install/mlc_llm

# Common CUDA versions:
# CUDA 11.8: mlc-llm-nightly-cu118 mlc-ai-nightly-cu118
# CUDA 12.1: mlc-llm-nightly-cu121 mlc-ai-nightly-cu121
# CPU-only: mlc-llm-nightly mlc-ai-nightly
```

**Environment Setup Issues**
```bash
# If conda environment detection fails in setup
# Manually set the Python site-packages path:
PYTHON_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cp mlc-llm-source/web/dist/wasm/mlc_wasm_runtime.bc $PYTHON_SITE/tvm/wasm_runtime.bc
cp mlc-llm-source/web/dist/wasm/mlc_wasm_runtime.bc $PYTHON_SITE/tvm/tvmjs_support.bc
cp mlc-llm-source/web/dist/wasm/mlc_wasm_runtime.bc $PYTHON_SITE/tvm/webgpu_runtime.bc

# If Emscripten path issues occur
# Add to your ~/.bashrc for persistence:
echo 'export PATH="/path/to/your/project/emsdk:/path/to/your/project/emsdk/upstream/emscripten:$PATH"' >> ~/.bashrc
echo 'export MLC_LLM_SOURCE_DIR="/path/to/your/project/mlc-llm-source"' >> ~/.bashrc
source ~/.bashrc

# Verify WebGPU compilation works
python scripts/convert_to_mlc.py --model_path models/finetuned --output_dir models/webllm --quantization q4f16_1
```

**Verification Steps**
```bash
# Check successful WebGPU compilation
ls models/webllm/libs/*.wasm        # Should exist
file models/webllm/libs/*.wasm      # Should show "WebAssembly (wasm) binary module"
du -h models/webllm/libs/*.wasm     # Should be several MB in size

# Test the compiled model loads correctly
python -c "
import json
with open('models/webllm/webllm_config.json') as f:
    config = json.load(f)
    print('✅ WebLLM config loaded successfully')
    print(f'Model ID: {config.get(\"model_id\")}')
    print(f'Quantization: {config.get(\"quantization\")}')
"
```

**Model Loading Errors**
```bash
# Verify model files
ls -la models/webllm/
# Check file sizes and formats
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) for the WebLLM framework
- [Qwen](https://github.com/QwenLM/Qwen) for the base model
- [HuggingFace](https://huggingface.co/) for model hosting

## 📞 Support

- 📧 Email: support@yourproject.com
- 💬 Discord: [Join our community](https://discord.gg/your-invite)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/personalizedwebllm/issues)
