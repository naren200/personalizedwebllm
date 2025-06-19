# Testing Guide for Personalized WebLLM

This guide will help you test your fine-tuned model in the web demo.

## Prerequisites

- Python 3.8+
- A modern web browser (Chrome, Firefox, Safari, Edge)
- Your fine-tuned model files in `models/webllm/`

## Quick Start

### Step 1: Verify Model Server Setup

First, test that your model server can serve files correctly:

```bash
cd /home/thom/Documents/GitHub/personalizedwebllm
python test_model_server.py
```

You should see all green checkmarks (✅). If any tests fail, fix those issues first.

### Step 2: Start the Model Server

Open a terminal and navigate to the project root:

```bash
cd /home/thom/Documents/GitHub/personalizedwebllm
python serve_model.py
```

You should see output like:
```
Starting model server...
Serving from: /home/thom/Documents/GitHub/personalizedwebllm
✓ Weights directory found with 15 files
✓ Model library found: finetuned-q4f16_1-MLC-webgpu.wasm

🚀 Server running at http://127.0.0.1:8002/
📁 Model weights: http://127.0.0.1:8002/models/webllm/weights/
📦 Model library: http://127.0.0.1:8002/models/webllm/libs/

Press Ctrl+C to stop the server
```

**Keep this terminal open** - the model server needs to stay running.

### Step 3: Start the Web Demo

Open a **new terminal** and navigate to the vanilla-js demo:

```bash
cd /home/thom/Documents/GitHub/personalizedwebllm/examples/vanilla-js
python -m http.server 8000
```

You should see:
```
Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...
```

**Keep this terminal open** too.

### Step 4: Open the Web Demo

1. Open your web browser
2. Navigate to: `http://localhost:8000`
3. **Immediately open Developer Tools** (Press F12)
4. Go to the **Console** tab to see loading progress and any errors
5. You should see the "🚀 Elevator Pitch Bot" interface

### Step 5: Monitor Loading

Watch the console for messages like:
```
WebLLM Configuration: {model_list: [...]}
Selected Model: finetuned-q4f16_1-MLC
Loading progress: {progress: 0.1, text: "Downloading..."}
```

**If your custom model fails**, the system will automatically try a fallback model.

## What to Expect

### Loading Process

1. **Initial Status**: "Loading model..."
2. **Model Download**: You'll see progress like "Loading model: 45%"
3. **Ready Status**: "✅ Model loaded successfully! Ready to chat."

### Testing the Model

Once loaded, try these test prompts:

#### Basic Functionality
- "Hello, how are you?"
- "Tell me about yourself"
- "What's your name?"

#### Elevator Pitch Specific (Your Fine-tuned Domain)
- "Help me create an elevator pitch"
- "What makes a good elevator pitch?"
- "Tell me about Naren's background"
- "Describe your professional experience"

## Troubleshooting

### Common Issues

#### 1. "Failed to load model" Error

**Symptoms**: Red error message in the web interface

**Solutions**:
- Check that the model server is running on port 8002
- Verify model files exist in `models/webllm/`
- Check browser console for detailed errors (F12 → Console)

#### 2. Model Server Won't Start

**Error**: `OSError: [Errno 98] Address already in use`

**Solution**: Change the port in `serve_model.py`:
```python
PORT = 8003  # Or any other available port
```

Then update the URLs in `examples/vanilla-js/elevator-pitch-bot.js`:
```javascript
model: "http://localhost:8003/models/webllm/weights/",
model_lib: "http://localhost:8003/models/webllm/libs/finetuned-q4f16_1-MLC-webgpu.wasm",
```

**Important**: Make sure both files use the same port number!

#### 3. 404 Errors for Model Files

**Symptoms**: Console shows 404 errors for model files

**Solutions**:
- Ensure you're running `serve_model.py` from the project root
- Check that files exist:
  ```bash
  ls models/webllm/weights/
  ls models/webllm/libs/
  ```

#### 4. CORS Errors

**Symptoms**: Browser blocks requests due to CORS policy

**Solution**: The `serve_model.py` script includes CORS headers. If still having issues, try Chrome with:
```bash
google-chrome --disable-web-security --user-data-dir=/tmp/chrome_dev
```

### Browser Compatibility

#### Supported Browsers
- ✅ Chrome/Chromium 90+
- ✅ Firefox 90+
- ✅ Safari 15+
- ✅ Edge 90+

#### Required Features
- WebGPU support
- SharedArrayBuffer
- WebAssembly

#### Check WebGPU Support
Visit: `chrome://gpu/` or `about:gpu` to verify WebGPU is enabled.

## Performance Notes

### Model Loading Time
- **First load**: 30-60 seconds (downloading ~500MB model)
- **Subsequent loads**: Faster due to browser caching
- **Response time**: 1-3 seconds per response

### System Requirements
- **RAM**: 4GB+ available
- **GPU**: WebGPU-compatible GPU recommended
- **Storage**: 1GB+ free space for model caching

## Advanced Testing

### Custom Configuration

Edit `examples/vanilla-js/elevator-pitch-bot.js` to modify:

```javascript
// Model configuration
overrides: {
    context_window_size: 2048,    // Increase for longer conversations
    prefill_chunk_size: 512,     // Adjust for performance
    conv_template: "qwen2"       // Must match your model
}

// Generation parameters
const response = await this.engine.chat.completions.create({
    messages: conversationHistory,
    temperature: 0.7,    // Creativity (0.0-1.0)
    max_tokens: 512,     // Response length
    stream: false        // Enable streaming for real-time responses
});
```

### Testing Different Scenarios

#### 1. Conversation Memory Test
Have a multi-turn conversation to test context retention:
1. "My name is John"
2. "I work in software engineering"
3. "What's my name and profession?"

#### 2. Personalization Test
Ask domain-specific questions related to your training data:
- Questions about elevator pitches
- Professional background inquiries
- Industry-specific terminology

#### 3. Performance Test
- Send multiple rapid messages
- Test with long input texts
- Monitor browser memory usage (F12 → Performance)

## Debugging

### Browser Console Logs

Open Developer Tools (F12) and check the Console tab for:
- Model loading progress
- Error messages
- Network request failures

### Network Tab

Check the Network tab for:
- Model weight downloads
- Failed requests (red status codes)
- Loading times and file sizes

### Server Logs

The model server shows requests in the terminal:
```
[Wed Jun 19 23:20:15 2025] "GET /models/webllm/weights/params_shard_0.bin HTTP/1.1" 200 -
[Wed Jun 19 23:20:16 2025] "GET /models/webllm/libs/finetuned-q4f16_1-MLC-webgpu.wasm HTTP/1.1" 200 -
```

## File Structure Reference

```
personalizedwebllm/
├── serve_model.py              # Model server script
├── examples/
│   └── vanilla-js/
│       ├── index.html          # Web interface
│       └── elevator-pitch-bot.js  # Main application logic
└── models/
    └── webllm/
        ├── weights/            # Model weights (15 files)
        │   ├── params_shard_*.bin
        │   ├── tokenizer.json
        │   └── mlc-chat-config.json
        └── libs/               # Compiled model library
            └── finetuned-q4f16_1-MLC-webgpu.wasm
```

## Next Steps

Once local testing works:

1. **Deploy to Production**: Upload model to HuggingFace and GitHub releases
2. **Optimize Performance**: Adjust model parameters and caching
3. **Add Features**: Implement conversation history, user profiles, etc.
4. **Integration**: Use the model in other applications (React, Vue, etc.)

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Review browser console errors
3. Verify model files are complete and uncorrupted
4. Test with the fallback model (Llama-3-8B-Instruct-q4f32_1-MLC) to isolate issues