#!/usr/bin/env python3
"""
Quick test script for the fine-tuned model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_model(model_path="models/finetuned/", test_input="Tell me about yourself."):
    """Test the fine-tuned model with a sample input."""
    print(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model and apply LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Format input as chat
    messages = [{"role": "user", "content": test_input}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    print(f"Input: {test_input}")
    print("Generating response...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Response: {response}")
    print("-" * 50)

if __name__ == "__main__":
    # Test with prompts that match your training data
    test_prompts = [
        "What's your elevator pitch?",
        "How would you introduce yourself?", 
        "What's your professional summary?",
        "Tell me your elevator pitch"
    ]
    
    for prompt in test_prompts:
        test_model(test_input=prompt)