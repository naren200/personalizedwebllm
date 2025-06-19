#!/usr/bin/env python3
"""
Test script to verify model server is working correctly
"""
import requests
import sys
from pathlib import Path

def test_server():
    base_url = "http://localhost:8002"
    
    print("🔍 Testing Model Server...")
    print(f"Base URL: {base_url}")
    print()
    
    # Test 1: Server is running
    try:
        response = requests.get(base_url, timeout=5)
        print(f"✅ Server is running (Status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running or not accessible")
        print("   Run: python serve_model.py")
        return False
    except Exception as e:
        print(f"❌ Server error: {e}")
        return False
    
    # Test 2: Model weights directory
    weights_url = f"{base_url}/models/webllm/weights/"
    try:
        response = requests.get(weights_url, timeout=5)
        if response.status_code == 200:
            print(f"✅ Model weights accessible at {weights_url}")
        else:
            print(f"❌ Model weights not found (Status: {response.status_code})")
            print(f"   URL: {weights_url}")
    except Exception as e:
        print(f"❌ Error accessing weights: {e}")
    
    # Test 3: Model library file
    lib_url = f"{base_url}/models/webllm/libs/finetuned-q4f16_1-MLC-webgpu.wasm"
    try:
        response = requests.head(lib_url, timeout=5)  # HEAD request to check existence
        if response.status_code == 200:
            size_mb = int(response.headers.get('content-length', 0)) / (1024*1024)
            print(f"✅ Model library accessible at {lib_url}")
            print(f"   Size: {size_mb:.1f} MB")
        else:
            print(f"❌ Model library not found (Status: {response.status_code})")
            print(f"   URL: {lib_url}")
    except Exception as e:
        print(f"❌ Error accessing library: {e}")
    
    # Test 4: Check specific required files
    print()
    print("🔍 Checking required model files...")
    
    required_files = [
        "mlc-chat-config.json",
        "tokenizer.json",
        "params_shard_0.bin"
    ]
    
    for filename in required_files:
        file_url = f"{base_url}/models/webllm/weights/{filename}"
        try:
            response = requests.head(file_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {filename}")
            else:
                print(f"❌ {filename} (Status: {response.status_code})")
        except Exception as e:
            print(f"❌ {filename} (Error: {e})")
    
    # Test 5: CORS headers
    print()
    print("🔍 Checking CORS headers...")
    try:
        response = requests.get(weights_url, timeout=5)
        cors_origin = response.headers.get('Access-Control-Allow-Origin')
        if cors_origin == '*':
            print("✅ CORS headers configured correctly")
        else:
            print(f"❌ CORS headers missing or incorrect: {cors_origin}")
    except Exception as e:
        print(f"❌ Error checking CORS: {e}")
    
    print()
    print("📋 Summary:")
    print("If all tests pass, your model server should work with WebLLM.")
    print("If any tests fail, check the error messages and fix the issues.")
    print()
    print("Next steps:")
    print("1. Open http://localhost:8000 in your browser")
    print("2. Open browser developer tools (F12)")
    print("3. Check the Console tab for any error messages")
    
    return True

if __name__ == "__main__":
    try:
        test_server()
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted")
        sys.exit(0)