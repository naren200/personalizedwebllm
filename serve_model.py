#!/usr/bin/env python3
"""
Simple HTTP server to serve model files for local testing
"""
import http.server
import socketserver
import os
import sys
from pathlib import Path

# Server configuration
PORT = 8002
HOST = '127.0.0.1'

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support for local testing"""
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[{self.date_time_string()}] {format % args}")

def main():
    # Change to the project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print(f"Starting model server...")
    print(f"Serving from: {project_root}")
    print(f"Model weights: {project_root}/models/webllm/weights/")
    print(f"Model library: {project_root}/models/webllm/libs/")
    
    # Check if model files exist
    weights_dir = project_root / "models" / "webllm" / "weights"
    libs_dir = project_root / "models" / "webllm" / "libs"
    
    if not weights_dir.exists():
        print(f"WARNING: Weights directory not found: {weights_dir}")
    else:
        print(f"✓ Weights directory found with {len(list(weights_dir.glob('*')))} files")
    
    if not libs_dir.exists():
        print(f"WARNING: Libs directory not found: {libs_dir}")
    else:
        wasm_files = list(libs_dir.glob("*.wasm"))
        if wasm_files:
            print(f"✓ Model library found: {wasm_files[0].name}")
        else:
            print(f"WARNING: No .wasm files found in {libs_dir}")
    
    # Start server
    with socketserver.TCPServer((HOST, PORT), CORSHTTPRequestHandler) as httpd:
        print(f"\n🚀 Server running at http://{HOST}:{PORT}/")
        print(f"📁 Model weights: http://{HOST}:{PORT}/models/webllm/weights/")
        print(f"📦 Model library: http://{HOST}:{PORT}/models/webllm/libs/")
        print(f"\nPress Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped")
            sys.exit(0)

if __name__ == "__main__":
    main()