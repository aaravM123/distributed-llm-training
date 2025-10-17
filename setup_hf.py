#!/usr/bin/env python3
"""
Hugging Face Setup Script
This script sets up Hugging Face authentication for accessing gated models
"""

import os
import sys
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if env_file.exists():
        print("📁 Loading environment variables from .env file...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        return True
    return False

def setup_huggingface():
    """Set up Hugging Face authentication"""
    
    print("🔑 Setting up Hugging Face authentication...")
    
    # Try to load from .env file first
    if not load_env_file():
        print("⚠️  No .env file found. Please create one with your Hugging Face token.")
        print("📝 Create a .env file with:")
        print("HUGGINGFACE_HUB_TOKEN=your_token_here")
        return False
    
    # Check if token is set
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token or hf_token == "your_token_here":
        print("❌ No valid Hugging Face token found in environment variables.")
        print("📝 Please set HUGGINGFACE_HUB_TOKEN in your .env file")
        return False
    
    print("✅ Hugging Face token loaded from environment variables!")
    
    # Verify authentication
    print("🔍 Verifying authentication...")
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Successfully authenticated as: {user_info['name']}")
        print(f"📧 Email: {user_info.get('email', 'Not provided')}")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("Please check your token and try again.")
        return False

def main():
    """Main function"""
    success = setup_huggingface()
    
    if success:
        print("\n🚀 You can now run your training script:")
        print("./venv/bin/python -m torch.distributed.run --nproc_per_node=1 train_fsdp_hf.py")
        print("\n💡 To make this permanent, add this line to your ~/.bashrc:")
        print('export HUGGINGFACE_HUB_TOKEN="your_token_here"')
    else:
        print("\n❌ Setup failed. Please check your token and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()