#!/bin/bash

# Hugging Face Setup Script
# This script sets up Hugging Face authentication for accessing gated models

echo "🔑 Setting up Hugging Face authentication..."

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "📁 Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
    echo "✅ Environment variables loaded from .env file!"
else
    echo "⚠️  No .env file found. Please create one with your Hugging Face token."
    echo "📝 Create a .env file with:"
    echo "HUGGINGFACE_HUB_TOKEN=your_token_here"
    exit 1
fi

# Check if token is set
if [ -z "$HUGGINGFACE_HUB_TOKEN" ] || [ "$HUGGINGFACE_HUB_TOKEN" = "your_token_here" ]; then
    echo "❌ No valid Hugging Face token found in environment variables."
    echo "📝 Please set HUGGINGFACE_HUB_TOKEN in your .env file"
    exit 1
fi

echo "✅ Hugging Face token loaded from environment variables!"

# Verify the token is working
echo "🔍 Verifying authentication..."
python -c "
import os
from huggingface_hub import whoami

try:
    user_info = whoami()
    print(f'✅ Successfully authenticated as: {user_info[\"name\"]}')
    print(f'📧 Email: {user_info.get(\"email\", \"Not provided\")}')
except Exception as e:
    print(f'❌ Authentication failed: {e}')
    print('Please check your token and try again.')
"

echo ""
echo "🚀 You can now run your training script:"
echo "./venv/bin/python -m torch.distributed.run --nproc_per_node=1 train_fsdp_hf.py"
echo ""
echo "💡 To make this permanent, add this line to your ~/.bashrc:"
echo "export HUGGINGFACE_HUB_TOKEN=\"your_token_here\""