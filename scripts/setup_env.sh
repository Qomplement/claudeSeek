#!/bin/bash
# Setup script for DeepSeek-OCR-2 + vLLM environment
# Usage: bash scripts/setup_env.sh

set -e

ENV_NAME="deepseek-ocr2"
PYTHON_VERSION="3.12.9"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing PyTorch (CUDA 11.8) ==="
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing vLLM from source ==="
if [ -d "vllm" ]; then
    echo "vLLM directory exists, pulling latest..."
    cd vllm && git pull && cd ..
else
    git clone https://github.com/vllm-project/vllm.git
fi
cd vllm
pip install -e .
cd ..

echo "=== Installing flash-attention ==="
pip install flash-attn==2.7.3 --no-build-isolation

echo "=== Pinning transformers (workaround for PR #33389) ==="
pip install transformers==4.46.3

echo "=== Installing additional dependencies ==="
pip install openai einops addict easydict

echo "=== Verifying DeepSeek-OCR-2 support ==="
python -c "
from vllm import ModelRegistry
archs = ModelRegistry.get_supported_archs()
if 'DeepseekOCR2ForCausalLM' in archs:
    print('SUCCESS: DeepseekOCR2ForCausalLM is registered in vLLM')
else:
    print('WARNING: DeepseekOCR2ForCausalLM NOT found. You may need a newer vLLM commit.')
    print('Available DeepSeek models:', [a for a in archs if 'deepseek' in a.lower()])
"

echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_NAME"
