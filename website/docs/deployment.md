---
id: deployment
title: Deployment Guide
sidebar_position: 4
---

# Deployment Guide

## Version Requirements (Tested & Working)

Exact versions tested end-to-end on AWS EC2 g5.xlarge (NVIDIA A10G, 24GB VRAM).

### Core Stack

| Component | Version | Notes |
|-----------|---------|-------|
| **Python** | 3.10.12 | 3.10+ required by vLLM |
| **vLLM** | 0.16.0 | First version with native `DeepseekOCR2ForCausalLM` |
| **transformers** | 4.57.6 | vLLM 0.16.0 requires `>=4.56.0` |
| **PyTorch** | 2.9.1+cu128 | Installed by vLLM |
| **CUDA (runtime)** | 12.8 | Via PyTorch wheel |
| **NVIDIA Driver** | 580.126.09 | Must support CUDA 12.8+ |

### Key Dependencies (auto-installed by vLLM)

| Package | Version |
|---------|---------|
| tokenizers | 0.22.2 |
| huggingface_hub | 0.36.2 |
| safetensors | 0.7.0 |
| pillow | 12.1.1 |
| numpy | 2.2.6 |
| triton | 3.5.1 |
| xformers | 0.0.28.post3 |

### Hardware Tested

| Component | Spec |
|-----------|------|
| **GPU** | NVIDIA A10G (24 GB, compute 8.6) |
| **Instance** | AWS g5.xlarge (4 vCPU, 16 GB RAM) |
| **OS** | Ubuntu 22.04.5 LTS |
| **AMI** | ami-0669ac5db8b1292fe |

### Version Compatibility Notes

- **vLLM < 0.16.0** — No `DeepseekOCR2ForCausalLM` in registry. Must register manually.
- **transformers >= 4.48** — Accuracy bug: `_update_causal_mask()` removed, breaks Visual Causal Flow. Model hallucinates text. [PR #33389](https://github.com/vllm-project/vllm/pull/33389) not merged yet.
- **transformers >= 4.56.0** — Required by vLLM 0.16.0. Cannot pin to 4.46.3.
- **CUDA 12.x** — Required. No Apple Silicon / AMD GPU support.

## Quick Start

```bash
# Install vLLM (installs PyTorch, transformers, everything)
pip3 install vllm==0.16.0

# Verify
python3 -c "from vllm import ModelRegistry; assert 'DeepseekOCR2ForCausalLM' in ModelRegistry.get_supported_archs(); print('OK')"

# Start server (first run downloads ~6 GB model)
python3 -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-OCR-2 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --host 0.0.0.0 --port 8000 \
    --logits-processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor
```

## AWS EC2 Setup

1. Launch **g5.xlarge** with AMI `ami-0669ac5db8b1292fe` (Ubuntu 22.04 + NVIDIA drivers)
2. Open ports 22 (SSH) and 8000 (API)
3. SSH in, `pip3 install vllm==0.16.0`, start server

## API Usage

### Python (openai SDK)

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

with open("document.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-OCR-2",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": "Convert the document to markdown."},
        ],
    }],
    max_tokens=8192,
    temperature=0.0,
    extra_body={
        "vllm_xargs": {
            "ngram_size": 20,
            "window_size": 90,
            "whitelist_token_ids": [128821, 128822],
        },
    },
)
print(response.choices[0].message.content)
```

## Docker

```dockerfile
FROM vllm/vllm-openai:latest
EXPOSE 8000
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "deepseek-ai/DeepSeek-OCR-2", \
     "--trust-remote-code", "--dtype", "bfloat16", \
     "--max-model-len", "8192", "--gpu-memory-utilization", "0.90", \
     "--enforce-eager", "--no-enable-prefix-caching", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--logits-processors", "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor"]
```

```bash
docker build --platform linux/amd64 -t deepseek-ocr2-vllm .
docker run --gpus all -p 8000:8000 deepseek-ocr2-vllm
```

## Performance (g5.xlarge / A10G)

| Metric | Value |
|--------|-------|
| Throughput | ~61 tok/s |
| Per page | 16-26 seconds |
| 2-page invoice | ~42 seconds |
| Model load | ~11 seconds |
| GPU memory | ~21 GB |

## Server Arguments

| Argument | Value | Description |
|----------|-------|-------------|
| `--model` | `deepseek-ai/DeepSeek-OCR-2` | HuggingFace model ID |
| `--trust-remote-code` | flag | Required for custom HF code |
| `--dtype` | `bfloat16` | Model precision |
| `--max-model-len` | `8192` | Max context (model max) |
| `--gpu-memory-utilization` | `0.90` | VRAM fraction |
| `--enforce-eager` | flag | Disables CUDA graphs |
| `--no-enable-prefix-caching` | flag | OCR doesn't reuse prefixes |
| `--logits-processors` | `vllm...deepseek_ocr:NGramPerReqLogitsProcessor` | Prevents repetition |

## Request Parameters (via `vllm_xargs`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ngram_size` | 20 | Official: 20 for images, 40 for batch |
| `window_size` | 90 | Official: 90 for images, 50 for PDFs |
| `whitelist_token_ids` | [128821, 128822] | `<td>`, `</td>` exempt from ban |
