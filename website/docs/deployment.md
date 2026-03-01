---
id: deployment
title: Deployment Guide
sidebar_position: 4
---

# Deployment Guide

## Quick Start

### Environment Setup

```bash
conda create -n deepseek-ocr2 python=3.12.9
conda activate deepseek-ocr2

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install vLLM 0.16+ (has native DeepSeek-OCR-2 support)
pip install 'vllm>=0.16'

pip install flash-attn==2.7.3 --no-build-isolation
```

### Verify

```python
from vllm import ModelRegistry
assert "DeepseekOCR2ForCausalLM" in ModelRegistry.get_supported_archs()
```

## Offline Batch Processing

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR-2",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
    enable_prefix_caching=False,
    logits_processors=[NGramPerReqLogitsProcessor],
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=4096,
    skip_special_tokens=False,
    extra_args={
        "ngram_size": 30,
        "window_size": 300,
        "whitelist_token_ids": [128821, 128822],
    },
)

prompt = "<image>\nConvert the document to markdown."
result = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": "document.png"}},
    sampling_params=sampling_params,
)
print(result[0].outputs[0].text)
```

## Online API Server

```bash
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

> **Important:** The logits processor module is `deepseek_ocr` (not `deepseek_ocr2`) and the flag uses hyphens (`--logits-processors`).

### API Call Example

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
    max_tokens=4096,
    temperature=0.0,
    extra_body={
        "vllm_xargs": {
            "ngram_size": 30,
            "window_size": 300,
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
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
    "--model", "deepseek-ai/DeepSeek-OCR-2", \
    "--trust-remote-code", "--dtype", "bfloat16", \
    "--max-model-len", "8192", "--gpu-memory-utilization", "0.90", \
    "--enforce-eager", "--no-enable-prefix-caching", \
    "--host", "0.0.0.0", "--port", "8000", \
    "--logits-processors", "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor"]
```

```bash
# ALWAYS build for amd64 when deploying to cloud
docker build --platform linux/amd64 -t deepseek-ocr2-vllm .
docker run --gpus all -p 8000:8000 deepseek-ocr2-vllm
```

## GPU Memory Requirements

| Configuration | VRAM Required |
|---------------|---------------|
| bf16, single GPU | ~8-10 GB |
| bf16 + KV cache (8192 ctx) | ~12-14 GB |
| bf16 + 0.90 util | ~20 GB (g5.xlarge A10G tested) |
| Multi-GPU (TP=2) | ~8 GB per GPU |

## Tested Performance (g5.xlarge / A10G 24GB)

| Metric | Value |
|--------|-------|
| Throughput | ~61 tok/s |
| Page processing time | 16-26s per page |
| 2-page invoice total | ~42s |

## Prompt Templates

| Task | Prompt |
|------|--------|
| Document to Markdown | `Convert the document to markdown.` |
| Markdown with bounding boxes | `<\|grounding\|>Convert the document to markdown.` |
| Plain text OCR | `Convert the document to text.` |
| Free OCR question | `What is the total amount on this invoice?` |

## Configuration Reference

### Engine Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--trust-remote-code` | required | Model uses custom HF code |
| `--dtype bfloat16` | recommended | Model precision |
| `--max-model-len 8192` | 8192 | Max context length |
| `--gpu-memory-utilization` | 0.90 | Fraction of GPU memory |
| `--enforce-eager` | recommended | Disable CUDA graphs for stability |
| `--no-enable-prefix-caching` | recommended | OCR doesn't benefit |
| `--logits-processors` | required | Must include NGramPerReqLogitsProcessor |

### Sampling Parameters (via `vllm_xargs`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.0 | Greedy decoding for OCR |
| `max_tokens` | 4096 | Max output tokens |
| `ngram_size` | 30 | N-gram repetition check size |
| `window_size` | 300 | Sliding window (default 90 too small for invoices) |
| `whitelist_token_ids` | [128821, 128822] | Table tokens (`<td>`, `</td>`) exempt from ban |
