# Deployment Guide

## Version Requirements (Tested & Working)

These exact versions were tested end-to-end on AWS EC2 g5.xlarge (NVIDIA A10G, 24GB VRAM).

### Core Stack

| Component | Version | Notes |
|-----------|---------|-------|
| **Python** | 3.10.12 | Ubuntu 22.04 system Python. 3.10+ required by vLLM |
| **vLLM** | 0.16.0 | First version with native `DeepseekOCR2ForCausalLM` support |
| **transformers** | 4.57.6 | vLLM 0.16.0 requires `>=4.56.0`. See accuracy bug note below |
| **PyTorch** | 2.9.1+cu128 | Installed automatically by vLLM |
| **CUDA (runtime)** | 12.8 | Via PyTorch wheel. Driver must support this |
| **NVIDIA Driver** | 580.126.09 | Must support CUDA 12.8+ |

### Key Dependencies (installed automatically by vLLM)

| Package | Version | Purpose |
|---------|---------|---------|
| tokenizers | 0.22.2 | Fast tokenization |
| huggingface_hub | 0.36.2 | Model downloads |
| safetensors | 0.7.0 | Model weight loading |
| pillow | 12.1.1 | Image processing |
| numpy | 2.2.6 | Array operations |
| triton | 3.5.1 | GPU kernel compilation |
| xformers | 0.0.28.post3 | Memory-efficient attention |
| torchvision | 0.24.1 | Image transforms |
| torchaudio | 2.9.1 | (dependency, not directly used) |

### Hardware Tested

| Component | Spec |
|-----------|------|
| **GPU** | NVIDIA A10G (24 GB VRAM, compute capability 8.6) |
| **Instance** | AWS EC2 g5.xlarge (4 vCPU, 16 GB RAM) |
| **OS** | Ubuntu 22.04.5 LTS |
| **AMI** | ami-0669ac5db8b1292fe (Deep Learning Base OSS Nvidia Driver GPU AMI) |

### Version Compatibility Notes

**vLLM version:**
- `< 0.16.0` — Does NOT have `DeepseekOCR2ForCausalLM` in the model registry. You'd need to register it manually.
- `>= 0.16.0` — Native support. Use this.

**transformers version:**
- `< 4.48` — Best accuracy. The `_update_causal_mask()` method works correctly for DeepSeek-OCR-2's Visual Causal Flow.
- `>= 4.48` — **Accuracy bug.** `_update_causal_mask()` was removed from Qwen2Model, breaking the hybrid attention mask. The model can still identify document structure but hallucinates text content. [PR #33389](https://github.com/vllm-project/vllm/pull/33389) fixes this but hasn't merged yet.
- `>= 4.56.0` — Required by vLLM 0.16.0. You **cannot** pin to 4.46.3 with vLLM 0.16.

**CUDA version:**
- CUDA 12.x required (PyTorch 2.9.1 ships with CUDA 12.8 runtime)
- NVIDIA driver must be >= 525.60.13 for CUDA 12.x support
- No CUDA on Apple Silicon / AMD GPUs — vLLM requires NVIDIA GPUs

**flash-attn:**
- NOT required (not installed in our working setup)
- vLLM uses xformers instead
- If you want it: `pip install flash-attn --no-build-isolation` (requires matching CUDA toolkit)

---

## Quick Start (Bare Metal / EC2)

### 1. Install vLLM

```bash
# On a fresh Ubuntu 22.04 with NVIDIA GPU drivers already installed:
pip3 install vllm==0.16.0

# This automatically installs PyTorch, transformers, and all dependencies
# Verify:
python3 -c "import vllm; print(vllm.__version__)"  # 0.16.0
python3 -c "import torch; print(torch.cuda.is_available())"  # True
```

### 2. Verify DeepSeek-OCR-2 Support

```python
python3 -c "
from vllm import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert 'DeepseekOCR2ForCausalLM' in archs, 'Not supported!'
print('DeepSeek-OCR-2 support confirmed')
"
```

### 3. Start the Server

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-OCR-2 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000 \
    --logits-processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor
```

First startup downloads the model (~6 GB). Subsequent starts use the cached model.

### 4. Test It

```bash
# Check the server is running:
curl http://localhost:8000/v1/models

# Send an OCR request (replace with your image):
IMAGE_B64=$(base64 -i document.png)

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-OCR-2",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMAGE_B64"'"}},
        {"type": "text", "text": "Convert the document to markdown."}
      ]
    }],
    "max_tokens": 8192,
    "temperature": 0.0,
    "vllm_xargs": {
      "ngram_size": 20,
      "window_size": 90,
      "whitelist_token_ids": [128821, 128822]
    }
  }'
```

---

## AWS EC2 Setup (Step by Step)

For devs starting from scratch on AWS:

### 1. Launch Instance

- **Instance type:** g5.xlarge (1x A10G 24GB, $1.006/hr on-demand)
- **AMI:** `ami-0669ac5db8b1292fe` (Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 22.04)
- **Storage:** 100 GB gp3 (model is ~6 GB, plus OS and dependencies)
- **Security group:** Open ports 22 (SSH) and 8000 (vLLM API)
- **Region:** us-east-1

### 2. SSH In and Install

```bash
ssh -i your-key.pem ubuntu@<instance-ip>

# The AMI already has NVIDIA drivers. Verify:
nvidia-smi  # Should show A10G, 24 GB

# Install vLLM (installs everything: PyTorch, transformers, etc.)
pip3 install vllm==0.16.0

# Start the server (first run downloads the model, takes ~5 min)
python3 -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-OCR-2 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000 \
    --logits-processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor
```

### 3. Run in Background

```bash
# Write startup script
cat > /tmp/start_vllm.sh << 'EOF'
#!/bin/bash
python3 -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-OCR-2 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000 \
    --logits-processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor
EOF
chmod +x /tmp/start_vllm.sh

# Run in background
nohup /tmp/start_vllm.sh > /tmp/vllm.log 2>&1 &

# Check if ready (wait for "Application startup complete"):
tail -f /tmp/vllm.log
```

---

## Deployment Options

### Offline Batch Processing

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR-2",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
    enable_prefix_caching=False,
    tensor_parallel_size=1,  # increase for multi-GPU
    logits_processors=["vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor"],
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    skip_special_tokens=False,
    extra_args={
        "ngram_size": 20,
        "window_size": 90,
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

### Python API Client

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

### Docker Deployment

```dockerfile
FROM vllm/vllm-openai:latest

# Download model at build time (optional, speeds up startup)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-OCR-2')"

EXPOSE 8000

CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "deepseek-ai/DeepSeek-OCR-2", \
     "--trust-remote-code", \
     "--dtype", "bfloat16", \
     "--max-model-len", "8192", \
     "--gpu-memory-utilization", "0.90", \
     "--enforce-eager", \
     "--no-enable-prefix-caching", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--logits-processors", "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor"]
```

```bash
# ALWAYS build for amd64 when deploying to cloud/ECS
docker build --platform linux/amd64 -t deepseek-ocr2-vllm .
docker run --gpus all -p 8000:8000 deepseek-ocr2-vllm
```

---

## GPU Memory Requirements

| Configuration | VRAM Required | Notes |
|---------------|---------------|-------|
| bf16, single GPU | ~6.3 GB | Model weights (3B params × 2 bytes) |
| bf16 + KV cache (8192 ctx) | ~20 GB | With gpu_memory_utilization=0.90 |
| Multi-GPU (TP=2) | ~10 GB per GPU | tensor_parallel_size=2 |

### Minimum GPU Requirements

| GPU | VRAM | Works? |
|-----|------|--------|
| A10G (g5.xlarge) | 24 GB | Yes (tested) |
| T4 (g4dn.xlarge) | 16 GB | Maybe — reduce gpu_memory_utilization to 0.70, max_model_len to 4096 |
| A100 40GB | 40 GB | Yes — can increase max_model_len or batch size |
| L4 (g6.xlarge) | 24 GB | Should work (same VRAM as A10G) |
| Apple Silicon | N/A | No — vLLM requires NVIDIA CUDA GPUs |

---

## Tested Performance (g5.xlarge / A10G 24GB)

| Metric | Value |
|--------|-------|
| Throughput | ~61 tokens/sec |
| Page processing time | 16-26 seconds per page |
| 2-page invoice total | ~42 seconds |
| Model load time | ~11 seconds |
| GPU memory used | ~21 GB (with 0.90 utilization) |

---

## Prompt Templates

| Task | Prompt | Notes |
|------|--------|-------|
| Document to Markdown | `Convert the document to markdown.` | Clean text output |
| Markdown + Bounding Boxes | `<\|grounding\|>Convert the document to markdown.` | Includes `[[x1,y1,x2,y2]]` coordinates |
| Plain text OCR | `Convert the document to text.` | No formatting |
| Specific question | `What is the total amount on this invoice?` | Free-form Q&A |

---

## Configuration Reference

### Server Arguments

| Argument | Value | Required? | Description |
|----------|-------|-----------|-------------|
| `--model` | `deepseek-ai/DeepSeek-OCR-2` | Yes | HuggingFace model ID |
| `--trust-remote-code` | (flag) | Yes | Model uses custom HF code |
| `--dtype bfloat16` | `bfloat16` | Recommended | Model precision |
| `--max-model-len` | `8192` | Recommended | Max context (model max is 8192) |
| `--gpu-memory-utilization` | `0.90` | Recommended | Fraction of GPU VRAM to use |
| `--enforce-eager` | (flag) | Recommended | Disables CUDA graphs for stability |
| `--no-enable-prefix-caching` | (flag) | Recommended | OCR workloads don't reuse prefixes |
| `--logits-processors` | `vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor` | Yes | Prevents repetitive output |
| `--tensor-parallel-size` | `1` | Optional | Set >1 for multi-GPU |

### Request Parameters (via `vllm_xargs`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | `0.0` | Greedy decoding for OCR |
| `max_tokens` | `8192` | Max output tokens (model max) |
| `ngram_size` | `20` | N-gram size for repetition check (official: 20 for images, 40 for batch) |
| `window_size` | `90` | Sliding window for check (official: 90 for images, 50 for PDFs) |
| `whitelist_token_ids` | `[128821, 128822]` | `<td>` and `</td>` tokens exempt from n-gram ban |

---

## Troubleshooting

### "Model architectures not supported"
Your vLLM is too old (< 0.16.0). Upgrade:
```bash
pip3 install vllm==0.16.0
```

### Accuracy degradation / hallucinated text
Known issue with `transformers >= 4.48` (PR #33389). The model reads document structure correctly but invents text content. No workaround currently — vLLM 0.16.0 requires transformers >= 4.56.0.

### Repetitive output (repeated paragraphs/rows)
NGram logits processor not active. Check:
1. Server was started with `--logits-processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor`
2. Request includes `vllm_xargs` with `ngram_size` and `window_size`

### OOM (Out of Memory)
- Reduce `--gpu-memory-utilization` (e.g., `0.70`)
- Reduce `--max-model-len` (e.g., `4096`)
- Use `--tensor-parallel-size 2` with multiple GPUs
- Kill any zombie Python/CUDA processes: `killall -9 python3`

### GPU memory not freed after stopping server
vLLM spawns child EngineCore processes that can survive the parent. Check with:
```bash
nvidia-smi  # Look for "VLLM::EngineCore" processes
kill -9 <PID>
```

### Slow first request
Model warmup on first inference. Subsequent requests are faster (~61 tok/s on A10G).
