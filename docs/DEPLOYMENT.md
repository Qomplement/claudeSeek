# Deployment Guide

## Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n deepseek-ocr2 python=3.12.9
conda activate deepseek-ocr2

# Install PyTorch (CUDA 11.8)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install vLLM 0.16+ (has native DeepSeek-OCR-2 support)
pip install 'vllm>=0.16'

# Install flash-attention
pip install flash-attn==2.7.3 --no-build-isolation
```

### Verify Installation

```python
from vllm import ModelRegistry

archs = ModelRegistry.get_supported_archs()
assert "DeepseekOCR2ForCausalLM" in archs, "DeepSeek-OCR-2 not registered!"
print("DeepSeek-OCR-2 support confirmed")
```

---

## Deployment Options

### 1. Offline Batch Processing

Best for processing batches of documents/images.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR-2",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=8192,
    enforce_eager=False,
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

# Single image
prompt = "<image>\n<|grounding|>Convert the document to markdown."
result = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": "document.png"}},
    sampling_params=sampling_params,
)
print(result[0].outputs[0].text)
```

### 2. Online API Server

Best for serving OCR as an API endpoint.

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

> **Important:** The logits processor module is `deepseek_ocr` (not `deepseek_ocr2`) and the flag uses hyphens (`--logits-processors`). The `NGramPerReqLogitsProcessor` is critical for preventing repetitive output.

#### Calling the API

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Load image
with open("document.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-OCR-2",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {
                    "type": "text",
                    "text": "<|grounding|>Convert the document to markdown.",
                },
            ],
        }
    ],
    max_tokens=8192,
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

#### cURL Example

```bash
IMAGE_B64=$(base64 -i document.png)

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-OCR-2",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,'$IMAGE_B64'"}},
          {"type": "text", "text": "<|grounding|>Convert the document to markdown."}
        ]
      }
    ],
    "max_tokens": 8192,
    "temperature": 0.0,
    "vllm_xargs": {
      "ngram_size": 30,
      "window_size": 300,
      "whitelist_token_ids": [128821, 128822]
    }
  }'
```

### 3. Docker Deployment

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
# Build (ALWAYS use linux/amd64 for ECS/cloud deployments)
docker build --platform linux/amd64 -t deepseek-ocr2-vllm .

# Run locally
docker run --gpus all -p 8000:8000 deepseek-ocr2-vllm
```

---

## GPU Memory Requirements

| Configuration | VRAM Required | Notes |
|---------------|---------------|-------|
| bf16, single GPU | ~8-10 GB | 3B params × 2 bytes |
| bf16 + KV cache (8192 ctx) | ~12-14 GB | Depends on batch size |
| bf16 + 0.75 util | ~16 GB GPU recommended | Leaves headroom |
| Multi-GPU (TP=2) | ~8 GB per GPU | tensor_parallel_size=2 |

---

## Prompt Templates

### Document to Markdown (with layout)
```
<image>\n<|grounding|>Convert the document to markdown.
```

### Plain Text OCR (no layout)
```
<image>\nConvert the document to text.
```

### Free OCR (specific question)
```
<image>\nWhat is the total amount on this invoice?
```

---

## Configuration Reference

### Engine Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--trust-remote-code` | required | Model uses custom HF code |
| `--dtype bfloat16` | recommended | Model precision |
| `--max-model-len 8192` | 8192 | Max context length |
| `--gpu-memory-utilization` | 0.9 | Fraction of GPU memory to use |
| `--no-enable-prefix-caching` | recommended | OCR doesn't benefit from prefix caching |
| `--tensor-parallel-size` | 1 | Number of GPUs |
| `--enforce-eager` | False | Set True to disable CUDA graphs (debugging) |

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.0 | Greedy decoding for OCR |
| `max_tokens` | 8192 | Max output tokens |
| `skip_special_tokens` | False | Keep structural tokens in output |
| `ngram_size` | 20 | N-gram repetition check size (via `vllm_xargs`). Official: 20 for images/PDFs, 40 for batch eval |
| `window_size` | 90 | Sliding window for repetition check. Official: 90 for images, 50 for PDFs |
| `whitelist_token_ids` | [128821, 128822] | Table tokens (`<td>`, `</td>`) exempt from n-gram ban |

---

## Troubleshooting

### "Model architectures not supported"

Your vLLM doesn't include PR #33165. Options:
1. Build vLLM from source (`git clone` + `pip install -e .`)
2. Use runtime registration (see VLLM_INTEGRATION.md)
3. Fallback to bundled vLLM 0.8.5

### Accuracy degradation / garbled output

Likely the `transformers>=4.48` issue (PR #33389). Fix:
```bash
pip install transformers==4.46.3
```

### Repetitive output (repeated paragraphs/rows)

NGram logits processor not active. Ensure you pass `extra_args` or use `--logits_processors` flag.

### OOM (Out of Memory)

- Reduce `--gpu-memory-utilization` (e.g., 0.6)
- Reduce `--max-model-len` (e.g., 4096)
- Use `--tensor-parallel-size 2` with multiple GPUs
- Use `--enforce-eager True` (disables CUDA graphs, uses less memory)

### Slow first request

Model warmup + CUDA graph compilation. Subsequent requests will be faster. Use `--enforce-eager True` to skip graph compilation (trades throughput for faster startup).
