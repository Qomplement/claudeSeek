# DeepSeek-OCR-2 on vLLM — Compatibility & Deployment Guide

> Making [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) run on the latest [vLLM](https://docs.vllm.ai/en/latest/) inference engine.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Compatibility Status](#compatibility-status)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Known Issues & Fixes](#known-issues--fixes)
- [References](#references)

---

## Overview

**DeepSeek-OCR-2** is a 3B-parameter vision-language model for document OCR, released January 27, 2026 by DeepSeek AI. It introduces "Visual Causal Flow" — a learned causal token reordering system that replaces fixed raster-scan reading, achieving 33% better reading order accuracy and 33% lower repetition rates vs. v1.

### The Problem

DeepSeek-OCR-2 uses the architecture `DeepseekOCR2ForCausalLM`, which was **not** in the vLLM model registry at launch. Attempting to load it on standard vLLM produced:

```
Model architectures ['DeepseekOCR2ForCausalLM'] are not supported for now.
```

### The Solution

As of **February 2, 2026**, vLLM [PR #33165](https://github.com/vllm-project/vllm/pull/33165) merged native support. Building vLLM from `main` (or using a release that includes this PR) gives full support. For older vLLM versions, out-of-tree registration or the bundled vLLM 0.8.5 scripts work as fallbacks.

---

## Architecture

### Model Overview

```
DeepseekOCR2ForCausalLM (3B params, ~500M active per token)
├── Vision Pipeline
│   ├── SAM ViT-B (80M params) — image tokenizer
│   ├── Qwen2-0.5B (500M params) — encoder with hybrid attention (DeepEncoder V2)
│   └── Linear Projector (896 → 1280 dims)
└── Language Model
    └── DeepSeek-V2 3B MoE (12 layers, 64 routed experts, 2 shared, 6 active per token)
```

### Data Flow

```
Input Image (B, 3, 1024, 1024)
    │
    ▼
┌─────────────────────────────────────────────┐
│  SAM ViT-B                                  │
│  Patch embed → 12 ViT blocks → Neck → Down  │
│  Output: (B, 896, 16, 16) = 256 tokens      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Qwen2Decoder2Encoder (Visual Causal Flow)  │
│  Concat 256 image tokens + 256 query tokens │
│  Hybrid attention mask:                     │
│    - Image tokens: BIDIRECTIONAL            │
│    - Query tokens: CAUSAL + attend to all   │
│      image tokens                           │
│  24 Qwen2 layers → extract query outputs    │
│  Output: (B, 256, 896)                      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Linear Projector (896 → 1280)              │
│  Output: (B, 256, 1280)                     │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Token Merging                              │
│  Replace <image> placeholders with visual   │
│  embeddings in the text token sequence      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  DeepSeek-V2 3B MoE LLM                    │
│  12 layers, 64 experts, 6 active/token      │
│  Hidden: 1280, Vocab: 129,280               │
└─────────────────────────────────────────────┘
    │
    ▼
Output tokens (markdown, OCR text, etc.)
```

### Multi-Crop Strategy

| View | Resolution | Visual Tokens |
|------|-----------|---------------|
| Global | 1024×1024 | 256 |
| Local crop (each) | 768×768 | 144 |
| Local crops (0-6) | — | 0–864 |
| **Total range** | — | **256–1120** |

A learned `view_separator` token is inserted between global and local features.

### v1 vs v2 Architecture Comparison

| Feature | DeepSeek-OCR (v1) | DeepSeek-OCR-2 (v2) |
|---------|-------------------|---------------------|
| Architecture class | `DeepseekOCRForCausalLM` | `DeepseekOCR2ForCausalLM` |
| Vision encoder | CLIP-L-14 + SAM ViT-B | SAM ViT-B + Qwen2-0.5B |
| Projector input dim | 2048 (CLIP+SAM concat) | 896 (Qwen2 output) |
| Reading order | Fixed raster scan | Learned causal reordering |
| Max visual tokens | 1156 | 1120 |
| `config.json` model_type | `deepseek_vl_v2` | `deepseek_vl_v2` |

### Benchmark: v2 Improvements over v1

| Metric | v1 | v2 | Delta |
|--------|-----|-----|-------|
| OmniDocBench v1.5 | 87.36% | 91.09% | +3.73% |
| Reading Order Edit Dist | 0.085 | 0.057 | -33% |
| Formula Edit Dist | 0.236 | 0.198 | -16% |
| Table Edit Dist | 0.123 | 0.096 | -22% |
| Text Edit Dist | 0.073 | 0.048 | -34% |
| Online repetition rate | 6.25% | 4.17% | -33% |

---

## Compatibility Status

### vLLM Support Timeline

| Date | Event |
|------|-------|
| Jan 27, 2026 | DeepSeek-OCR-2 released with bundled vLLM 0.8.5 scripts |
| Feb 2, 2026 | [PR #33165](https://github.com/vllm-project/vllm/pull/33165) merged — native `DeepseekOCR2ForCausalLM` in vLLM |
| Feb 3, 2026 | [PR #33642](https://github.com/vllm-project/vllm/pull/33642) merged — BOS token fix for chat template |
| Open | [PR #33389](https://github.com/vllm-project/vllm/pull/33389) — accuracy fix for `transformers>=4.48` |

### Files Added to vLLM by PR #33165

| File | Lines | Purpose |
|------|-------|---------|
| `vllm/model_executor/models/deepseek_ocr2.py` | 444 | Main model class, multimodal processor, logits processor |
| `vllm/model_executor/models/deepencoder2.py` | 283 | DeepEncoder V2 vision component |
| `vllm/transformers_utils/processors/deepseek_ocr2.py` | 320 | HuggingFace processor integration |
| `vllm/model_executor/models/registry.py` | +1 | Registry entry |
| `vllm/transformers_utils/chat_templates/registry.py` | +1 | Chat template |
| `tests/models/registry.py` | +3 | Test registry |
| `docs/models/supported_models.md` | +1 | Docs |

### Registry Entry

In `vllm/model_executor/models/registry.py` → `_MULTIMODAL_MODELS`:

```python
"DeepseekOCR2ForCausalLM": ("deepseek_ocr2", "DeepseekOCR2ForCausalLM"),
```

### Deployment Options Summary

| Approach | vLLM Version | Effort | Status |
|----------|-------------|--------|--------|
| vLLM from `main` branch | latest | `pip install -e .` | Works now |
| vLLM release (≥ v0.15.x) | v0.15.0+ | `pip install vllm` | Verify registry has `deepseek_ocr2` |
| Out-of-tree registration | Any 0.11+ | Copy 3 files + register | Works on any modern vLLM |
| Bundled vLLM 0.8.5 | 0.8.5 pinned | Use official repo scripts | Works but locks old version |

---

## Setup & Installation

### Prerequisites

- CUDA 11.8+
- Python 3.12+
- GPU with sufficient VRAM (16GB+ recommended for bf16)

### Option A: vLLM from Source (Recommended)

Gets you the latest support including PR #33165 and all subsequent fixes.

```bash
# Create environment
conda create -n deepseek-ocr2 python=3.12.9
conda activate deepseek-ocr2

# Install PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Install vLLM from source
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .

# Install flash-attention
pip install flash-attn==2.7.3 --no-build-isolation
```

### Option B: vLLM Release

```bash
conda create -n deepseek-ocr2 python=3.12.9
conda activate deepseek-ocr2

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U vllm
pip install flash-attn==2.7.3 --no-build-isolation
```

Verify support:
```python
from vllm import ModelRegistry
assert "DeepseekOCR2ForCausalLM" in ModelRegistry.get_supported_archs()
```

### Option C: Out-of-Tree Registration (Older vLLM)

If your vLLM version doesn't include PR #33165, you can register the model externally:

```bash
# Clone the DeepSeek-OCR-2 repo for the vLLM integration files
git clone https://github.com/deepseek-ai/DeepSeek-OCR-2.git
```

Then register at runtime:

```python
from vllm import ModelRegistry

ModelRegistry.register_model(
    "DeepseekOCR2ForCausalLM",
    "path.to.deepseek_ocr2:DeepseekOCR2ForCausalLM"
)
```

### Option D: Bundled vLLM 0.8.5 (Fallback)

```bash
conda create -n deepseek-ocr2 python=3.12.9
conda activate deepseek-ocr2

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Install the pinned vLLM 0.8.5 wheel from DeepSeek's releases
# (Check https://github.com/deepseek-ai/DeepSeek-OCR-2/releases for the wheel URL)
pip install vllm==0.8.5
pip install flash-attn==2.7.3 --no-build-isolation
```

---

## Usage

### Offline Batch Inference (Python API)

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR-2",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=8192,
    enforce_eager=False,
    gpu_memory_utilization=0.75,
    enable_prefix_caching=False,
)

# Sampling params with NGram logits processor config
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    skip_special_tokens=False,
    extra_args={
        "ngram_size": 30,
        "window_size": 90,
        "whitelist_token_ids": {128821, 128822},  # <td>, </td> table tokens
    },
)

# Run inference
prompt = "<image>\n<|grounding|>Convert the document to markdown."
outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": "path/to/document.png"},
    },
    sampling_params=sampling_params,
)

print(outputs[0].outputs[0].text)
```

### Online Serving (OpenAI-Compatible API)

```bash
vllm serve deepseek-ai/DeepSeek-OCR-2 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75 \
    --no-enable-prefix-caching \
    --logits_processors vllm.model_executor.models.deepseek_ocr2:NGramPerReqLogitsProcessor
```

Then call via the OpenAI API:

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

with open("document.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-OCR-2",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": "<|grounding|>Convert the document to markdown."},
            ],
        }
    ],
    max_tokens=8192,
    temperature=0.0,
    extra_body={
        "ngram_size": 30,
        "window_size": 90,
        "whitelist_token_ids": [128821, 128822],
    },
)

print(response.choices[0].message.content)
```

### Using the Bundled Scripts (vLLM 0.8.5)

```bash
cd DeepSeek-OCR-2/DeepSeek-OCR2-vllm

# Edit config.py to set your paths
# Then run:
python run_dpsk_ocr2_image.py    # Single image (streaming output)
python run_dpsk_ocr2_pdf.py      # PDF processing (concurrent pages)
python run_dpsk_ocr2_eval_batch.py  # Batch evaluation
```

---

## Known Issues & Fixes

### 1. `transformers>=4.48` Breaks Accuracy

**Problem:** `_update_causal_mask()` was removed from `Qwen2Model` in `transformers>=4.48`. This breaks the hybrid attention mask in `deepencoder2.py` (bidirectional for image tokens, causal for query tokens), causing accuracy degradation.

**Fix (PR #33389 — still open):** Adds auto-detection of transformers version with dual code paths.

**Workaround:** Pin transformers:
```bash
pip install transformers==4.46.3
```

### 2. NGram Logits Processor Required for Quality

The OCR model **requires** the `NoRepeatNGramLogitsProcessor` to prevent repetitive outputs. Without it, you'll see repeated text blocks especially in tables and lists.

**Config:**
```python
extra_args={
    "ngram_size": 30,      # n-gram window to check for repeats
    "window_size": 90,      # sliding window size
    "whitelist_token_ids": {128821, 128822},  # tokens exempt from n-gram ban
}
```

### 3. vLLM v1 Engine Logits Processor API Change

If using vLLM 0.11+ and writing custom logits processors, the old per-request `SamplingParams.logits_processors` list is gone. Use the `AdapterLogitsProcessor` pattern:

```python
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

class MyOCRLogitsProcessor(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        ngram_size = params.extra_args.get("ngram_size")
        if ngram_size is None:
            return None
        return NoRepeatNGramLogitsProcessor(
            ngram_size=ngram_size,
            window_size=params.extra_args.get("window_size", 90),
            whitelist_token_ids=params.extra_args.get("whitelist_token_ids", set()),
        )
```

Pass via `extra_args` in `SamplingParams` instead of the old `logits_processors` list.

### 4. Initialization Order (vLLM 0.11+)

The OCR processor must **not** be invoked before the vLLM engine is fully initialized. Always create the engine first, then prepare inputs:

```python
# CORRECT order:
engine = AsyncLLMEngine.from_engine_args(engine_args)  # engine first
inputs = prepare_ocr_inputs(image)                      # then inputs

# WRONG order (will crash):
inputs = prepare_ocr_inputs(image)                      # don't do this first
engine = AsyncLLMEngine.from_engine_args(engine_args)
```

---

## HuggingFace Custom Code Files

DeepSeek-OCR-2 requires `trust_remote_code=True` because it uses custom modeling code not in upstream `transformers`:

| File | Size | Contents |
|------|------|----------|
| `modeling_deepseekocr2.py` | 39.2 kB | `DeepseekOCR2ForCausalLM`, `DeepseekOCR2Model`, `DeepseekOCR2Config`, image preprocessing, inference |
| `deepencoderv2.py` | 36.3 kB | `ImageEncoderViT` (SAM ViT-B), `Qwen2Decoder2Encoder`, `CustomQwen2Decoder`, hybrid attention mask |
| `modeling_deepseekv2.py` | 82.2 kB | `DeepseekV2ForCausalLM` — base MoE language model |
| `configuration_deepseek_v2.py` | 10.6 kB | `DeepseekV2Config` |
| `conversation.py` | 9.25 kB | Chat template / conversation formatting |

Key config.json excerpt:
```json
{
  "architectures": ["DeepseekOCR2ForCausalLM"],
  "model_type": "deepseek_vl_v2",
  "hidden_size": 1280,
  "num_hidden_layers": 12,
  "n_routed_experts": 64,
  "n_shared_experts": 2,
  "num_experts_per_tok": 6,
  "vocab_size": 129280,
  "vision_config": {
    "model_name": "deepencoderv2",
    "image_size": 1024,
    "width": {
      "sam_vit_b": { "layers": 12, "width": 768, "heads": 12 },
      "qwen2-0-5b": { "dim": 896 }
    }
  },
  "projector_config": {
    "input_dim": 896,
    "n_embed": 1280,
    "projector_type": "linear"
  }
}
```

---

## vLLM Internals Reference

### Model Registration API

**In-tree (fork vLLM):**
```python
# vllm/model_executor/models/registry.py
_MULTIMODAL_MODELS = {
    "DeepseekOCR2ForCausalLM": ("deepseek_ocr2", "DeepseekOCR2ForCausalLM"),
}
```

**Out-of-tree (runtime):**
```python
from vllm import ModelRegistry
ModelRegistry.register_model("DeepseekOCR2ForCausalLM", "my_module:DeepseekOCR2ForCausalLM")
```

**Plugin system (pip installable):**
```python
# setup.py
setup(
    name='deepseek-ocr2-vllm',
    entry_points={
        'vllm.general_plugins': [
            'deepseek_ocr2 = deepseek_ocr2_plugin:register'
        ]
    }
)

# deepseek_ocr2_plugin/__init__.py
def register():
    from vllm import ModelRegistry
    if "DeepseekOCR2ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "DeepseekOCR2ForCausalLM",
            "deepseek_ocr2_plugin.model:DeepseekOCR2ForCausalLM",
        )
```

### Key vLLM Source Files

| File | Purpose |
|------|---------|
| `vllm/model_executor/models/registry.py` | Model architecture → class mapping |
| `vllm/model_executor/models/deepseek_ocr.py` | DeepSeek-OCR v1 implementation |
| `vllm/model_executor/models/deepseek_ocr2.py` | DeepSeek-OCR v2 implementation |
| `vllm/model_executor/models/deepencoder2.py` | DeepEncoder V2 vision component |
| `vllm/v1/sample/logits_processor/interface.py` | `LogitsProcessor` base ABC |
| `vllm/v1/sample/logits_processor/__init__.py` | `AdapterLogitsProcessor` wrapper |
| `vllm/logits_process.py` | `RequestLogitsProcessor` type alias |
| `vllm/transformers_utils/processors/deepseek_ocr2.py` | HF processor integration |

---

## References

### Official Resources
- [DeepSeek-OCR-2 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR-2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR-2 Paper (arXiv:2601.20552)](https://arxiv.org/abs/2601.20552)

### vLLM
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [vLLM DeepSeek-OCR Recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- [PR #33165 — Add DeepSeek-OCR-2 Support](https://github.com/vllm-project/vllm/pull/33165)
- [PR #33389 — Accuracy Fix (open)](https://github.com/vllm-project/vllm/pull/33389)
- [PR #33642 — BOS Token Fix](https://github.com/vllm-project/vllm/pull/33642)
- [Custom Logits Processors Docs](https://docs.vllm.ai/en/stable/features/custom_logitsprocs/)
- [Logits Processors Design](https://docs.vllm.ai/en/latest/design/logits_processors/)
- [Model Registration Docs](https://docs.vllm.ai/en/stable/contributing/model/registration/)

### Community
- [vLLM Forum: Running DeepSeek-OCR-2](https://discuss.vllm.ai/t/how-to-run-deep-seek-ocr-2-in-vllm/2280)
- [HuggingFace Discussion: Architecture Not Supported](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/discussions/3)
- [DeepSeek-OCR Issue #231: vLLM 0.11 Compatibility](https://github.com/deepseek-ai/DeepSeek-OCR/issues/231)
- [vLLM Announcement on X](https://x.com/vllm_project/status/2016065526058090967)
