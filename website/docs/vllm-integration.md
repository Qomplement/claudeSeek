---
id: vllm-integration
title: vLLM Integration
sidebar_position: 3
---

# vLLM Integration Guide

## Model Registry System

vLLM uses a dictionary-based registry in `vllm/model_executor/models/registry.py` mapping HuggingFace architecture names to vLLM module/class pairs.

### DeepSeek-OCR-2 Registry Entry

```python
# In _MULTIMODAL_MODELS:
"DeepseekOCR2ForCausalLM": ("deepseek_ocr2", "DeepseekOCR2ForCausalLM"),
```

This was added by [PR #33165](https://github.com/vllm-project/vllm/pull/33165) (merged Feb 2, 2026).

### Files Added to vLLM

| File | Lines | Purpose |
|------|-------|---------|
| `vllm/model_executor/models/deepseek_ocr2.py` | 444 | Main model class |
| `vllm/model_executor/models/deepencoder2.py` | 283 | DeepEncoder V2 vision component |
| `vllm/transformers_utils/processors/deepseek_ocr2.py` | 320 | HF processor integration |

## Three Ways to Register

### 1. In-Tree (fork vLLM)

Add to `_MULTIMODAL_MODELS` in `registry.py`:

```python
"DeepseekOCR2ForCausalLM": ("deepseek_ocr2", "DeepseekOCR2ForCausalLM"),
```

### 2. Runtime Registration

```python
from vllm import ModelRegistry

ModelRegistry.register_model(
    "DeepseekOCR2ForCausalLM",
    "my_module:DeepseekOCR2ForCausalLM"
)
```

### 3. Plugin System (pip-installable)

```python
# setup.py
setup(
    name='deepseek-ocr2-vllm-plugin',
    entry_points={
        'vllm.general_plugins': [
            'deepseek_ocr2 = deepseek_ocr2_plugin:register'
        ]
    }
)
```

## Logits Processor System (vLLM 0.11+)

### Old Way (v0 -- removed)

```python
# NO LONGER WORKS in vLLM 0.11+
sampling_params = SamplingParams(
    logits_processors=[my_processor_instance]
)
```

### New Way (AdapterLogitsProcessor)

Logits processors are now **global** (instantiated at engine init) and manage per-request state internally.

```python
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

class NGramPerReqLogitsProcessor(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        if not params.extra_args:
            return None
        ngram_size = params.extra_args.get("ngram_size")
        if ngram_size is None:
            return None
        return NoRepeatNGramLogitsProcessor(
            ngram_size=ngram_size,
            window_size=params.extra_args.get("window_size", 90),
            whitelist_token_ids=params.extra_args.get("whitelist_token_ids", set()),
        )
```

Per-request parameters go via `SamplingParams.extra_args`:

```python
sampling_params = SamplingParams(
    extra_args={"ngram_size": 30, "window_size": 90}
)
```

### Loading Logits Processors

```bash
# CLI
vllm serve model_name --logits_processors my_module:MyLogitsProcessor

# Python
llm = LLM(model="...", logits_processors=[MyLogitsProcessor])
```

## v0 to v1 Engine Migration

| Aspect | v0 (removed in 0.11) | v1 (current) |
|--------|----------------------|--------------|
| Logits processors | Per-request via `SamplingParams` | Global `AdapterLogitsProcessor` + `extra_args` |
| `SamplingMetadata` | `vllm.model_executor` | `vllm.v1.sample.metadata` |
| `_call_hf_processor` | Fixed signature | Must accept `**kwargs` |

## Key vLLM Source Files

| File | Purpose |
|------|---------|
| `vllm/model_executor/models/registry.py` | Architecture to class mapping |
| `vllm/model_executor/models/deepseek_ocr2.py` | DeepSeek-OCR v2 model |
| `vllm/model_executor/models/deepencoder2.py` | DeepEncoder V2 vision |
| `vllm/v1/sample/logits_processor/interface.py` | `LogitsProcessor` ABC |
| `vllm/v1/sample/logits_processor/__init__.py` | `AdapterLogitsProcessor` |
