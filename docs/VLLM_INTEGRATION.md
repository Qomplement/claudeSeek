# vLLM Integration Guide

## How vLLM Model Registration Works

### Registry System

vLLM uses a dictionary-based registry in `vllm/model_executor/models/registry.py` that maps HuggingFace architecture names (from `config.json`) to vLLM module/class pairs.

```python
# Structure:
_MULTIMODAL_MODELS = {
    "ArchitectureName": ("module_name", "ClassName"),
    # module_name is relative to vllm/model_executor/models/
}

# All categories merged:
_VLLM_MODELS = {
    **_TEXT_GENERATION_MODELS,
    **_EMBEDDING_MODELS,
    **_CROSS_ENCODER_MODELS,
    **_MULTIMODAL_MODELS,
    **_SPECULATIVE_DECODING_MODELS,
    **_TRANSFORMERS_SUPPORTED_MODELS,
    **_TRANSFORMERS_BACKEND_MODELS,
}
```

### DeepSeek-OCR-2 Registry Entry

```python
_MULTIMODAL_MODELS = {
    "DeepseekOCR2ForCausalLM": ("deepseek_ocr2", "DeepseekOCR2ForCausalLM"),
    # ...
}
```

This tells vLLM: when the HF config says `architectures: ["DeepseekOCR2ForCausalLM"]`, import `DeepseekOCR2ForCausalLM` from `vllm.model_executor.models.deepseek_ocr2`.

---

## Three Ways to Register a Custom Model

### 1. In-Tree (Fork vLLM)

Add your entry to the appropriate dict in `registry.py`:

```python
_MULTIMODAL_MODELS = {
    "DeepseekOCR2ForCausalLM": ("deepseek_ocr2", "DeepseekOCR2ForCausalLM"),
}
```

Create the model file at `vllm/model_executor/models/deepseek_ocr2.py`.

### 2. Runtime Registration (Out-of-Tree)

```python
from vllm import ModelRegistry

# Lazy string-based (preferred — avoids premature CUDA init)
ModelRegistry.register_model(
    "DeepseekOCR2ForCausalLM",
    "my_package.deepseek_ocr2:DeepseekOCR2ForCausalLM"
)

# Or direct class reference
from my_package.deepseek_ocr2 import DeepseekOCR2ForCausalLM
ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)
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

# deepseek_ocr2_plugin/__init__.py
def register():
    from vllm import ModelRegistry
    if "DeepseekOCR2ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "DeepseekOCR2ForCausalLM",
            "deepseek_ocr2_plugin.model:DeepseekOCR2ForCausalLM",
        )
```

---

## vLLM Multimodal Model Requirements

A multimodal model class must implement these interfaces:

```python
@MULTIMODAL_REGISTRY.register_processor(
    MyMultiModalProcessor,
    info=MyProcessingInfo,
    dummy_inputs=MyDummyInputsBuilder,
)
class MyVLMForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    # Required attributes
    hf_to_vllm_mapper = WeightsMapper(...)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        ...

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        ...

    def get_multimodal_embeddings(self, **kwargs):
        # Process images through vision pipeline
        ...

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None):
        # Merge vision embeddings into text sequence
        ...

    def load_weights(self, weights):
        ...
```

### Three Required Companion Classes

```python
class MyProcessingInfo(BaseProcessingInfo):
    """Manages config and processor instantiation."""
    def get_hf_config(self): ...
    def get_hf_processor(self): ...

class MyDummyInputsBuilder(BaseDummyInputsBuilder[MyProcessingInfo]):
    """Creates dummy inputs for vLLM's profiling/warmup."""
    def get_dummy_processor_inputs(self, seq_len, mm_counts): ...

class MyMultiModalProcessor(BaseMultiModalProcessor[MyProcessingInfo]):
    """Handles HF processor calls and prompt token replacements."""
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, **kwargs): ...
    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs): ...
```

---

## Logits Processor System (vLLM 0.11+ / v1 Engine)

### Old Way (v0 — REMOVED in 0.11.0)

```python
# NO LONGER WORKS
sampling_params = SamplingParams(
    logits_processors=[my_processor_instance]  # per-request, removed in v1
)
```

### New Way (v1 — AdapterLogitsProcessor)

Logits processors are now **global** (instantiated at engine init) and manage per-request state internally.

```python
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
from vllm.sampling_params import SamplingParams

class NGramPerReqLogitsProcessor(AdapterLogitsProcessor):
    """Wraps per-request NoRepeatNGramLogitsProcessor for v1 engine."""

    @classmethod
    def validate_params(cls, params: SamplingParams):
        """Validate extra_args at request submission time."""
        if params.extra_args:
            ngram_size = params.extra_args.get("ngram_size")
            if ngram_size is not None and ngram_size < 1:
                raise ValueError("ngram_size must be >= 1")

    def is_argmax_invariant(self) -> bool:
        return False  # Can change which token has highest logit

    def new_req_logits_processor(self, params: SamplingParams):
        """Create a per-request processor from extra_args, or None."""
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

### Loading Logits Processors

```bash
# CLI (Fully-Qualified Class Name)
vllm serve model_name --logits_processors my_module:MyLogitsProcessor

# Python API
llm = LLM(model="...", logits_processors=[MyLogitsProcessor])

# Per-request parameters via extra_args
sampling_params = SamplingParams(
    extra_args={"ngram_size": 30, "window_size": 90}
)
```

---

## v0 → v1 Engine Migration Summary

| Aspect | v0 (removed in 0.11) | v1 (current) |
|--------|----------------------|--------------|
| Engine classes | `LLMEngine`, `AsyncLLMEngine` | New implementations under same names |
| Logits processors | Per-request via `SamplingParams` | Global `AdapterLogitsProcessor` + `extra_args` |
| `SamplingMetadata` | `vllm.model_executor` | `vllm.v1.sample.metadata` |
| `_call_hf_processor` | Fixed signature | Must accept `**kwargs` |
| Scheduling | In-process | Isolated EngineCore process |

### Breaking Changes Checklist

- [ ] Replace per-request `logits_processors` with `AdapterLogitsProcessor` subclass
- [ ] Pass per-request params via `SamplingParams.extra_args`
- [ ] Update `SamplingMetadata` import path
- [ ] Add `**kwargs` to `_call_hf_processor` signatures
- [ ] Ensure engine is fully initialized before invoking model processors
- [ ] Use `hf_overrides={"architectures": [...]}` for architecture overrides

---

## Key vLLM Source Files Reference

| File | Purpose |
|------|---------|
| `vllm/model_executor/models/registry.py` | Architecture → class mapping |
| `vllm/model_executor/models/deepseek_ocr.py` | DeepSeek-OCR v1 (reference) |
| `vllm/model_executor/models/deepseek_ocr2.py` | DeepSeek-OCR v2 |
| `vllm/model_executor/models/deepencoder.py` | DeepEncoder v1 |
| `vllm/model_executor/models/deepencoder2.py` | DeepEncoder v2 |
| `vllm/v1/sample/logits_processor/interface.py` | `LogitsProcessor` ABC |
| `vllm/v1/sample/logits_processor/__init__.py` | `AdapterLogitsProcessor` |
| `vllm/logits_process.py` | `RequestLogitsProcessor` type |
| `vllm/transformers_utils/processors/deepseek_ocr2.py` | HF processor bridge |
| `vllm/transformers_utils/chat_templates/registry.py` | Chat templates |
| `tests/models/registry.py` | Test model examples |

---

## vLLM Version History (Relevant)

| Version | Date | Relevant Changes |
|---------|------|------------------|
| v0.8.5 | ~May 2025 | Bundled with DeepSeek-OCR-2 repo, day-0 support |
| v0.11.0 | Oct 4, 2025 | v0 engine removed, AdapterLogitsProcessor introduced |
| v0.14.0 | Jan 20, 2026 | Async scheduling default, PyTorch 2.9.1 |
| v0.15.0 | Jan 29, 2026 | Major release, 335 commits |
| v0.15.1 | Feb 4, 2026 | Security fixes, latest stable |

PR #33165 merged Feb 2, 2026 — between v0.15.0 and v0.15.1 releases.
