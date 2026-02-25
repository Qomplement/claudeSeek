---
id: known-issues
title: Known Issues
sidebar_position: 5
---

# Known Issues & Workarounds

## 1. `transformers>=4.48` Breaks Accuracy

**Status:** Open — [PR #33389](https://github.com/vllm-project/vllm/pull/33389)

`_update_causal_mask()` was removed from `Qwen2Model` in `transformers>=4.48`. This breaks the hybrid attention mask in `deepencoder2.py`, causing accuracy degradation.

**Workaround:**
```bash
pip install transformers==4.46.3
```

## 2. BOS Token Missing in Chat Template

**Status:** Fixed — [PR #33642](https://github.com/vllm-project/vllm/pull/33642) (merged Feb 3, 2026)

Update to a vLLM version that includes this PR.

## 3. vLLM v1 Engine Logits Processor Migration

**Status:** By design (vLLM 0.11+)

Per-request `logits_processors` in `SamplingParams` no longer works. Use `AdapterLogitsProcessor`:

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

## 4. Initialization Order (vLLM v1)

The OCR processor must **not** be invoked before the engine is fully initialized:

```python
# CORRECT
engine = AsyncLLMEngine.from_engine_args(args)
inputs = prepare_inputs(image)

# WRONG (will crash)
inputs = prepare_inputs(image)
engine = AsyncLLMEngine.from_engine_args(args)
```

## 5. `DeepseekOCR2ForCausalLM` Not in Registry

**Status:** Fixed — [PR #33165](https://github.com/vllm-project/vllm/pull/33165) (merged Feb 2, 2026)

If you see `Model architectures ['DeepseekOCR2ForCausalLM'] are not supported`, your vLLM is too old. Options:
1. Build vLLM from `main`
2. Use out-of-tree registration
3. Use bundled vLLM 0.8.5

## 6. Prefix Caching Incompatibility

Always disable prefix caching — OCR workloads don't reuse prefixes:

```bash
vllm serve ... --no-enable-prefix-caching
```

## 7. Platform Mismatch (arm64 vs amd64)

Always build for the target platform:

```bash
docker build --platform linux/amd64 -t deepseek-ocr2-vllm .
```
