---
id: known-issues
title: Known Issues
sidebar_position: 5
---

# Known Issues & Workarounds

## 1. `transformers>=4.48` Breaks Accuracy

**Status:** Open — [PR #33389](https://github.com/vllm-project/vllm/pull/33389)

`_update_causal_mask()` was removed from `Qwen2Model` in `transformers>=4.48`. This breaks the hybrid attention mask in `deepencoder2.py`, causing accuracy degradation.

**Tested impact (vLLM 0.16.0 + transformers 4.57.6 on g5.xlarge / A10G):**
The model can identify document structure (headers, tables, sections) but hallucinates specific text content. For example, on a Mexican CFDI invoice it correctly found the company name and table layout but invented bank names and account numbers.

**Workaround:**
vLLM 0.16.0 requires `transformers>=4.56.0`, so pinning to 4.46.3 is not possible with the latest vLLM. Wait for PR #33389 to merge.

## 2. BOS Token Missing in Chat Template

**Status:** Fixed — [PR #33642](https://github.com/vllm-project/vllm/pull/33642) (merged Feb 3, 2026)

Update to a vLLM version that includes this PR.

## 3. vLLM v1 Engine Logits Processor Migration

**Status:** By design (vLLM 0.11+)

Per-request `logits_processors` in `SamplingParams` no longer works. Use the built-in `NGramPerReqLogitsProcessor`:

```python
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

# Register globally at engine init
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR-2",
    logits_processors=[NGramPerReqLogitsProcessor],
)

# Pass parameters per-request via extra_args
sampling_params = SamplingParams(
    extra_args={"ngram_size": 20, "window_size": 90, "whitelist_token_ids": [128821, 128822]}
)
```

**For the OpenAI-compatible API server:**

```bash
# Server flag (hyphens, not underscores; module is deepseek_ocr, not deepseek_ocr2)
--logits-processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor
```

```json
// Client: use vllm_xargs in the request body
{
  "vllm_xargs": {
    "ngram_size": 20,
    "window_size": 90,
    "whitelist_token_ids": [128821, 128822]
  }
}
```

> **Note:** Official defaults are `ngram_size=20`, `window_size=90` for images and `window_size=50` for PDFs. Repetition issues with these defaults may indicate the transformers accuracy bug (Issue 1).

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

If you see `Model architectures ['DeepseekOCR2ForCausalLM'] are not supported`, your vLLM is too old. Use vLLM >= 0.16.

## 6. Prefix Caching Incompatibility

Always disable prefix caching — OCR workloads don't reuse prefixes:

```bash
--no-enable-prefix-caching
```

## 7. Platform Mismatch (arm64 vs amd64)

Always build for the target platform:

```bash
docker build --platform linux/amd64 -t deepseek-ocr2-vllm .
```
