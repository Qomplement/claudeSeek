# Known Issues & Workarounds

## Issue 1: `transformers>=4.48` Breaks Hybrid Attention Mask

**Status:** Open — [PR #33389](https://github.com/vllm-project/vllm/pull/33389)

**Problem:**
`_update_causal_mask()` was removed from `Qwen2Model` in `transformers>=4.48`. The DeepEncoder V2 component relies on this method to create its hybrid attention mask (bidirectional for image tokens, causal for query tokens). Without it, all tokens get the same mask, degrading OCR accuracy.

**Symptoms:**
- Garbled or inaccurate OCR output
- Wrong reading order
- Missing text segments

**Workaround:**
```bash
pip install transformers==4.46.3
```

**Proper fix (from PR #33389):**
Auto-detect transformers version and use dual code paths — one for `<4.48` (uses `_update_causal_mask`) and one for `>=4.48` (custom mask construction).

---

## Issue 2: BOS Token Missing in Chat Template

**Status:** Fixed — [PR #33642](https://github.com/vllm-project/vllm/pull/33642) (merged Feb 3, 2026)

**Problem:**
The initial chat template registration didn't include the BOS (Beginning of Sequence) token, causing issues with the model's expected input format.

**Fix:** Update to a vLLM version that includes this PR, or build from main.

---

## Issue 3: vLLM v1 Engine Logits Processor Migration

**Status:** Resolved (by design in vLLM 0.11+)

**Problem:**
vLLM 0.11 removed the v0 engine. Per-request `logits_processors` in `SamplingParams` no longer work. The `NoRepeatNGramLogitsProcessor` critical for OCR quality must be adapted.

**Old way (broken):**
```python
sampling_params = SamplingParams(
    logits_processors=[NoRepeatNGramLogitsProcessor(30, 90, {128821, 128822})]
)
```

**New way:**
```python
# Register globally at engine init
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR-2",
    logits_processors=[NGramPerReqLogitsProcessor],  # class, not instance
)

# Pass parameters per-request via extra_args
sampling_params = SamplingParams(
    extra_args={"ngram_size": 30, "window_size": 90, "whitelist_token_ids": {128821, 128822}}
)
```

---

## Issue 4: Initialization Order Constraint (vLLM v1)

**Status:** By design

**Problem:**
In vLLM v1, the OCR processor must NOT be invoked before the engine is fully initialized. This differs from v0 behavior.

**Wrong:**
```python
inputs = prepare_inputs(image)  # crashes — engine not ready
engine = AsyncLLMEngine.from_engine_args(args)
```

**Correct:**
```python
engine = AsyncLLMEngine.from_engine_args(args)  # engine first
inputs = prepare_inputs(image)  # then prepare inputs
```

---

## Issue 5: `DeepseekOCR2ForCausalLM` Not in Registry

**Status:** Fixed — [PR #33165](https://github.com/vllm-project/vllm/pull/33165) (merged Feb 2, 2026)

**Problem:**
On vLLM releases before this PR, loading the model gives:
```
Model architectures ['DeepseekOCR2ForCausalLM'] are not supported for now.
```

**Fix options:**
1. Build vLLM from main (includes the PR)
2. Use out-of-tree registration:
   ```python
   from vllm import ModelRegistry
   ModelRegistry.register_model("DeepseekOCR2ForCausalLM", "my_module:DeepseekOCR2ForCausalLM")
   ```
3. Use bundled vLLM 0.8.5 from official repo

---

## Issue 6: Prefix Caching Incompatibility

**Status:** Known limitation

**Problem:**
OCR workloads don't reuse prompt prefixes (each image is unique), so prefix caching provides no benefit and can cause issues.

**Fix:**
Always disable prefix caching:
```bash
vllm serve ... --no-enable-prefix-caching
```
```python
llm = LLM(..., enable_prefix_caching=False)
```

---

## Issue 7: Platform Mismatch (arm64 vs amd64)

**Status:** User error (but common)

**Problem:**
Building Docker images on Apple Silicon (arm64) and deploying to cloud/ECS (amd64) causes crashes.

**Fix:**
Always specify platform when building:
```bash
docker build --platform linux/amd64 -t deepseek-ocr2-vllm .
```
