---
id: overview
title: Overview
sidebar_position: 1
slug: /overview
---

# DeepSeek-OCR-2 on vLLM

> Making [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) run on the latest [vLLM](https://docs.vllm.ai/en/latest/) inference engine.

## What is DeepSeek-OCR-2?

**DeepSeek-OCR-2** is a 3B-parameter vision-language model for document OCR, released January 27, 2026 by DeepSeek AI. It introduces "Visual Causal Flow" — a learned causal token reordering system that replaces fixed raster-scan reading, achieving **33% better reading order** accuracy and **33% lower repetition rates** vs. v1.

## The Problem

DeepSeek-OCR-2 uses the architecture `DeepseekOCR2ForCausalLM`, which was **not** in the vLLM model registry at launch. Attempting to load it on standard vLLM produced:

```
Model architectures ['DeepseekOCR2ForCausalLM'] are not supported for now.
```

## The Solution

As of **February 2, 2026**, vLLM [PR #33165](https://github.com/vllm-project/vllm/pull/33165) merged native support. Building vLLM from `main` (or using a release that includes this PR) gives full support. For older vLLM versions, out-of-tree registration or the bundled vLLM 0.8.5 scripts work as fallbacks.

## Compatibility Status

### vLLM Support Timeline

| Date | Event |
|------|-------|
| Jan 27, 2026 | DeepSeek-OCR-2 released with bundled vLLM 0.8.5 scripts |
| Feb 2, 2026 | [PR #33165](https://github.com/vllm-project/vllm/pull/33165) merged — native `DeepseekOCR2ForCausalLM` in vLLM |
| Feb 3, 2026 | [PR #33642](https://github.com/vllm-project/vllm/pull/33642) merged — BOS token fix for chat template |
| Open | [PR #33389](https://github.com/vllm-project/vllm/pull/33389) — accuracy fix for `transformers>=4.48` |

### Deployment Options

| Approach | vLLM Version | Effort | Status |
|----------|-------------|--------|--------|
| vLLM from `main` branch | latest | `pip install -e .` | Works now |
| vLLM release (>= v0.15.x) | v0.15.0+ | `pip install vllm` | Verify registry has `deepseek_ocr2` |
| Out-of-tree registration | Any 0.11+ | Copy 3 files + register | Works on any modern vLLM |
| Bundled vLLM 0.8.5 | 0.8.5 pinned | Use official repo scripts | Works but locks old version |

## Benchmark: v2 vs v1

| Metric | v1 | v2 | Delta |
|--------|-----|-----|-------|
| OmniDocBench v1.5 | 87.36% | 91.09% | +3.73% |
| Reading Order Edit Dist | 0.085 | 0.057 | -33% |
| Formula Edit Dist | 0.236 | 0.198 | -16% |
| Table Edit Dist | 0.123 | 0.096 | -22% |
| Text Edit Dist | 0.073 | 0.048 | -34% |
| Online repetition rate | 6.25% | 4.17% | -33% |

## References

### Official
- [DeepSeek-OCR-2 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR-2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR-2 Paper (arXiv:2601.20552)](https://arxiv.org/abs/2601.20552)

### vLLM
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [vLLM DeepSeek-OCR Recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- [PR #33165 — Add DeepSeek-OCR-2 Support](https://github.com/vllm-project/vllm/pull/33165)
- [Custom Logits Processors Docs](https://docs.vllm.ai/en/stable/features/custom_logitsprocs/)
- [Model Registration Docs](https://docs.vllm.ai/en/stable/contributing/model/registration/)

### Community
- [vLLM Forum: Running DeepSeek-OCR-2](https://discuss.vllm.ai/t/how-to-run-deep-seek-ocr-2-in-vllm/2280)
- [HuggingFace Discussion: Architecture Not Supported](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/discussions/3)
- [DeepSeek-OCR Issue #231: vLLM 0.11 Compatibility](https://github.com/deepseek-ai/DeepSeek-OCR/issues/231)
