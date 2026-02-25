---
id: architecture
title: Architecture Deep Dive
sidebar_position: 2
---

# DeepSeek-OCR-2 Architecture

## High-Level Model Composition

```
DeepseekOCR2ForCausalLM  (3B params total, ~500M active/token)
|
+-- Vision Pipeline
|   +-- ImageEncoderViT (SAM ViT-B)           -- 80M params
|   |     12 ViT blocks, 768-dim, window+global attention
|   |     Multi-scale neck -> 896-dim output
|   |
|   +-- Qwen2Decoder2Encoder (DeepEncoder V2) -- 500M params
|   |     Qwen2-0.5B repurposed as encoder
|   |     24 decoder layers, 896 hidden, 14 attn heads, 2 KV heads
|   |     Hybrid attention mask (bidirectional images + causal queries)
|   |     Learnable query tokens (144 for 768px, 256 for 1024px)
|   |
|   +-- MlpProjector (Linear)
|         896 -> 1280 dims
|
+-- Token Merging
|     Replaces image placeholder embeddings with visual embeddings
|     view_separator token between global and local crop features
|
+-- DeepSeek-V2 3B MoE LLM
      12 layers, hidden_size=1280
      64 routed experts, 2 shared experts, 6 active/token
      MoE intermediate size: 896
      Vocab: 129,280 | Max positions: 8192
```

## Data Flow

```
Input Image (B, 3, 1024, 1024)
    |
    v
[SAM ViT-B] Patch embed -> 12 ViT blocks -> Neck -> Downsample
    | Output: (B, 896, 16, 16) = 256 tokens
    v
[Qwen2Decoder2Encoder] Concat with 256 query tokens -> Hybrid attention -> Extract queries
    | Output: (B, 256, 896)
    v
[Linear Projector] 896 -> 1280
    | Output: (B, 256, 1280)
    v
[Token Merging] Replace <image> token embeddings
    v
[DeepSeek-V2 3B MoE LLM] 12 layers, 64 experts
    v
Output tokens (markdown, OCR text, etc.)
```

## Visual Causal Flow (Key Innovation)

### Problem with v1

v1 flattened image tokens in a fixed top-left to bottom-right raster scan. This doesn't match how humans read documents with columns, tables, and complex layouts.

### Solution: Learned Causal Reordering

v2 uses learnable query tokens that **attend to all image tokens** but only to **previous query tokens** (causal). This lets the model learn the optimal reading order during training.

### Hybrid Attention Mask

```
                  Image Tokens (m)    Query Tokens (n)
                +------------------+------------------+
Image Tokens    |    1 (full)      |    0 (blocked)   |
   (m rows)     |  m x m ones      |  m x n zeros     |
                +------------------+------------------+
Query Tokens    |    1 (full)      |  Lower Triangle  |
   (n rows)     |  n x m ones      |  n x n causal    |
                +------------------+------------------+

- Image tokens see ALL image tokens (bidirectional, like a ViT)
- Query tokens see ALL image tokens (global visual context)
- Query tokens see only PREVIOUS query tokens (causal/autoregressive)
```

## Multi-Crop Dynamic Resolution

| View | Resolution | Visual Tokens |
|------|-----------|---------------|
| Global | 1024x1024 | 256 |
| Local crop (each) | 768x768 | 144 |
| Local crops (0-6) | -- | 0-864 |
| **Total range** | -- | **256-1120** |

A learned `view_separator` token is inserted between global and local features.

## v1 vs v2 Comparison

| Feature | DeepSeek-OCR (v1) | DeepSeek-OCR-2 (v2) |
|---------|-------------------|---------------------|
| Architecture class | `DeepseekOCRForCausalLM` | `DeepseekOCR2ForCausalLM` |
| Vision encoder | CLIP-L-14 + SAM ViT-B | SAM ViT-B + Qwen2-0.5B |
| Projector input dim | 2048 (CLIP+SAM concat) | 896 (Qwen2 output) |
| Reading order | Fixed raster scan | Learned causal reordering |
| Max visual tokens | 1156 | 1120 |

## Language Model (DeepSeek-V2 3B MoE)

Identical between v1 and v2:

```json
{
  "hidden_size": 1280,
  "num_hidden_layers": 12,
  "num_attention_heads": 10,
  "num_key_value_heads": 10,
  "n_routed_experts": 64,
  "n_shared_experts": 2,
  "num_experts_per_tok": 6,
  "moe_intermediate_size": 896,
  "use_mla": false,
  "max_position_embeddings": 8192,
  "vocab_size": 129280
}
```

## NGram Logits Processor

OCR models are prone to repetitive outputs. The `NoRepeatNGramLogitsProcessor` prevents this:

- `ngram_size: 30` — check for repeated 30-token sequences
- `window_size: 90` — within a sliding window of 90 tokens
- `whitelist: {128821, 128822}` — exempt table structure tokens

## HuggingFace Custom Code

Requires `trust_remote_code=True`:

| File | Size | Contents |
|------|------|----------|
| `modeling_deepseekocr2.py` | 39.2 kB | Model, config, preprocessing, inference |
| `deepencoderv2.py` | 36.3 kB | SAM ViT-B, Qwen2 encoder, hybrid attention |
| `modeling_deepseekv2.py` | 82.2 kB | Base MoE language model |
| `configuration_deepseek_v2.py` | 10.6 kB | Config class |
| `conversation.py` | 9.25 kB | Chat template |
