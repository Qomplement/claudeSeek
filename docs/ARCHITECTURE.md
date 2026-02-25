# DeepSeek-OCR-2 Architecture Deep Dive

## 1. High-Level Model Composition

```
DeepseekOCR2ForCausalLM  (top-level, 3B params total, ~500M active/token)
│
├── Vision Pipeline
│   ├── ImageEncoderViT (SAM ViT-B)          — 80M params
│   │     12 ViT blocks, 768-dim, window+global attention
│   │     Multi-scale neck → 896-dim output
│   │
│   ├── Qwen2Decoder2Encoder (DeepEncoder V2) — 500M params
│   │     Qwen2-0.5B repurposed as encoder
│   │     24 decoder layers, 896 hidden, 14 attn heads, 2 KV heads
│   │     Hybrid attention mask (bidirectional images + causal queries)
│   │     Learnable query tokens (144 for 768px, 256 for 1024px)
│   │
│   └── MlpProjector (Linear)
│         896 → 1280 dims
│
├── Token Merging
│     Replaces <image> placeholder embeddings with visual embeddings
│     view_separator token between global and local crop features
│
└── DeepSeek-V2 3B MoE LLM
      12 layers, hidden_size=1280
      64 routed experts, 2 shared experts, 6 active/token
      MoE intermediate size: 896
      use_mla: false
      Vocab: 129,280
      Max position embeddings: 8192
```

## 2. SAM ViT-B (Image Tokenizer)

Unchanged from v1. Converts raw pixels into spatial feature maps.

```
Input:  (B, 3, 1024, 1024)  — RGB image

PatchEmbed:
  16×16 patches → (B, 64, 64, 768)

12 ViT Blocks:
  Window attention (window_size=14) at most layers
  Global attention at layers [2, 5, 8, 11]
  Output: (B, 64, 64, 768)

Neck (Multi-scale):
  net_1: Conv2d → (B, 256, 64, 64)
  net_2: Downsample → (B, 512, 32, 32)
  net_3: Downsample → (B, 896, 16, 16)

Output: (B, 896, 16, 16) = 256 spatial positions × 896 dims
```

## 3. Visual Causal Flow (The Key Innovation)

### Problem with v1 (Fixed Raster Scan)

v1 flattened image tokens in a fixed top-left → bottom-right order. This doesn't match how humans read documents (columns, tables, figures have complex reading orders).

### Solution: Learned Causal Reordering

v2 uses learnable query tokens that **attend to all image tokens** but only to **previous query tokens** (causal). This lets the model learn the optimal reading order during training.

### Qwen2Decoder2Encoder Implementation

```python
class Qwen2Decoder2Encoder(nn.Module):
    def __init__(self):
        self.model = CustomQwen2Decoder(...)      # 24-layer Qwen2-0.5B
        self.query_768 = nn.Embedding(144, 896)   # queries for 768×768 crops
        self.query_1024 = nn.Embedding(256, 896)  # queries for 1024×1024 global

    def forward(self, x):
        # x: (B, 256, 896) from SAM ViT-B (flattened spatial features)

        # Select queries based on resolution
        queries = self.query_1024.weight  # (256, 896) for global view

        # Concatenate: [image_tokens | query_tokens]
        x_combined = cat([x, queries], dim=1)  # (B, 512, 896)

        # Build token_type_ids
        token_type_ids = cat([
            zeros(B, 256),  # 0 = image tokens (bidirectional)
            ones(B, 256),   # 1 = query tokens (causal)
        ])

        # Run through Qwen2 with hybrid mask
        y = self.model(x_combined, token_type_ids)

        # Return ONLY query outputs (discard image outputs)
        return y[:, 256:, :]  # (B, 256, 896)
```

### Hybrid Attention Mask

```
                  Image Tokens (m)    Query Tokens (n)
                ┌─────────────────┬─────────────────┐
Image Tokens    │    1 (full)     │    0 (blocked)  │
   (m rows)     │  m×m ones       │  m×n zeros      │
                ├─────────────────┼─────────────────┤
Query Tokens    │    1 (full)     │  Lower Triangle │
   (n rows)     │  n×m ones       │  n×n causal     │
                └─────────────────┴─────────────────┘

- Image tokens see ALL image tokens (bidirectional, like a ViT)
- Query tokens see ALL image tokens (global visual context)
- Query tokens see only PREVIOUS query tokens (causal/autoregressive)
```

This is the "Visual Causal Flow" — the queries learn a 1D causal ordering of the 2D visual information.

## 4. Multi-Crop Dynamic Resolution

```
Input document image (arbitrary aspect ratio)
    │
    ▼
┌────────────────────────────────┐
│  dynamic_preprocess()          │
│  Decides optimal tiling:       │
│  - 1 global view (1024×1024)   │
│  - 0-6 local crops (768×768)   │
│  Based on aspect ratio & size  │
└────────────────────────────────┘
    │
    ▼
Global view → SAM → Qwen2 → 256 tokens
Local crop 1 → SAM → Qwen2 → 144 tokens
Local crop 2 → SAM → Qwen2 → 144 tokens
...
Local crop N → SAM → Qwen2 → 144 tokens
    │
    ▼
[global_tokens] [view_sep] [crop1_tokens] [view_sep] [crop2_tokens] ...
    │
    ▼
Linear projection (896 → 1280) for all tokens
    │
    ▼
Merged into text sequence at <image> positions
```

## 5. Language Model (DeepSeek-V2 3B MoE)

Identical between v1 and v2. Key specs:

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
  "vocab_size": 129280,
  "rope_theta": 10000.0
}
```

- **MoE routing**: Top-6 out of 64 routed experts + 2 shared (always active)
- **No MLA**: Standard grouped-query attention (10 heads, 10 KV heads = MHA)
- **~500M active params** per token despite 3B total

## 6. NGram Logits Processor (Why It's Required)

OCR models are prone to repetitive outputs (e.g., repeating table rows). The `NoRepeatNGramLogitsProcessor` prevents this:

```
Parameters:
  ngram_size: 30    — check for repeated 30-token sequences
  window_size: 90   — within a sliding window of 90 tokens
  whitelist: {128821, 128822}  — exempt <td>/<td> (table structure tokens)

Algorithm:
  For each candidate token:
    - Look at the last (ngram_size - 1) tokens
    - Check if this n-gram appeared before within window_size
    - If yes, set logit to -inf (ban the token)
    - Exception: whitelisted tokens are never banned
```

Without this processor, quality degrades significantly on structured documents (tables, lists, forms).

## 7. Weight Mapping (HuggingFace → vLLM)

vLLM uses a `WeightsMapper` to remap HuggingFace checkpoint keys:

```python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "model.embed_tokens.": "language_model.model.embed_tokens.",
        "model.layers.": "language_model.model.layers.",
        "model.norm.": "language_model.model.norm.",
        "lm_head.": "language_model.lm_head.",
        "model.": "",  # vision components keep original names
    }
)
```

Vision encoder weights (`sam_model.*`, `vision_model.*`, `projector.*`) keep their original prefixes. Only the language model weights get remapped under `language_model.*`.
