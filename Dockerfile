FROM vllm/vllm-openai:latest

# Pin transformers to avoid accuracy bug (PR #33389)
RUN pip install transformers==4.46.3

# Optional: pre-download model at build time
# Uncomment to bake the model into the image (larger image, faster startup)
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-OCR-2')"

EXPOSE 8000

ENTRYPOINT ["vllm", "serve", "deepseek-ai/DeepSeek-OCR-2", \
    "--trust-remote-code", \
    "--dtype", "bfloat16", \
    "--max-model-len", "8192", \
    "--gpu-memory-utilization", "0.75", \
    "--no-enable-prefix-caching", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--logits_processors", "vllm.model_executor.models.deepseek_ocr2:NGramPerReqLogitsProcessor"]
