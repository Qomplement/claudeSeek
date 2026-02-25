#!/bin/bash
# Start vLLM server for DeepSeek-OCR-2
# Usage: bash scripts/start_server.sh [--port 8000] [--tp 1]

set -e

PORT="${PORT:-8000}"
TP="${TP:-1}"
GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --gpu-util) GPU_UTIL="$2"; shift 2 ;;
        --max-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Starting DeepSeek-OCR-2 vLLM Server ==="
echo "Port: $PORT"
echo "Tensor Parallel: $TP"
echo "GPU Utilization: $GPU_UTIL"
echo "Max Model Length: $MAX_MODEL_LEN"

vllm serve deepseek-ai/DeepSeek-OCR-2 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --no-enable-prefix-caching \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --logits_processors vllm.model_executor.models.deepseek_ocr2:NGramPerReqLogitsProcessor
