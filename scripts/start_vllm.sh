#!/usr/bin/env bash
#
# Start vLLM on H100 with baseline configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Flag rationale:
#   --max-model-len 16384          : prompts are 1.5-3K tokens + short SQL output, 8K is safe headroom
#   --max-num-seqs 32             : baseline concurrency; tune up if queue depth grows under load
#   --gpu-memory-utilization 0.90 : leave 10% for CUDA kernels and torch overhead
#   --enable-prefix-caching       : schema prefix is identical across calls to same DB, cache hits expected
#   --disable-log-requests        : reduces CPU overhead under load

set -euo pipefail

# Load environment variables (HF_TOKEN, VLLM_MODEL, etc.)
set -a
source "$(dirname "$0")/../.env"
set +a

MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --disable-log-requests