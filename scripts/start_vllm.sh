#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

# MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000
    --device cpu
