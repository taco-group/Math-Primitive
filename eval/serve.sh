#!/usr/bin/env bash
# Serve a model with vLLM for the Prim inference scripts.
#   bash eval/serve.sh <model path or HF id> <served-model-name> <max-model-len> [port]
# Env: CUDA_VISIBLE_DEVICES, TP (tensor parallel size, default 1), GPU_UTIL (default 0.9)
#
# Context lengths used in the paper:
#   131072  generation.py, execution.py
#    40960  discovery.py, digestion.py
# NEVER add --reasoning-parser: the scripts read the full thinking trace + answer from
# message.content, and a reasoning parser silently moves the answer out of it.
set -euo pipefail
MODEL=$1; NAME=$2; MLEN=$3; PORT=${4:-8000}
exec vllm serve "$MODEL" --served-model-name "$NAME" --port "$PORT" \
    --tensor-parallel-size "${TP:-1}" --max-model-len "$MLEN" \
    --gpu-memory-utilization "${GPU_UTIL:-0.9}"
