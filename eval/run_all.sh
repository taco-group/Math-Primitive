#!/usr/bin/env bash
# Run inference for all four Prim dimensions for ONE model. Score with eval/judge_all.sh.
#   bash eval/run_all.sh <model path or HF id> <served-model-name>
# Env: CUDA_VISIBLE_DEVICES, TP (default 1), PORT (default 8000), OUT_DIR (default results)
#
#   @131072: generation (answer, no primitive), execution (answer from the gold primitive)
#   @40960 : discovery (write the primitive from the problem), digestion (from problem + solution)
# Every step resumes, so re-running the script only fills in missing / failed rows.
set -euo pipefail
MODEL=$1; NAME=$2
PORT=${PORT:-8000}; URL=http://localhost:$PORT/v1
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT_DIR:-results}/$NAME; mkdir -p "$OUT/logs"
SERVER_PID=""

start_server () {  # $1 = max-model-len
  echo "[run_all] starting vLLM for $NAME @ $1"
  bash "$HERE/serve.sh" "$MODEL" "$NAME" "$1" "$PORT" > "$OUT/logs/vllm_$1.log" 2>&1 &
  SERVER_PID=$!
  until curl -sf "$URL/models" > /dev/null; do
    kill -0 "$SERVER_PID" 2> /dev/null || { echo "vLLM exited, see $OUT/logs/vllm_$1.log"; exit 1; }
    sleep 10
  done
}
stop_server () {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2> /dev/null && wait "$SERVER_PID" 2> /dev/null || true
  SERVER_PID=""
}
trap stop_server EXIT

start_server 131072
python "$HERE/generation.py" --model "$NAME" --base-url "$URL" --workers 20 \
    --out "$OUT/generation.jsonl" 2>&1 | tee "$OUT/logs/generation.log"
python "$HERE/execution.py" --model "$NAME" --base-url "$URL" --workers 8 \
    --out "$OUT/execution.jsonl" 2>&1 | tee "$OUT/logs/execution.log"
stop_server

start_server 40960
python "$HERE/discovery.py" --model "$NAME" --base-url "$URL" --workers 12 \
    --out "$OUT/discovery.jsonl" 2>&1 | tee "$OUT/logs/discovery.log"
python "$HERE/digestion.py" --model "$NAME" --base-url "$URL" --workers 12 \
    --out "$OUT/digestion.jsonl" 2>&1 | tee "$OUT/logs/digestion.log"
stop_server
echo "[run_all] done -> $OUT"
