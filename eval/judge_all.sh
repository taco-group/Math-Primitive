#!/usr/bin/env bash
# Judge and score all Prim outputs of ONE model. Calls the OpenAI API (costs money):
# requires OPENAI_API_KEY. Run after eval/run_all.sh.
#   bash eval/judge_all.sh results/<served-name>
set -euo pipefail
D=$1
HERE=$(cd "$(dirname "$0")" && pwd)
[ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY is not set"; exit 1; }
for t in generation execution; do
  if [ -f "$D/$t.jsonl" ]; then python "$HERE/judge_answer.py" --input "$D/$t.jsonl"; fi
done
for t in discovery digestion; do
  if [ -f "$D/$t.jsonl" ]; then python "$HERE/judge_primitive.py" --input "$D/$t.jsonl"; fi
done
python "$HERE/score.py" --results "$D"
