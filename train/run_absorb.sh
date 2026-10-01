#!/usr/bin/env bash
# Train Absorb on math-phd-qual-709. Run from INSIDE your OPSD checkout after copying
# this directory's files into it (see README, "Training").
#
#   SIZE=9B NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_absorb.sh
#
# SIZE : 4B | 9B | 27B     (Qwen3.5 backbone, pinned to the revision we trained on)
# NGPU : number of GPUs. Gradient accumulation is set to 32/NGPU so the effective batch is 32.
# OUT  : output root (default ./runs)
#
# Trains for one epoch over the 709 problems; the adapter is saved to $OUT/absorb_qwen3.5-<size>.
# Memory: the trainer colocates a vLLM engine on each GPU that takes VLLM_UTIL (default 0.43) of
# its memory. If you run out of memory, lower it, e.g. VLLM_UTIL=0.35.
set -euo pipefail
SIZE=${SIZE:-9B}
case "$SIZE" in
  4B)  REPO=Qwen/Qwen3.5-4B;  REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a ;;
  9B)  REPO=Qwen/Qwen3.5-9B;  REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a ;;
  27B) REPO=Qwen/Qwen3.5-27B; REV=fc05daec18b0a78c049392ed2e771dde82bdf654 ;;
  *) echo "SIZE must be 4B, 9B or 27B"; exit 1 ;;
esac
NGPU=${NGPU:-1}
if (( 32 % NGPU != 0 )); then echo "NGPU must divide 32"; exit 1; fi
ACCUM=$((32 / NGPU))
OUT=${OUT:-runs}
RUN=absorb_qwen3.5-${SIZE,,}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-offline}

SNAP=$(python -c "from huggingface_hub import snapshot_download as d; print(d('$REPO', revision='$REV'))")
echo "[absorb] $REPO@$REV -> $SNAP | NGPU=$NGPU ACCUM=$ACCUM (effective batch 32) | run=$OUT/$RUN"

accelerate launch \
    --config_file accelerate_opsd.yaml \
    --num_processes "$NGPU" \
    --gradient_accumulation_steps "$ACCUM" \
    --main_process_port "${PORT:-12953}" \
    opsd_train.py \
    --model_name_or_path "$SNAP" \
    --train_dataset shuoxing/math-phd-qual-709 \
    --privilege_style primitive \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$ACCUM" \
    --output_dir "$OUT" \
    --run_config "$RUN" \
    --num_train_epochs 1 \
    --max_completion_length 4096 \
    --top_k_loss 128 \
    --logging_steps 1 \
    --attn_implementation sdpa \
    --dtype bfloat16 \
    --max_length 6144 \
    --beta 1 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_UTIL:-0.43}" \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in_proj_qkv in_proj_a in_proj_b in_proj_z out_proj \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.06 \
    --teacher_thinking False \
    --wandb_project absorb \
    "$@"

echo "[absorb] done. Merge the adapter for serving:"
echo "  python merge_lora.py $SNAP $OUT/$RUN merged/$RUN"
