#!/bin/bash
set -euo pipefail

NODES=("g1" "g2")
NODE_NUMS=${#NODES[@]}
NUM_PROCESSES=16
MASTER_ADDR=${NODES[0]}
MASTER_PORT=29500
RUN_NAME="test"
FILE_DIR=$(cd "$(dirname "$0")" && pwd)
# ---------------------

PDSH_PIDS=()

cleanup() {
  echo -e "\n>>> Cleanup: killing local pdsh and remote training processes..."
  if [ ${#PDSH_PIDS[@]} -gt 0 ]; then
    kill "${PDSH_PIDS[@]}" 2>/dev/null || true
  fi
  pdsh -R ssh -w "${NODES[*]}" "pkill -f 'accelerate launch' || true"
}
trap cleanup EXIT SIGINT SIGTERM

echo "Training directory: $FILE_DIR"
echo "Launching on nodes: ${NODES[*]}"

for i in "${!NODES[@]}"; do
  NODE=${NODES[$i]}
  NODE_RANK=$i

  echo "-> Launching on $NODE (rank $NODE_RANK)..."

  pdsh -R ssh -w "$NODE" bash -lc "
    source ~/miniconda3/bin/activate arl
    cd '$FILE_DIR'
    export TOKENIZERS_PARALLELISM=false
    export WANDB_PROJECT=VLM-RFT
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export NODE_RANK=$NODE_RANK
    export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
    export MONITOR_INTERVAL=30

    accelerate launch \
      --config_file config_files/fsdp2_dst.yml \
      --num_machines=$NODE_NUMS \
      --num_processes=$NUM_PROCESSES \
      --machine_rank=$i \
      --main_process_ip=$MASTER_ADDR \
      --main_process_port=$MASTER_PORT \
      --same_network \
      grpo.py \
      --output_dir output/$RUN_NAME \
      --clear_device true \
      --model_name_or_path output/resume-sft \
      --dataset_name /share_data/data1/GUIData/train_ody_aitwc_mb_ac_locate.jsonl \
      --eval_dataset_name /share_data/data1/GUIData/valid_f_ody_aitwc_mb_ac.jsonl \
      --max_prompt_length 2048 \
      --max_completion_length 128 \
      --max_line_res 1120 \
      --hist_length 1 \
      --num_generations 32 \
      --num_iterations 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 32 \
      --dataloader_prefetch_factor 4 \
      --dataloader_num_workers 4 \
      --dataloader_drop_last true \
      --max_grad_norm 1.0 \
      --logging_steps 1 \
      --learning_rate 1e-6 \
      --warmup_steps 10 \
      --weight_decay 0.1 \
      --eval_strategy steps \
      --per_device_eval_batch_size 4 \
      --eval_steps 25 \
      --adam_beta2 0.99 \
      --lr_scheduler_type 'constant' \
      --tune_vision true \
      --bf16 \
      --beta 0.0 \
      --data_seed 41 \
      --report_to wandb \
      --num_train_epochs 3 \
      --run_name $RUN_NAME \
      --save_steps 50 \
      --save_only_model true \
      --attn_implementation flash_attention_2 \
      --reward_funcs 'type' 'args'
  " &

  PDSH_PIDS+=($!)
done

wait
