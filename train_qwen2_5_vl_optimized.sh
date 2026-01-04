#!/bin/bash

# 设置环境变量
export IMAGE_FACTOR=14
export MIN_PIXELS=1024
export MAX_PIXELS=204800
export MAX_RATIO=200
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DIST_FORK_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 数据路径配置
MODEL_PATH="/home/lijiaji/Qwen2.5-VL-7B-Instruct"
TRAIN_JSON="/home/lijiaji/mimic_cxr/ms_swift_train.jsonl"
VAL_JSON="/home/lijiaji/mimic_cxr/ms_swift_val.jsonl"
OUTPUT_DIR="/home/lijiaji/ms-swift/output/mimic-lora-optimized"

# 自动检测 GPU 数量
nproc_per_node=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Detected GPUs: $nproc_per_node"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=$nproc_per_node

mkdir -p ${OUTPUT_DIR}

# 创建 DeepSpeed 配置文件（ZeRO-2, 优化版）
cat <<EOL > deepspeed_config_optimized.json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "gradient_accumulation_steps": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false,
  "steps_per_print": 10
}
EOL

# 计算梯度累积步数
# 假设全局批次大小为 16
GLOBAL_BATCH_SIZE=16
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * nproc_per_node)))

# 启动训练命令
swift sft \
  --model ${MODEL_PATH} \
  --dataset ${TRAIN_JSON} \
  --val_dataset ${VAL_JSON} \
  --train_type lora \
  --columns '{"report": "text", "image_path": "images"}' \
  --num_train_epochs 3 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --learning_rate 1e-4 \
  --output_dir ${OUTPUT_DIR} \
  --logging_dir ${OUTPUT_DIR}/logs \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --save_strategy steps \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --report_to tensorboard \
  --deepspeed deepspeed_config_optimized.json \
  --bf16 True \
  --dataloader_num_workers 16 \
  --ddp_timeout 18000000 \
  --remove_unused_columns True \
  --run_name qwen2_5_vl_lora_mimic_optimized \
  --use_hf False \
  --train_dataloader_shuffle True \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --lora_rank 32 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --early_stopping_patience 3 \
  --ignore_args_error True \
  --model_kwargs '{"device_map": null}' \
  --disable_image_saving \
  --max_length 2048
