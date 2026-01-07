#!/bin/bash

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
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
MODEL_PATH="/home/lijiaji/Qwen/Qwen3-VL-8B-Thinking"
TRAIN_JSON="/home/lijiaji/mimic_cxr/img2text_train.jsonl"
VAL_JSON="/home/lijiaji/mimic_cxr/img2text_val.jsonl"
OUTPUT_DIR="/home/lijiaji/ms-swift/output/mimic-lora-optimized"

# --- GPU 配置 ---
# 请在此处指定要使用的GPU。对于4张卡, 设置为 "0,1,2,3"。
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 自动计算指定的GPU数量
# 通过计算逗号数量再加1来实现
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
nproc_per_node=$((nproc_per_node + 1))

echo "指定的 GPUs: $CUDA_VISIBLE_DEVICES"
echo "GPU 数量 (nproc_per_node): $nproc_per_node"
export NPROC_PER_NODE=$nproc_per_node
# --- GPU 配置结束 ---

mkdir -p ${OUTPUT_DIR}



# 计算梯度累积步数
# 假设全局批次大小为 16
GLOBAL_BATCH_SIZE=16
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * nproc_per_node)))

# 启动训练命令 (基于 Qwen3-VL 官方最佳实践重构)
swift sft \
  --model ${MODEL_PATH} \
  --custom_dataset_info custom_dataset_info.json \
  --dataset mimic_cxr_train mimic_cxr_val \
  --train_type lora \  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --attn_impl flash_attn \
  --padding_free true \
  --learning_rate 1e-4 \
  --lora_rank 32 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --logging_steps 10 \
  --max_length 2048 \
  --output_dir ${OUTPUT_DIR} \
  --warmup_ratio 0.05 \
  --deepspeed zero2 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --ddp_timeout 18000000 \
  --ignore_args_error True

