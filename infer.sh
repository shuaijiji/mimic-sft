#!/bin/bash

# --- 配置模型路径 ---
# 原始的、未经微调的基础模型路径
MODEL_PATH="/home/lijiaji/Qwen/Qwen3-VL-8B-Thinking"

# 训练完成后，效果最佳的LoRA权重路径
# 这是swift框架根据验证集表现自动为你挑选的
CKPT_PATH="/home/lijiaji/ms-swift/output/mimic-lora-optimized/v30-20260107-160108/checkpoint-21000"

# --- 启动推理命令 ---
swift infer \
  --model ${MODEL_PATH} \
  --ckpt_dir ${CKPT_PATH} \
  --stream true \

