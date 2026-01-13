#!/bin/bash

# --- 路径配置 ---
MODEL_PATH="/home/lijiaji/Qwen/Qwen3-VL-8B-Thinking"
CKPT_PATH="/home/lijiaji/ms-swift/output/mimic-lora-optimized/v30-20260107-160108/checkpoint-21000"
TEST_JSON="/home/lijiaji/mimic_cxr/img2text_test.jsonl"

# --- 执行评估命令 ---
# 我们明确添加 --eval_human false 来强制关闭交互模式，进入批量处理模式
echo "Starting batch inference for evaluation..."

swift infer \
  --model "${MODEL_PATH}" \
  --ckpt_dir "${CKPT_PATH}" \
  --dataset "${TEST_JSON}" \
  --max_length 2048 \
  --eval_human false

echo "Batch inference finished."
echo "Now you can run 'python calculate_metrics.py' to get the scores."
