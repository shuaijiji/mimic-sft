#!/bin/bash

# --- 路径配置 ---
MODEL_PATH="/home/lijiaji/Qwen/Qwen3-VL-8B-Thinking"
CKPT_PATH="/home/lijiaji/ms-swift/output/mimic-lora-optimized/v30-20260107-160108/checkpoint-21000"
TEST_JSON="/home/lijiaji/mimic_cxr/img2text_test.jsonl"

# swift infer 会自动在 ckpt_dir/infer_result/ 目录下创建结果文件
# 我们不再需要手动指定输出目录
swift infer \
  --model "${MODEL_PATH}" \
  --ckpt_dir "${CKPT_PATH}" \
  --dataset "${TEST_JSON}" \
  --max_length 2048

echo "Prediction generation finished."
