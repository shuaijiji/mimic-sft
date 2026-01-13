#!/bin/bash

# --- 路径配置 ---
# 原始基础模型路径
MODEL_PATH="/home/lijiaji/Qwen/Qwen3-VL-8B-Thinking"

# 训练产出的最佳LoRA权重路径
CKPT_PATH="/home/lijiaji/ms-swift/output/mimic-lora-optimized/v30-20260107-160108/checkpoint-21000"

# !!! 重要 !!!
# 用于评估的测试集文件路径，这个文件不能与训练集或验证集有重合
TEST_JSON="/home/lijiaji/mimic_cxr/img2text_test.jsonl"


# --- 启动评估命令 ---
echo "Starting evaluation..."
echo "Model: ${MODEL_PATH}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Test Set: ${TEST_JSON}"

swift eval \
  --model "${MODEL_PATH}" \
  --ckpt_dir "${CKPT_PATH}" \
  --dataset "${TEST_JSON}" \

  --max_length 2048

echo "Evaluation finished. Results are saved in the checkpoint directory."
