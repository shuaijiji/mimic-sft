#!/bin/bash

# --- 路径配置 ---
MODEL_PATH="/home/lijiaji/Qwen/Qwen3-VL-8B-Thinking"
CKPT_PATH="/home/lijiaji/ms-swift/output/mimic-lora-optimized/v30-20260107-160108/checkpoint-21000"
TEST_JSON="/home/lijiaji/mimic_cxr/img2text_test.jsonl"

# 定义预测结果的输出路径
# swift infer 会自动在这个目录下创建一个带时间戳的 .jsonl 文件
RESULT_DIR="${CKPT_PATH}/test_predictions"
mkdir -p ${RESULT_DIR}

# --- 第一步：使用 swift infer 生成预测结果 ---
echo "Step 1: Generating predictions for the test set..."
echo "Results will be saved in: ${RESULT_DIR}"

# 注意：我们使用 swift infer，它会遍历数据集并保存输出
swift infer \
  --model "${MODEL_PATH}" \
  --ckpt_dir "${CKPT_PATH}" \
  --dataset "${TEST_JSON}" \
  --result_dir "${RESULT_DIR}" \
  --max_length 2048 \
  --eval_batch_size 4

echo "Prediction generation finished."
