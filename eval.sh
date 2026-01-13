#!/bin/bash

# --- 路径配置 ---
MODEL_PATH="/home/lijiaji/Qwen/Qwen3-VL-8B-Thinking"
CKPT_PATH="/home/lijiaji/ms-swift/output/mimic-lora-optimized/v30-20260107-160108/checkpoint-21000"

# 转换后的评估数据集文件名
EVAL_FILE="eval_qa_dataset.jsonl"

# --- 步骤 1: 转换数据集格式 ---
echo "--- Step 1: Preparing dataset for swift eval ---"
python prepare_eval_data.py
# 检查上一步是否成功
if [ $? -ne 0 ]; then
    echo "Error: Data preparation failed. Aborting."
    exit 1
fi
echo "Dataset preparation successful."


# --- 步骤 2: 运行 swift eval (官方文档推荐方法) ---
echo -e "\n--- Step 2: Starting evaluation with swift eval ---"
# 我们使用训练时所在的 QwenGen 环境
# 注意：你需要确保这个环境里安装了 ms-swift[eval]
# conda activate QwenGen

swift eval \
  --model "${MODEL_PATH}" \
  --ckpt_dir "${CKPT_PATH}" \
  --eval_dataset general_qa \
  --eval_dataset_args "path=${EVAL_FILE}" \
  --max_length 2048

echo -e "\n--- Evaluation finished ---"
echo "Results are saved in the checkpoint directory under 'eval_result.json'."
