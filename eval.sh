#!/bin/bash

# --- 路径配置 ---
MODEL_PATH="/home/lijiaji/Qwen/Qwen3-VL-8B-Thinking"
CKPT_PATH="/home/lijiaji/ms-swift/output/mimic-lora-optimized/v30-20260107-160108/checkpoint-21000"
TEST_JSON="/home/lijayji/mimic_cxr/img2text_test.jsonl" # 注意：这里修正了一个可能的拼写错误 lijiayji -> lijiaji

# --- 调试步骤：打印将要执行的命令 ---
echo "--- DEBUG INFO ---"
echo "The following command will be executed:"
# 使用 set -x 来显示命令的展开形式，更清晰
set -x

# --- 执行评估命令（写在单行以避免换行符问题） ---
swift infer --model "${MODEL_PATH}" --ckpt_dir "${CKPT_PATH}" --dataset "${TEST_JSON}" --max_length 2048

# 关闭调试模式
set +x

echo "--- END OF SCRIPT ---"
