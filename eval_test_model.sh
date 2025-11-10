# 模型路径和测试集配置
MODEL_PATH="/home/lijiaji/ms-swift/output/mimic-lora/v18-20250708-165849/checkpoint-22000-merged"
TEST_JSON_DIR="/home/lijiaji/mimic_cxr"         # 包含 evalscope_test.jsonl 的文件夹
SUBSET_NAME="evalscope_test"                    # 对应你的文件名：evalscope_test.jsonl

# 执行评测命令
CUDA_VISIBLE_DEVICES=0,1 \
swift eval \
    --model ${MODEL_PATH} \
    --eval_backend Native \
    --infer_backend pt \
    --model_type qwen2_5_vl \
    --eval_dataset general_qa \
    --dataset_args "{
        \"general_qa\": {
            \"local_path\": \"${TEST_JSON_DIR}\",
            \"subset_list\": [\"${SUBSET_NAME}\"],
            \"prompt_template\": \"你是一名专业的放射科医生。现在请根据以下胸部X光图像生成一份中文医学影像诊断报告，要求客观、准确、用语规范，包含如下部分：\\n1. 图像质量描述（如图像是否清晰、是否存在伪影）；\\n2. 肺部观察结果（是否有结节、实变、肺气肿、间质性病变等）；\\n3. 心影和纵隔情况（心影大小、纵隔宽度等）；\\n4. 胸膜或其他异常（如积液、气胸、骨骼异常等）；\\n5. 结论和建议（是否发现异常、是否建议进一步检查）。\\n请用专业术语生成完整的报告。\\n\\n问题：{query}\\n答案：\",
            \"metric_list\": [\"Rouge-1-F\", \"Rouge-L-F\", \"BLEU-4\",\"BERTScore\"],
            \"generation_config\": {
                \"max_tokens\": 200,
                \"temperature\": 0.7,
                \"top_p\": 0.9,
                \"repetition_penalty\": 1.2
            }
        }
    }" \
    --seed 42 
