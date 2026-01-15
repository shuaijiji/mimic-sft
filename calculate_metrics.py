# calculate_metrics.py
import json
import argparse
import evaluate
from tqdm import tqdm

# 加载评估指标
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

def calculate_metrics(original_dataset_path, prediction_path, lang='en'):
    """
    Calculates metrics by comparing ground truth from the original dataset
    with predictions from the swift infer output.
    """
    # 1. 读取真实标签
    ground_truths = []
    with open(original_dataset_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                label = item['messages'][1]['content']
                ground_truths.append(label.strip())
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    
    # 2. 读取模型预测
    predictions = []
    with open(prediction_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                # 假设 'response' 字段包含了模型的输出
                prediction = item['response']
                predictions.append(prediction.strip())
            except (json.JSONDecodeError, KeyError):
                continue

    if len(ground_truths) != len(predictions):
        print(f"Warning: Number of ground truths ({len(ground_truths)}) does not match number of predictions ({len(predictions)}).")
        # 截取较短的长度以进行比较
        min_len = min(len(ground_truths), len(predictions))
        ground_truths = ground_truths[:min_len]
        predictions = predictions[:min_len]

    print(f"Calculating metrics for {len(predictions)} samples...")

    # 3. 计算总体指标
    # 注意：HuggingFace evaluate 的 rouge 需要 list of lists for references
    references_for_rouge = [[gt] for gt in ground_truths]

    overall_bleu = bleu.compute(predictions=predictions, references=ground_truths)
    overall_rouge = rouge.compute(predictions=predictions, references=references_for_rouge)
    overall_meteor = meteor.compute(predictions=predictions, references=ground_truths)
    overall_bert = bertscore.compute(predictions=predictions, references=ground_truths, lang=lang)
    
    avg_bert_f1 = sum(overall_bert['f1']) / len(overall_bert['f1'])

    print("\\n--- Overall Evaluation Metrics ---")
    print(f"Overall BLEU: {overall_bleu['bleu']:.4f}")
    print(f"Overall ROUGE-1: {overall_rouge['rouge1']:.4f}")
    print(f"Overall ROUGE-2: {overall_rouge['rouge2']:.4f}")
    print(f"Overall ROUGE-L: {overall_rouge['rougeL']:.4f}")
    print(f"Overall METEOR: {overall_meteor['meteor']:.4f}")
    print(f"Overall BERTScore F1: {avg_bert_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics from swift infer output.")
    parser.add_argument('--original_dataset', type=str, required=True, help="Path to the original .jsonl dataset with ground truths.")
    parser.add_argument('--prediction_file', type=str, required=True, help="Path to the .jsonl file with model predictions from swift infer.")
    parser.add_argument('--lang', type=str, default='en', help="Language for BERTScore.")
    args = parser.parse_args()

    calculate_metrics(args.original_dataset, args.prediction_file, args.lang)
