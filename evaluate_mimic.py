import os
import csv
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
from PIL import Image
import evaluate

# --- 加载评估指标 ---
# 确保你已经安装了 evaluate, bert_score, rouge_score, nltk, bleu
# pip install evaluate bert_score rouge_score nltk bleu
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Qwen-VL model for medical report generation.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the merged model with LoRA weights.")
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the tokenizer.")
    parser.add_argument('--test_data_path', type=str, required=True, help="Path to the test data in .jsonl format.")
    parser.add_argument('--output_dir', type=str, default="./eval_output", help="Directory to save the evaluation results.")
    parser.add_argument('--max_new_tokens', type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run the model on.")
    parser.add_argument('--image_root_path', type=str, default="", help="Root path for the image files if paths in jsonl are relative.")
    return parser.parse_args()


class MimicDataset(Dataset):
    """Dataset for loading MIMIC-CXR data in the new jsonl format."""
    def __init__(self, jsonl_path, image_root_path=''):
        self.data = []
        self.image_root_path = image_root_path
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 提取图片路径
        # 假设 'images' 列表里只有一个图片
        if not item.get("images"):
            raise ValueError(f"Missing 'images' key in data at index {idx}")
        relative_image_path = item["images"][0]
        # 如果jsonl中的路径不是绝对路径，需要拼接一个根路径
        image_path = os.path.join(self.image_root_path, relative_image_path)

        # 提取指令和真实报告
        messages = item.get("messages", [])
        if len(messages) < 2:
            raise ValueError(f"Invalid 'messages' format at index {idx}")
            
        instruction_content = messages[0].get("content", "")
        # 移除 <image>\n 标记
        instruction = instruction_content.replace("<image>\\n", "").strip()
        
        ground_truth = messages[1].get("content", "")

        return {
            "image_path": image_path,
            "instruction": instruction,
            "ground_truth": ground_truth
        }


def collate_fn(batch):
    image_paths = [item['image_path'] for item in batch]
    instructions = [item['instruction'] for item in batch]
    ground_truths = [item['ground_truth'] for item in batch]
    return {
        "image_paths": image_paths,
        "instructions": instructions,
        "ground_truths": ground_truths
    }


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def main():
    print("--- Starting Evaluation Script ---")
    args = parse_args()
    print(f"Arguments: {args}")
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- 加载模型和分词器 ---
    print(f"Step 1: Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    print("Tokenizer loaded.")
    
    print(f"Step 2: Loading model from {args.model_path}. This may take a while...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
        fp16=True, # 或者 bf16=True，根据你的硬件调整
    ).eval()
    print("Model loaded successfully.")

    # --- 加载数据集 ---
    print(f"Step 3: Loading dataset from {args.test_data_path}...")
    test_dataset = MimicDataset(args.test_data_path, image_root_path=args.image_root_path)
    print(f"Found {len(test_dataset)} samples in the dataset.")
    
    print("Initializing DataLoader...")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Changed to 0 for debugging
    )
    print("DataLoader initialized.")

    # --- 准备输出文件 ---
    print(f"Step 4: Preparing output directory and file at {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "evaluation_results.csv")
    print(f"Output will be saved to {output_path}")
    
    all_preds = []
    all_labels = []

    print("Step 5: Starting evaluation loop...")
    with open(output_path, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["ImagePath", "Instruction", "GroundTruth", "Prediction", "BLEU", "ROUGE-1", "ROUGE-L", "METEOR", "BERTScore-F1"])

        for batch in tqdm(test_dataloader, desc="Evaluating"):
            image_paths = batch["image_paths"]
            instructions = batch["instructions"]
            ground_truths = batch["ground_truths"]

            # --- 构建模型输入 ---
            queries = []
            for img_path, instruction in zip(image_paths, instructions):
                queries.append([{'image': img_path}, {'text': instruction}])

            # --- 生成预测 ---
            responses, _ = model.chat(tokenizer, queries=queries, history=None, max_new_tokens=args.max_new_tokens)
            
            # --- 后处理和指标计算 ---
            decoded_preds, decoded_labels = postprocess_text(responses, ground_truths)
            
            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)

            for i in range(len(decoded_preds)):
                pred = decoded_preds[i]
                label = decoded_labels[i]
                
                bleu_score = bleu.compute(predictions=[pred], references=[label], max_order=4)['bleu']
                rouge_score = rouge.compute(predictions=[pred], references=[label], rouge_types=['rouge1', 'rougeL'])
                meteor_score = meteor.compute(predictions=[pred], references=[label])['meteor']
                bert_score = bertscore.compute(predictions=[pred], references=[label], lang='en') # Assuming lang is 'en'
                bert_f1 = sum(bert_score['f1']) / len(bert_score['f1']) if bert_score['f1'] else 0.0

                writer.writerow([
                    image_paths[i],
                    instructions[i],
                    ground_truths[i],
                    responses[i],
                    bleu_score,
                    rouge_score['rouge1'],
                    rouge_score['rougeL'],
                    meteor_score,
                    bert_f1
                ])
    print("Evaluation loop finished.")

    # --- 计算并打印总体指标 ---
    print("\n--- Overall Evaluation Metrics ---")
    
    overall_bleu = bleu.compute(predictions=all_preds, references=all_labels, max_order=4)
    print(f"Overall BLEU: {overall_bleu['bleu']:.4f}")

    overall_rouge = rouge.compute(predictions=all_preds, references=all_labels, rouge_types=['rouge1', 'rouge2', 'rougeL'])
    print(f"Overall ROUGE-1: {overall_rouge['rouge1']:.4f}")
    print(f"Overall ROUGE-2: {overall_rouge['rouge2']:.4f}")
    print(f"Overall ROUGE-L: {overall_rouge['rougeL']:.4f}")

    overall_meteor = meteor.compute(predictions=all_preds, references=all_labels)
    print(f"Overall METEOR: {overall_meteor['meteor']:.4f}")

    overall_bert = bertscore.compute(predictions=all_preds, references=all_labels, lang=args.lang)
    avg_bert_f1 = sum(overall_bert['f1']) / len(overall_bert['f1'])
    print(f"Overall BERTScore F1: {avg_bert_f1:.4f}")
    
    print(f"\nEvaluation complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
