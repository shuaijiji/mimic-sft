#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算预测结果的评估指标（BLEU, ROUGE等）
支持 batch_infer.py 的输出格式：{"prediction": "...", "ground_truth": "..."}
"""
import json
import argparse
import re
from typing import List, Dict
import sys

# 尝试加载评估指标，支持多种方式
try:
    import evaluate
    USE_EVALUATE = True
except ImportError:
    USE_EVALUATE = False
    print("警告: evaluate 库未安装，将尝试使用 swift 内置指标")

# 尝试使用 swift 内置指标（适用于中文）
try:
    from swift.plugin import compute_rouge_bleu
    USE_SWIFT_METRICS = True
except ImportError:
    USE_SWIFT_METRICS = False
    if not USE_EVALUATE:
        print("警告: 无法导入 swift 模块，将仅使用 evaluate 库")


def _load_evaluate_metrics():
    """加载 evaluate 库的指标"""
    if not USE_EVALUATE:
        return None, None, None, None
    
    try:
        # 使用标准的 evaluate 指标名称
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")
        meteor_metric = evaluate.load("meteor")
        bertscore_metric = evaluate.load("bertscore")
        return bleu_metric, rouge_metric, meteor_metric, bertscore_metric
    except Exception as e:
        print(f"警告: evaluate 指标加载失败: {e}")
        return None, None, None, None


def calculate_metrics_with_evaluate(predictions: List[str], references: List[str], lang: str = 'en') -> Dict[str, float]:
    """使用 evaluate 库计算指标"""
    bleu, rouge, meteor, bertscore = _load_evaluate_metrics()
    results = {}
    
    if bleu is not None:
        try:
            # BLEU 需要列表格式的参考
            bleu_results = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
            # evaluate 库返回的 bleu 通常是 BLEU-4
            results['bleu-4'] = bleu_results.get('bleu', 0.0)
            
            # 计算 BLEU-1, BLEU-2, BLEU-3
            try:
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                smoothing = SmoothingFunction().method3
                
                bleu_1_scores = []
                bleu_2_scores = []
                bleu_3_scores = []
                
                for pred, ref in zip(predictions, references):
                    reference = [ref.split()]  # BLEU 需要列表格式
                    prediction = pred.split()
                    
                    if prediction and reference[0]:
                        bleu_1_scores.append(sentence_bleu(reference, prediction, weights=(1, 0, 0, 0), smoothing_function=smoothing))
                        bleu_2_scores.append(sentence_bleu(reference, prediction, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing))
                        bleu_3_scores.append(sentence_bleu(reference, prediction, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing))
                
                if bleu_1_scores:
                    results['bleu-1'] = sum(bleu_1_scores) / len(bleu_1_scores) * 100
                    results['bleu-2'] = sum(bleu_2_scores) / len(bleu_2_scores) * 100
                    results['bleu-3'] = sum(bleu_3_scores) / len(bleu_3_scores) * 100
            except Exception as e:
                print(f"警告: 计算 BLEU-1/2/3 失败: {e}，仅保留 BLEU-4")
        except Exception as e:
            print(f"警告: BLEU 计算失败: {e}")
            results['bleu-1'] = results['bleu-2'] = results['bleu-3'] = results['bleu-4'] = 0.0
    
    if rouge is not None:
        try:
            # ROUGE 需要列表格式的参考
            rouge_results = rouge.compute(predictions=predictions, references=[[ref] for ref in references])
            # 提取详细的 ROUGE 指标（precision, recall, f1）
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                rouge_key = rouge_type.replace('rouge', 'rouge-').replace('L', 'l')
                rouge_value = rouge_results.get(rouge_type, {})
                
                if isinstance(rouge_value, dict):
                    # 如果返回的是字典，提取 precision, recall, f1
                    results[f'{rouge_key}-precision'] = rouge_value.get('precision', 0.0)
                    results[f'{rouge_key}-recall'] = rouge_value.get('recall', 0.0)
                    results[f'{rouge_key}-f1'] = rouge_value.get('fmeasure', rouge_value.get('f1', 0.0))
                else:
                    # 如果返回的是单个值（通常是 F1），只保存 F1
                    results[f'{rouge_key}-f1'] = rouge_value if rouge_value else 0.0
        except Exception as e:
            print(f"警告: ROUGE 计算失败: {e}")
            for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l']:
                results[f'{rouge_type}-precision'] = 0.0
                results[f'{rouge_type}-recall'] = 0.0
                results[f'{rouge_type}-f1'] = 0.0
    
    if meteor is not None:
        try:
            meteor_results = meteor.compute(predictions=predictions, references=references)
            results['meteor'] = meteor_results.get('meteor', 0.0)
        except Exception as e:
            print(f"警告: METEOR 计算失败: {e}")
            results['meteor'] = 0.0
    
    if bertscore is not None:
        try:
            bert_results = bertscore.compute(predictions=predictions, references=references, lang=lang)
            if bert_results and 'f1' in bert_results:
                results['bertscore-f1'] = sum(bert_results['f1']) / len(bert_results['f1'])
            else:
                results['bertscore-f1'] = 0.0
        except Exception as e:
            print(f"警告: BERTScore 计算失败: {e}")
            results['bertscore-f1'] = 0.0
    
    return results


def calculate_metrics_with_swift(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """使用 swift 内置的指标计算函数（适用于中文），并计算详细的 ROUGE 指标"""
    import sys
    
    # 增加递归深度限制，避免递归超限错误
    original_recursion_limit = sys.getrecursionlimit()
    try:
        # 设置更高的递归深度（默认通常是1000）
        sys.setrecursionlimit(max(original_recursion_limit, 5000))
        
        # 尝试分批处理，避免一次性处理太多数据
        batch_size = 500
        all_results = []
        
        for i in range(0, len(predictions), batch_size):
            batch_preds = predictions[i:i + batch_size]
            batch_refs = references[i:i + batch_size]
            
            try:
                batch_results = compute_rouge_bleu(batch_preds, batch_refs)
                all_results.append(batch_results)
            except RecursionError as e:
                print(f"警告: 批次 {i//batch_size + 1} 递归深度超限，跳过该批次")
                continue
            except Exception as e:
                print(f"警告: 批次 {i//batch_size + 1} 处理失败: {e}")
                continue
        
        # 合并所有批次的结果
        if not all_results:
            return {}
        
        # 计算平均值（假设所有批次返回相同格式的字典）
        merged_results = {}
        for key in all_results[0].keys():
            values = [r.get(key, 0.0) for r in all_results if key in r]
            if values:
                merged_results[key] = sum(values) / len(values)
        
        # 计算 BLEU-1, BLEU-2, BLEU-3（swift 只返回 BLEU-4）
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import jieba
            smoothing = SmoothingFunction().method3
            
            bleu_1_scores = []
            bleu_2_scores = []
            bleu_3_scores = []
            
            for pred, ref in zip(predictions, references):
                # 使用 jieba 分词（与 swift 保持一致）
                prediction = list(jieba.cut(pred))
                reference = [list(jieba.cut(ref))]  # BLEU 需要列表格式
                
                if prediction and reference[0]:
                    bleu_1_scores.append(sentence_bleu(reference, prediction, weights=(1, 0, 0, 0), smoothing_function=smoothing))
                    bleu_2_scores.append(sentence_bleu(reference, prediction, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing))
                    bleu_3_scores.append(sentence_bleu(reference, prediction, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing))
            
            if bleu_1_scores:
                merged_results['bleu-1'] = sum(bleu_1_scores) / len(bleu_1_scores) * 100
                merged_results['bleu-2'] = sum(bleu_2_scores) / len(bleu_2_scores) * 100
                merged_results['bleu-3'] = sum(bleu_3_scores) / len(bleu_3_scores) * 100
        except Exception as e:
            print(f"警告: 计算 BLEU-1/2/3 失败: {e}，仅保留 BLEU-4")
        
        # 尝试计算详细的 ROUGE 指标（precision, recall, f1）
        try:
            import jieba
            from rouge.rouge import Rouge
            
            rouge = Rouge()
            rouge_precisions = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
            rouge_recalls = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
            rouge_f1s = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
            
            # 计算每个样本的详细 ROUGE 指标
            for pred, ref in zip(predictions, references):
                hypothesis = list(jieba.cut(pred))
                reference = list(jieba.cut(ref))
                if not hypothesis or not reference:
                    continue
                
                scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))[0]
                for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l']:
                    rouge_key = rouge_type.replace('rouge-', 'rouge')
                    if rouge_key in scores:
                        rouge_precisions[rouge_type].append(scores[rouge_key]['p'])
                        rouge_recalls[rouge_type].append(scores[rouge_key]['r'])
                        rouge_f1s[rouge_type].append(scores[rouge_key]['f'])
            
            # 计算平均值
            for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l']:
                if rouge_precisions[rouge_type]:
                    merged_results[f'{rouge_type}-precision'] = sum(rouge_precisions[rouge_type]) / len(rouge_precisions[rouge_type]) * 100
                    merged_results[f'{rouge_type}-recall'] = sum(rouge_recalls[rouge_type]) / len(rouge_recalls[rouge_type]) * 100
                    merged_results[f'{rouge_type}-f1'] = sum(rouge_f1s[rouge_type]) / len(rouge_f1s[rouge_type]) * 100
                else:
                    # 如果没有详细指标，使用 swift 返回的 F1 值
                    # swift 返回的键名格式：rouge-1, rouge-2, rouge-l
                    f1_key = rouge_type  # 已经是正确的格式
                    if f1_key in merged_results:
                        merged_results[f'{rouge_type}-f1'] = merged_results[f1_key]
                        # 删除旧的键，避免重复
                        del merged_results[f1_key]
                    else:
                        merged_results[f'{rouge_type}-f1'] = 0.0
                    merged_results[f'{rouge_type}-precision'] = 0.0
                    merged_results[f'{rouge_type}-recall'] = 0.0
        except Exception as e:
            print(f"警告: 计算详细 ROUGE 指标失败: {e}，仅使用 F1 值")
            # 如果计算详细指标失败，将现有的 F1 值重命名为 -f1
            for key in list(merged_results.keys()):
                if key.startswith('rouge-') and not key.endswith('-f1'):
                    merged_results[f'{key}-f1'] = merged_results[key]
                    merged_results[f'{key}-precision'] = 0.0
                    merged_results[f'{key}-recall'] = 0.0
        
        return merged_results
        
    except Exception as e:
        print(f"警告: swift 指标计算失败: {e}")
        return {}
    finally:
        # 恢复原始递归深度
        sys.setrecursionlimit(original_recursion_limit)


def is_invalid_prediction(pred: str) -> bool:
    """
    检查预测是否无效，需要被过滤掉
    无效情况包括：
    1. 空字符串或只包含空白字符
    2. 只包含图片ID（如 "img731", "image123"）
    3. 只包含占位符（如 "___.", "..."）
    4. 太短（少于3个字符）
    5. 只包含特殊字符或标点符号
    6. 只包含常见的短词（如 "Per", "The" 等，可能是截断的文本）
    """
    if not pred or not pred.strip():
        return True
    
    pred_clean = pred.strip()
    
    # 太短的预测（少于3个字符）可能是无效的
    if len(pred_clean) < 3:
        return True
    
    # 检查是否是图片ID格式（如 "img731", "image123", "pic001"）
    img_id_patterns = [
        r'^img\d+$',           # img731
        r'^image\d+$',          # image123
        r'^pic\d+$',            # pic001
        r'^photo\d+$',          # photo123
        r'^[a-z]{3,}\d+$',      # 任何3个以上小写字母+数字的组合（如 "img731"）
    ]
    for pattern in img_id_patterns:
        if re.match(pattern, pred_clean, re.IGNORECASE):
            return True
    
    # 检查是否是占位符或无效内容
    invalid_patterns = [
        r'^\.+$',               # 只有点号: "...", "...."
        r'^_+\.*$',              # 只有下划线和点号: "___.", "___"
        r'^[\._\-]+$',          # 只有标点符号
    ]
    for pattern in invalid_patterns:
        if re.match(pattern, pred_clean):
            return True
    
    # 检查是否是常见的短词（可能是截断的文本）
    # 只检查3-4个字符的常见词，避免误删有效的短预测
    if len(pred_clean) <= 4:
        invalid_short_words = [
            'per', 'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get',
            'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old',
            'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put',
            'say', 'she', 'too', 'use'
        ]
        if pred_clean.lower() in invalid_short_words:
            return True
    
    # 检查是否只包含特殊字符（没有字母或数字）
    if not re.search(r'[a-zA-Z0-9]', pred_clean):
        return True
    
    return False


def read_predictions(file_path: str) -> tuple[List[str], List[str]]:
    """
    读取预测文件，支持多种格式：
    1. batch_infer.py 格式: {"prediction": "...", "ground_truth": "..."}
    2. swift infer 格式: {"response": "...", ...}
    3. 其他格式: {"pred": "...", "ref": "..."} 等
    """
    predictions = []
    references = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # 支持多种字段名（按优先级）
                # batch_infer.py 格式
                pred = None
                for key in ['prediction', 'pred', 'response', 'output']:
                    if key in data:
                        pred_value = data[key]
                        if pred_value is not None and str(pred_value).strip():
                            pred = str(pred_value).strip()
                            break
                
                ref = None
                for key in ['ground_truth', 'reference', 'ref', 'label', 'labels']:
                    if key in data:
                        ref_value = data[key]
                        if ref_value is not None:
                            ref = str(ref_value).strip() if isinstance(ref_value, str) else str(ref_value)
                            break
                
                if pred is None:
                    # 只在调试模式下打印详细信息
                    if line_num <= 5 or line_num % 1000 == 0:
                        available_fields = list(data.keys())
                        print(f"警告: 第 {line_num} 行未找到有效的预测字段（可用字段: {available_fields}），跳过")
                    continue
                
                # 检查预测是否无效（图片ID、占位符等）
                if is_invalid_prediction(pred):
                    if line_num <= 5 or line_num % 1000 == 0:
                        print(f"警告: 第 {line_num} 行预测无效（内容: '{pred[:50]}...'），跳过")
                    continue
                
                predictions.append(pred)
                references.append(ref if ref else '')
                
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            except Exception as e:
                print(f"警告: 第 {line_num} 行处理失败: {e}")
                continue
    
    return predictions, references


def main():
    parser = argparse.ArgumentParser(description='计算预测结果的评估指标')
    parser.add_argument('--prediction_file', type=str, required=True, 
                       help='预测结果文件路径（JSONL格式），支持 batch_infer.py 输出格式')
    parser.add_argument('--original_dataset', type=str, default=None,
                       help='原始数据集路径（可选），如果预测文件中没有 ground_truth，则从此文件读取')
    parser.add_argument('--output_file', type=str, default=None, 
                       help='输出结果文件路径（可选，JSON格式）')
    parser.add_argument('--use_swift', action='store_true', 
                       help='优先使用 swift 内置的指标计算函数（适用于中文）')
    parser.add_argument('--use_evaluate', action='store_true', 
                       help='强制使用 evaluate 计算（需要可访问 Hugging Face 或有本地缓存）')
    parser.add_argument('--lang', type=str, default='en', 
                       help='BERTScore 语言代码（默认: en，中文使用 zh）')
    
    args = parser.parse_args()
    
    # 读取预测文件
    print(f"正在读取预测文件: {args.prediction_file}")
    predictions, references = read_predictions(args.prediction_file)
    
    # 如果预测文件中没有 ground_truth，尝试从原始数据集读取
    if not any(ref for ref in references) and args.original_dataset:
        print(f"预测文件中没有 ground_truth，尝试从原始数据集读取: {args.original_dataset}")
        try:
            with open(args.original_dataset, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        # 支持多种格式
                        if 'messages' in item and len(item['messages']) > 1:
                            ref = item['messages'][-1].get('content', '')
                        elif 'label' in item:
                            ref = item['label']
                        else:
                            continue
                        references.append(ref.strip())
                    except:
                        continue
            # 确保长度匹配
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
        except Exception as e:
            print(f"警告: 从原始数据集读取失败: {e}")
    
    if not predictions:
        print("错误: 未找到任何预测结果")
        sys.exit(1)
    
    if not any(ref for ref in references):
        print("错误: 未找到任何 ground_truth 标签")
        print("提示: 请确保预测文件包含 'ground_truth' 字段，或使用 --original_dataset 参数")
        sys.exit(1)
    
    # 过滤空引用
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if r]
    if len(valid_pairs) < len(predictions):
        print(f"警告: 过滤了 {len(predictions) - len(valid_pairs)} 个空引用")
        predictions, references = zip(*valid_pairs) if valid_pairs else ([], [])
    
    print(f"成功读取 {len(predictions)} 条有效预测结果")
    
    # 计算指标
    results = {}
    
    if (args.use_swift or USE_SWIFT_METRICS) and not args.use_evaluate:
        print("尝试使用 swift 内置指标计算函数...")
        swift_results = calculate_metrics_with_swift(list(predictions), list(references))
        
        if swift_results:
            print("swift 指标计算成功")
            results = swift_results
        else:
            print("swift 指标计算失败，回退到 evaluate 库...")
            if USE_EVALUATE:
                results = calculate_metrics_with_evaluate(list(predictions), list(references), args.lang)
            else:
                print("错误: evaluate 库不可用，无法计算指标")
                sys.exit(1)
    else:
        print("使用 evaluate 库计算指标...")
        if USE_EVALUATE:
            results = calculate_metrics_with_evaluate(list(predictions), list(references), args.lang)
        else:
            print("错误: evaluate 库不可用")
            sys.exit(1)
    
    if not results:
        print("错误: 无法计算任何指标，请检查依赖安装")
        sys.exit(1)
    
    # 打印结果（按指标类型分组）
    print("\n" + "=" * 50)
    print("评估结果:")
    print("=" * 50)
    
    # 按指标类型分组显示
    metric_groups = {
        'BLEU': [],
        'ROUGE-1': [],
        'ROUGE-2': [],
        'ROUGE-L': [],
        'METEOR': [],
        'BERTScore': [],
        '其他': []
    }
    
    for metric, value in sorted(results.items()):
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        
        # 分类指标
        if 'bleu' in metric.lower():
            # 统一 BLEU 键名显示，按顺序排列
            if metric == 'bleu-1':
                metric_groups['BLEU'].append(('BLEU-1', formatted_value))
            elif metric == 'bleu-2':
                metric_groups['BLEU'].append(('BLEU-2', formatted_value))
            elif metric == 'bleu-3':
                metric_groups['BLEU'].append(('BLEU-3', formatted_value))
            elif metric == 'bleu-4':
                metric_groups['BLEU'].append(('BLEU-4', formatted_value))
            elif metric == 'bleu':
                metric_groups['BLEU'].append(('BLEU', formatted_value))
            else:
                metric_groups['BLEU'].append((metric, formatted_value))
        elif 'rouge-1' in metric.lower():
            metric_groups['ROUGE-1'].append((metric, formatted_value))
        elif 'rouge-2' in metric.lower():
            metric_groups['ROUGE-2'].append((metric, formatted_value))
        elif 'rouge-l' in metric.lower() or 'rouge-l' in metric.lower():
            metric_groups['ROUGE-L'].append((metric, formatted_value))
        elif 'meteor' in metric.lower():
            metric_groups['METEOR'].append((metric, formatted_value))
        elif 'bertscore' in metric.lower():
            metric_groups['BERTScore'].append((metric, formatted_value))
        else:
            metric_groups['其他'].append((metric, formatted_value))
    
    # 按组打印（BLEU 组需要特殊排序）
    for group_name, metrics in metric_groups.items():
        if metrics:
            print(f"\n{group_name}:")
            if group_name == 'BLEU':
                # BLEU 按数字顺序排序
                bleu_order = {'BLEU-1': 1, 'BLEU-2': 2, 'BLEU-3': 3, 'BLEU-4': 4, 'BLEU': 5}
                metrics.sort(key=lambda x: bleu_order.get(x[0], 99))
            else:
                metrics.sort()
            for metric, value in metrics:
                print(f"  {metric}: {value}")
    
    print("=" * 50)
    
    # 保存结果
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output_file}")


if __name__ == '__main__':
    main()
