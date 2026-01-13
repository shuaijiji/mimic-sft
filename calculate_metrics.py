import json
import os
import glob
from tqdm import tqdm
import subprocess
import sys

# --- Helper function to install packages ---
def install(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# --- Try to import evaluate and install if not found ---
try:
    import evaluate
    from bert_score import score
except ImportError:
    print("Required libraries not found. Installing 'evaluate' and 'bert_score'...")
    install("evaluate")
    install("bert_score")
    import evaluate
    from bert_score import score
    print("Installation complete.")


# --- Configuration ---
# Path to your original test set file (containing the ground truth)
# IMPORTANT: This must match the TEST_JSON in your eval.sh
GROUND_TRUTH_FILE = "/home/lijiaji/mimic_cxr/img2text_test.jsonl"

# Directory where the model's predictions are saved
# IMPORTANT: This is the default output directory for swift infer
PREDICTIONS_DIR = "/home/lijiaji/ms-swift/output/mimic-lora-optimized/v30-20260107-160108/checkpoint-21000/infer_result"


def find_latest_prediction_file(directory):
    """Find the most recently created .jsonl file in a directory."""
    list_of_files = glob.glob(os.path.join(directory, '*.jsonl'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def load_data(ground_truth_path, predictions_path):
    """Load ground truth and predictions, ensuring they are aligned."""
    print(f"Loading ground truth from: {ground_truth_path}")
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truths = [json.loads(line) for line in f]

    print(f"Loading predictions from: {predictions_path}")
    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions_raw = [json.loads(line) for line in f]

    # The prediction file contains extra info, we only need the 'response'
    # It also might be in a different order, so we align them.
    # Let's assume the ground truth file has a unique identifier if possible,
    # or that the order is preserved.
    # For swift infer, the order should be preserved.

    if len(ground_truths) != len(predictions_raw):
        print(f"Warning: Mismatch in number of samples. Ground truth: {len(ground_truths)}, Predictions: {len(predictions_raw)}")
        # Attempt to align based on a common key if available, otherwise truncate
        # For now, we'll assume order is preserved and truncate to the smaller size
        min_len = min(len(ground_truths), len(predictions_raw))
        ground_truths = ground_truths[:min_len]
        predictions_raw = predictions_raw[:min_len]


    # Extract the actual text
    references = [[item['messages'][-1]['content']] for item in ground_truths]
    candidates = [item['response'] for item in predictions_raw]

    print(f"Loaded {len(references)} aligned samples for evaluation.")
    return candidates, references

def calculate_all_metrics(predictions, references):
    """Calculate and print all desired metrics."""
    print("\nCalculating metrics...")

    # 1. ROUGE
    print("--- Calculating ROUGE ---")
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=predictions, references=references)
    print(rouge_results)

    # 2. BLEU
    print("\n--- Calculating BLEU ---")
    bleu = evaluate.load('bleu')
    bleu_results = bleu.compute(predictions=predictions, references=references)
    print(bleu_results)

    # 3. METEOR
    print("\n--- Calculating METEOR ---")
    meteor = evaluate.load('meteor')
    meteor_results = meteor.compute(predictions=predictions, references=references)
    print(meteor_results)

    # 4. BERT-Score
    # BERT-Score expects flat lists of strings
    flat_references = [ref[0] for ref in references]
    print("\n--- Calculating BERT-Score (this may take a while) ---")
    P, R, F1 = score(predictions, flat_references, lang='en', verbose=True)
    bert_score_results = {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }
    print(f"BERT-Score Results: {bert_score_results}")


if __name__ == "__main__":
    latest_preds = find_latest_prediction_file(PREDICTIONS_DIR)
    if not latest_preds:
        print(f"Error: No prediction files (.jsonl) found in '{PREDICTIONS_DIR}'")
        print("Please run 'bash eval.sh' first to generate the predictions.")
        sys.exit(1)

    print(f"Found latest prediction file: {latest_preds}")

    predictions, references = load_data(GROUND_TRUTH_FILE, latest_preds)

    if not predictions:
        print("Error: No data loaded. Cannot proceed with evaluation.")
        sys.exit(1)

    calculate_all_metrics(predictions, references)
