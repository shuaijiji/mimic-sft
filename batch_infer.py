import json
import torch
from tqdm import tqdm
import subprocess
import sys
import os

# --- Ensure correct libraries are installed ---
try:
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration as Qwen3VLForCausalLM
    from peft import PeftModel
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure you are in the 'QwenGen' conda environment and have run:")
    print("pip install --upgrade ms-swift transformers accelerate peft sentencepiece")
    sys.exit(1)

# --- Configuration ---
MODEL_PATH = "/home/lijiaji/Qwen/Qwen3-VL-8B-Thinking"
CKPT_PATH = "/home/lijiaji/ms-swift/output/mimic-lora-optimized/v30-20260107-160108/checkpoint-21000"
TEST_DATASET_PATH = "/home/lijiaji/mimic_cxr/img2text_test.jsonl"
IMAGE_BASE_DIR = "/home/lijiaji/mimic_cxr"
OUTPUT_FILE_PATH = "predictions.jsonl"
BATCH_SIZE = 4  # Adjust based on your GPU memory

def main():
    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model and Tokenizer ---
    print("Loading base model and tokenizer...")
    # For Qwen3-VL, trust_remote_code is necessary
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen3VLForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # --- 2. Load LoRA Adapter ---
    print(f"Loading LoRA adapter from: {CKPT_PATH}")
    model = PeftModel.from_pretrained(model, CKPT_PATH)
    model.eval()
    print("Model and adapter loaded successfully.")

    # --- 3. Load Test Dataset ---
    print(f"Loading test dataset from: {TEST_DATASET_PATH}")
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    print(f"Found {len(test_data)} samples in the test set.")

    # --- 4. Batch Inference ---
    print(f"Starting batch inference with batch size: {BATCH_SIZE}")

    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as out_f:
        for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
            batch_data = test_data[i:i + BATCH_SIZE]
            
            prompts = []
            
            for item in batch_data:
                messages = item['messages']
                relative_image_path = item['images'][0]
                full_image_path = os.path.join(IMAGE_BASE_DIR, relative_image_path)
                
                # Format the prompt for the tokenizer
                query = tokenizer.from_list_format([
                    {'image': full_image_path},
                    {'text': messages[0]['content'].replace('<image>', '').strip()},
                ])
                prompts.append(query)

            # Tokenize and generate
            inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=2048)
            
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Save results for this batch
            for j, response in enumerate(responses):
                original_item = batch_data[j]
                # The response might contain the prompt, so we clean it
                cleaned_response = response.replace(prompts[j], '').strip()
                result_item = {
                    "prediction": cleaned_response,
                    "ground_truth": original_item['messages'][-1]['content']
                }
                out_f.write(json.dumps(result_item) + '\n')

    print(f"\nInference complete. Results saved to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()
