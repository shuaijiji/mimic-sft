import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import subprocess
import sys

# --- Helper function to install packages ---
def install(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# --- Try to import peft and install if not found ---
try:
    from peft import PeftModel
except ImportError:
    print("PEFT library not found. Installing 'peft'...")
    install("peft")
    from peft import PeftModel
    print("Installation complete.")


# --- Configuration ---
MODEL_PATH = "/home/lijiaji/Qwen/Qwen3-VL-8B-Thinking"
CKPT_PATH = "/home/lijiaji/ms-swift/output/mimic-lora-optimized/v30-20260107-160108/checkpoint-21000"
TEST_DATASET_PATH = "/home/lijiaji/mimic_cxr/img2text_test.jsonl"
OUTPUT_FILE_PATH = "predictions_for_metrics.jsonl"
BATCH_SIZE = 4  # Adjust based on your GPU memory

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load Model and Tokenizer ---
print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
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
results = []
print(f"Starting batch inference with batch size: {BATCH_SIZE}")

# Use tqdm for a progress bar
for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
    batch_data = test_data[i:i + BATCH_SIZE]
    
    # Prepare batch input
    batch_messages = [item['messages'] for item in batch_data]
    batch_images = [item.get('images') for item in batch_data] # Use .get for safety
    
    # This part is specific to Qwen-VL's `chat` method
    # We need to format the input correctly for the model
    # For simplicity, we process one by one within the batch loop, 
    # but a more advanced implementation could use batch generation.
    for idx, messages in enumerate(batch_messages):
        image_path = batch_images[idx][0] if batch_images[idx] else None
        
        # Construct the query for the model's chat interface
        query = tokenizer.from_list_format(messages)
        
        # Generate response
        with torch.no_grad():
            response, history = model.chat(tokenizer, query=query, history=None, image=image_path)
        
        # Get the ground truth response
        ground_truth = messages[-1]['content']
        
        results.append({
            "prediction": response,
            "ground_truth": ground_truth
        })


# --- 5. Save Results ---
print(f"Inference complete. Saving {len(results)} results to {OUTPUT_FILE_PATH}")
with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
    for item in results:
        f.write(json.dumps(item) + '\n')

print("All done!")
