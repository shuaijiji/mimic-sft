# debug_load_model.py
import torch
from transformers import AutoModelForCausalLM
import sys
import time

def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_load_model.py <path_to_model>")
        sys.exit(1)

    model_path = sys.argv[1]
    
    print(f"--- Attempting to load model from: {model_path} ---")
    print("This script will test if the model can be loaded into CPU RAM.")
    
    try:
        start_time = time.time()
        print(f"[{time.ctime()}] Starting AutoModelForCausalLM.from_pretrained...")
        
        # We load the model directly to CPU to isolate the problem.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            fp16=True,
        )
        
        end_time = time.time()
        print(f"[{time.ctime()}] SUCCESS! Model loaded successfully.")
        print(f"Loading took {end_time - start_time:.2f} seconds.")
        
        print("Model object created:", type(model))
        
    except Exception as e:
        print(f"--- ERROR ---")
        print(f"An error occurred while loading the model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
