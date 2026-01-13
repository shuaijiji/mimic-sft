import json
import os

# --- Configuration ---
INPUT_FILE = "/home/lijiaji/mimic_cxr/img2text_test.jsonl"
OUTPUT_FILE = "eval_qa_dataset.jsonl"
IMAGE_BASE_DIR = "/home/lijiaji/mimic_cxr" # Base directory for relative image paths

def transform_dataset():
    """
    Transforms the multi-modal dataset into the general_qa format
    required by `swift eval`.
    """
    print(f"Starting transformation: {INPUT_FILE} -> {OUTPUT_FILE}")
    
    transformed_data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            original_item = json.loads(line)
            
            # Extract the user prompt and ground truth response
            user_content = original_item['messages'][0]['content']
            ground_truth = original_item['messages'][-1]['content']
            
            # Construct the full image path
            relative_image_path = original_item['images'][0]
            full_image_path = os.path.join(IMAGE_BASE_DIR, relative_image_path)
            
            # The user content already contains an <image> tag.
            # We will replace the placeholder tag with a tag containing the full path,
            # which is what the model expects.
            # Example query: "<image>/path/to/image.jpg\nDescribe this chest X-ray."
            query = user_content.replace('<image>', f'<image>{full_image_path}')

            transformed_data.append({
                "query": query,
                "response": ground_truth
            })

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in transformed_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Transformation complete. {len(transformed_data)} samples saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    transform_dataset()
