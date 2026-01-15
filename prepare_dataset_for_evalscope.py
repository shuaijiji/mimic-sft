# prepare_dataset_for_evalscope.py
import json
import os
import argparse

def convert_dataset(input_file, output_file, image_root_path=''):
    """
    Converts the original jsonl dataset to a json format compatible with evalscope.
    """
    eval_data = []
    with open(input_file, 'r') as f_in:
        for line in f_in:
            try:
                original_item = json.loads(line)
                
                # Extract image path
                relative_image_path = original_item["images"][0]
                full_image_path = os.path.join(image_root_path, relative_image_path)
                
                # Extract prompt and label
                messages = original_item.get("messages", [])
                instruction_content = messages[0].get("content", "")
                prompt = instruction_content.replace("<image>\\n", "").strip()
                label = messages[1].get("content", "")
                
                # Create the new item
                new_item = {
                    "image": full_image_path,
                    "prompt": prompt,
                    "label": label
                }
                eval_data.append(new_item)
                
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Skipping a malformed line: {line.strip()}. Error: {e}")

    with open(output_file, 'w') as f_out:
        json.dump(eval_data, f_out, indent=2)
        
    print(f"Successfully converted {len(eval_data)} items.")
    print(f"Dataset saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset for evalscope.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input .jsonl file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output .json file.")
    parser.add_argument('--image_root_path', type=str, default="", help="Root path for the image files.")
    args = parser.parse_args()
    
    convert_dataset(args.input_file, args.output_file, args.image_root_path)
