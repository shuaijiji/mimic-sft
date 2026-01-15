# prepare_dataset_for_swift.py
import json
import os
import argparse

def convert_dataset_for_swift(input_file, output_file, image_root_path=''):
    """
    Converts the original jsonl dataset to a format compatible with `swift infer`.
    The output format will be a jsonl file where each line is a dictionary
    containing a 'query' list.
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        count = 0
        for line in f_in:
            try:
                original_item = json.loads(line)
                
                # Extract image path
                relative_image_path = original_item["images"][0]
                full_image_path = os.path.join(image_root_path, relative_image_path)
                
                # Extract instruction
                messages = original_item.get("messages", [])
                instruction_content = messages[0].get("content", "")
                instruction = instruction_content.replace("<image>\\n", "").strip()
                
                # Create the new query structure
                # The format is a list of dictionaries
                query_list = [
                    {'image': full_image_path},
                    {'text': instruction}
                ]
                
                # Create the new item for swift infer
                new_item = {"query": query_list}
                
                f_out.write(json.dumps(new_item) + '\\n')
                count += 1
                
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Skipping a malformed line: {line.strip()}. Error: {e}")
    
    print(f"Successfully converted {count} items.")
    print(f"Dataset for swift infer saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset for swift infer.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input .jsonl file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output .jsonl file for swift.")
    parser.add_argument('--image_root_path', type=str, default="", help="Root path for the image files.")
    args = parser.parse_args()
    
    convert_dataset_for_swift(args.input_file, args.output_file, args.image_root_path)
