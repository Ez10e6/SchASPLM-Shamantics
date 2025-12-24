import os
import json
import random
import re

def parse_asp_to_steps(asp_content):
    """
    Parses an ASP string into steps based on comments.
    
    Assumes the format:
    % Description of the constraint (IR)
    logic_rule(X) :- condition(X).
    
    Returns:
        list of tuples: [(ir_text, asp_code), ...]
    """
    lines = asp_content.splitlines()
    steps = []
    
    current_ir = None
    current_code_buffer = []

    for line in lines:
        line = line.strip()
        if not line: 
            continue

        if line.startswith('%'):
            # If we have a previous IR block pending, save it
            if current_ir is not None:
                code_block = "\n".join(current_code_buffer).strip()
                if code_block: # Only add if there is actual code
                    steps.append((current_ir, code_block))
            
            # Start new block
            # Remove the % and leading/trailing whitespace
            current_ir = line.lstrip('%').strip()
            current_code_buffer = []
        else:
            # It's code, add to buffer
            current_code_buffer.append(line)

    # Append the final block if exists
    if current_ir is not None:
        code_block = "\n".join(current_code_buffer).strip()
        if code_block:
            steps.append((current_ir, code_block))
            
    return steps

def generate_asp_dataset_2(data_root, output_dir, split_ratio=0.8):
    """
    Traverses the benchmark directory to create train.jsonl and valid.jsonl
    using a multi-turn conversation format (Step-by-Step).
    """
    all_examples = []
    
    # Check if root exists
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root not found at: {data_root}")

    print(f"Traversing {data_root}...")

    # 1. Traverse Directory Structure: Category -> Difficulty -> Problem Folder
    for category in os.listdir(data_root):
        cat_path = os.path.join(data_root, category)
        if not os.path.isdir(cat_path): continue
        
        for difficulty in os.listdir(cat_path):
            diff_path = os.path.join(cat_path, difficulty)
            if not os.path.isdir(diff_path): continue
            
            for problem_folder in os.listdir(diff_path):
                prob_path = os.path.join(diff_path, problem_folder)
                
                nl_file = os.path.join(prob_path, "NL.txt")
                asp_file = os.path.join(prob_path, "ASP.txt")
                
                # Ensure both files exist
                if not (os.path.exists(nl_file) and os.path.exists(asp_file)):
                    continue
                    
                # 2. Read Files
                try:
                    with open(nl_file, 'r', encoding='utf-8') as f:
                        nl_content = f.read().strip()
                    
                    with open(asp_file, 'r', encoding='utf-8') as f:
                        asp_content = f.read().strip()
                except Exception as e:
                    print(f"Error reading {prob_path}: {e}")
                    continue

                # 3. Parse ASP into Step-by-Step pairs
                steps = parse_asp_to_steps(asp_content)
                
                if not steps:
                    print(f"Warning: No commented steps found in {asp_file}. Skipping.")
                    continue

                # 4. Construct Multi-Turn Conversation
                # System Prompt includes the full NL description for context
                conversation = [
                    {
                        "role": "system", 
                        "content": f"You are an expert AI researcher. Translate the following problem description into syntactically correct Clingo ASP code step-by-step.\n\n### Problem Description:\n{nl_content}"
                    }
                ]

                # Add turns
                for ir_text, asp_code in steps:
                    # User asks to implement specific logic
                    conversation.append({
                        "role": "user",
                        "content": f"Implement: {ir_text}"
                    })
                    # Assistant provides the code
                    conversation.append({
                        "role": "assistant",
                        "content": asp_code
                    })

                all_examples.append({"messages": conversation})

    # 5. Shuffle and Split
    print(f"Found {len(all_examples)} problem instances.")
    random.seed(42) 
    random.shuffle(all_examples)
    
    split_idx = int(len(all_examples) * split_ratio)
    train_data = all_examples[:split_idx]
    valid_data = all_examples[split_idx:]

    # 6. Save to JSONL
    os.makedirs(output_dir, exist_ok=True)
    
    def save_jsonl(path, data):
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')

    save_jsonl(os.path.join(output_dir, "train.jsonl"), train_data)
    save_jsonl(os.path.join(output_dir, "valid.jsonl"), valid_data)
    
    return len(train_data), len(valid_data)

if __name__ == "__main__":
    # Allow running this script directly for testing
    import sys
    # Assume script is run from 'scripts/' folder, move up to find data
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_in = os.path.join(root_dir, "data", "Benchmark Data")
    data_out = os.path.join(root_dir, "data")
    
    try:
        t, v = generate_asp_dataset(data_in, data_out)
        print(f"Generated {t} training and {v} validation examples.")
    except Exception as e:
        print(f"Execution failed: {e}")