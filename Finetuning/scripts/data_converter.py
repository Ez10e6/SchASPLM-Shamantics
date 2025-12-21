import os
import json
import random

def generate_asp_dataset(data_root, output_dir, split_ratio=0.8):
    """
    Traverses the benchmark directory to create train.jsonl and valid.jsonl.
    """
    all_examples = []
    sys_prompt_nl = "You are an expert AI researcher. Translate the following natural language problem description into syntactically correct clingo ASP code."
    sys_prompt_ir = "You are an expert AI researcher. Translate the following structured intermediate representation (IR) into syntactically correct clingo ASP code."

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root not found at: {data_root}")

    # 1. Collect all possible pairs
    for category in os.listdir(data_root):
        cat_path = os.path.join(data_root, category)
        if not os.path.isdir(cat_path): continue
        
        for difficulty in os.listdir(cat_path):
            diff_path = os.path.join(cat_path, difficulty)
            if not os.path.isdir(diff_path): continue
            
            for problem_folder in os.listdir(diff_path):
                prob_path = os.path.join(diff_path, problem_folder)
                asp_file = os.path.join(prob_path, "ASP.txt")
                if not os.path.exists(asp_file): continue
                    
                with open(asp_file, 'r', encoding='utf-8') as f: 
                    assistant_content = f.read()

                for source_ext, prompt in [("NL.txt", sys_prompt_nl), ("IR.txt", sys_prompt_ir)]:
                    source_path = os.path.join(prob_path, source_ext)
                    if os.path.exists(source_path):
                        with open(source_path, 'r', encoding='utf-8') as f:
                            user_content = f.read()
                        
                        all_examples.append({
                            "messages": [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": assistant_content}
                            ]
                        })

    # 2. Shuffle and Split
    random.seed(42) # For reproducibility
    random.shuffle(all_examples)
    
    split_idx = int(len(all_examples) * split_ratio)
    train_data = all_examples[:split_idx]
    valid_data = all_examples[split_idx:]

    # 3. Save to standard names
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, data in [("train.jsonl", train_data), ("valid.jsonl", valid_data)]:
        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
    
    return len(train_data), len(valid_data)