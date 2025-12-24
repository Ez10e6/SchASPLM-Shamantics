import os
import json
import random
import re
import string

# --- CONFIGURATION ---
NUM_SHUFFLES = 2
ADD_NOISE = True 
ADD_REPAIR_TASKS = True 

STOPWORDS = {
    "The", "And", "For", "But", "With", "From", "This", "That", "Each", "Every", 
    "All", "Some", "Any", "Not", "Given", "When", "Then", "If", "Where", "Such",
    "There", "Here", "What", "Which", "While", "Since", "After", "Before"
}

# --- ERROR INJECTION HELPERS ---

def corrupt_asp_code(code):
    corruptions = []
    
    if code.strip().endswith('.'):
        broken = code.strip()[:-1]
        corruptions.append((broken, "error: syntax error, unexpected <EOF>, expecting ."))

    if ":-" in code and "." in code:
        parts = code.split(":-")
        head = parts[0].strip()
        if re.search(r'\b[A-Z][a-zA-Z0-9_]*\b', head):
            broken = head + "."
            var = re.search(r'\b[A-Z][a-zA-Z0-9_]*\b', head).group(0)
            msg = f"error: unsafe variables in: {broken}\nnote: '{var}' is unsafe"
            corruptions.append((broken, msg))

    if ":-" in code:
        broken = code.replace(":-", ":")
        corruptions.append((broken, "error: syntax error, unexpected :, expecting . or ; or :-"))

    if " not " in code:
        broken = code.replace(" not ", " ! ")
        corruptions.append((broken, "error: lexer error, unexpected !"))

    if "!=" in code:
        broken = code.replace("!=", "\\=")
        corruptions.append((broken, "error: syntax error, unexpected \\="))

    if " = " in code:
        broken = code.replace(" = ", " := ")
        corruptions.append((broken, "error: syntax error, unexpected :="))

    if "{" in code and "}" in code and "{{" not in code:
        broken = code.replace("{", "{{").replace("}", "}}")
        corruptions.append((broken, "error: syntax error, unexpected {"))

    if "\\" in code:
        broken = code.replace("\\", " mod ")
        corruptions.append((broken, "error: syntax error, unexpected <IDENTIFIER> 'mod'"))

    if "," in code and ":-" in code:
        parts = code.split(":-")
        if "," in parts[1]:
            broken = parts[0] + ":-" + parts[1].replace(",", " or ", 1)
            corruptions.append((broken, "error: syntax error, unexpected <IDENTIFIER> 'or'"))

    return corruptions

# --- AUGMENTATION HELPERS ---

def extract_domain_terms(nl_text):
    matches = re.findall(r'\b[A-Z][a-z]{2,}\b', nl_text)
    domain_terms = list(set([m for m in matches if m not in STOPWORDS]))
    if len(domain_terms) < 3: return ["Item", "Node", "Object", "Value", "Element"]
    return domain_terms

def get_local_variable_map(code_snippet, domain_pool):
    code_no_strings = re.sub(r'".*?"', '', code_snippet)
    vars_found = set(re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', code_no_strings))
    if not vars_found: return {}

    canonical_pool = [f"V{i}" for i in range(20)]
    alphabet_pool = list("XYZABCDEFGHIJKLMNOPQRSTUVW")
    combined_pool = canonical_pool + alphabet_pool + domain_pool
    random.shuffle(combined_pool)
    
    vars_list = list(vars_found)
    mapping = {}
    for i, var in enumerate(vars_list):
        if i < len(combined_pool): mapping[var] = combined_pool[i]
        else: mapping[var] = f"Var{i}"
    return mapping

def apply_variable_renaming(code, mapping):
    if not mapping: return code
    parts = re.split(r'(".*?")', code)
    new_parts = []
    sorted_vars = sorted(mapping.keys(), key=len, reverse=True)
    for part in parts:
        if part.startswith('"'): new_parts.append(part)
        else:
            for old_var in sorted_vars:
                new_var = mapping[old_var]
                part = re.sub(r'\b' + old_var + r'\b', new_var, part)
            new_parts.append(part)
    return "".join(new_parts)

def add_input_noise(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def parse_asp_to_steps(asp_content):
    lines = asp_content.splitlines()
    steps = []
    current_ir = None
    current_code_buffer = []
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('%'):
            if current_ir is not None:
                code_block = "\n".join(current_code_buffer).strip()
                if code_block: steps.append((current_ir, code_block))
            current_ir = line.lstrip('%').strip()
            current_code_buffer = []
        else:
            current_code_buffer.append(line)
    if current_ir is not None:
        code_block = "\n".join(current_code_buffer).strip()
        if code_block: steps.append((current_ir, code_block))
    return steps

def create_conversation(system_prompt, steps, apply_noise=False):
    conv = [{"role": "system", "content": system_prompt}]
    for ir, code in steps:
        user_content = ir
        if apply_noise: user_content = add_input_noise(ir)
        conv.append({"role": "user", "content": f"Implement: {user_content}"})
        conv.append({"role": "assistant", "content": code})
    return {"messages": conv}

# --- CORE PROCESSING ---

def process_single_problem(nl_path, asp_path, include_repairs=True):
    """
    Generates examples for one problem file. 
    include_repairs: If False, skips generating synthetic syntax error tasks.
    """
    examples = []
    try:
        with open(nl_path, 'r', encoding='utf-8') as f: nl_content = f.read().strip()
        with open(asp_path, 'r', encoding='utf-8') as f: asp_content = f.read().strip()
    except Exception as e:
        print(f"Error reading {nl_path}: {e}")
        return []

    steps = parse_asp_to_steps(asp_content)
    if not steps: return []

    domain_pool = extract_domain_terms(nl_content)
    sys_msg = f"You are an expert Clingo programmer. Translate the following problem description into syntactically correct Answer Set Programming (ASP) code step-by-step.\n\n### Problem Description:\n{nl_content}"

    # 1. Original (Multi-turn)
    examples.append(create_conversation(sys_msg, steps))

    # 2. Shuffled (Multi-turn)
    if len(steps) > 2:
        for _ in range(NUM_SHUFFLES):
            shuffled_steps = steps[:] 
            first = shuffled_steps.pop(0) 
            random.shuffle(shuffled_steps)
            shuffled_steps.insert(0, first)
            examples.append(create_conversation(sys_msg, shuffled_steps))

    # 3. Renamed (Multi-turn)
    renamed_steps = []
    for ir, code in steps:
        mapping = get_local_variable_map(code, domain_pool)
        new_code = apply_variable_renaming(code, mapping)
        renamed_steps.append((ir, new_code))
    examples.append(create_conversation(sys_msg, renamed_steps))

    # 4. Noisy (Multi-turn)
    if ADD_NOISE:
        examples.append(create_conversation(sys_msg, steps, apply_noise=True))

    # 5. REPAIR TRAINING (Conditional)
    # Only run this if it's an original file (not a variant), to avoid duplicate repair data.
    if ADD_REPAIR_TASKS and include_repairs:
        debug_sys_msg = "You are an expert Clingo code debugger. Fix the syntax errors in the following ASP rules."
        
        for ir, code in steps:
            possible_corruptions = corrupt_asp_code(code)
            
            for bad_code, error_msg in possible_corruptions:
                repair_example = {
                    "messages": [
                        {"role": "system", "content": debug_sys_msg},
                        {"role": "user", "content": f"The following code has an error:\n{bad_code}\n\nClingo Error:\n{error_msg}\n\nFix the code."},
                        {"role": "assistant", "content": code}
                    ]
                }
                examples.append(repair_example)

    return examples

def generate_asp_dataset_max(data_root, output_dir, split_ratio=0.8):
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root not found at: {data_root}")

    print(f"Traversing {data_root}...")
    
    # Dictionary to group variants together
    # Key: Base Problem Path (without _variant), Value: List of (NL, ASP) pairs
    problem_groups = {}

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
                
                if os.path.exists(nl_file) and os.path.exists(asp_file):
                    # Identify the "Base" ID
                    # If folder is "Prob1_variant_gemini", base is "Prob1"
                    base_name = problem_folder.split("_variant")[0] 
                    group_id = f"{category}/{difficulty}/{base_name}"
                    
                    if group_id not in problem_groups:
                        problem_groups[group_id] = []
                    
                    problem_groups[group_id].append({
                        "folder_name": problem_folder,
                        "nl_path": nl_file,
                        "asp_path": asp_file
                    })

    # Convert dictionary to list of groups
    all_groups = list(problem_groups.values())
    print(f"Found {len(all_groups)} unique logical problems (spanning {sum(len(g) for g in all_groups)} folders).")
    
    # Shuffle GROUPS to prevent leakage
    random.seed(42)
    random.shuffle(all_groups)

    split_idx = int(len(all_groups) * split_ratio)
    train_groups = all_groups[:split_idx]
    valid_groups = all_groups[split_idx:]

    print(f"Splitting: {len(train_groups)} Problem Groups for Training, {len(valid_groups)} for Validation.")

    # Helper to process a list of groups
    def process_groups(groups):
        dataset = []
        for group in groups:
            for item in group:
                folder_name = item["folder_name"]
                # Only generate repairs if it is the ORIGINAL folder (no _variant in name)
                is_original = "_variant" not in folder_name
                
                data = process_single_problem(
                    item["nl_path"], 
                    item["asp_path"], 
                    include_repairs=is_original
                )
                dataset.extend(data)
        return dataset

    train_data = process_groups(train_groups)
    valid_data = process_groups(valid_groups)

    # Final Shuffle of examples
    random.shuffle(train_data)
    random.shuffle(valid_data)

    os.makedirs(output_dir, exist_ok=True)
    
    def save_jsonl(path, data):
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')

    save_jsonl(os.path.join(output_dir, "train.jsonl"), train_data)
    save_jsonl(os.path.join(output_dir, "valid.jsonl"), valid_data)
    
    return len(train_data), len(valid_data)

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_in = os.path.join(root_dir, "data", "Benchmark Data")
    data_out = os.path.join(root_dir, "data")
    try:
        t, v = generate_asp_dataset_max(data_in, data_out)
        print(f"Done. Final Dataset Size -> Train: {t} examples, Valid: {v} examples")
    except Exception as e:
        print(f"Error: {e}")