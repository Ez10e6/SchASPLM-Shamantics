import os
import json
import random
import re
import string

# --- CONFIGURATION ---
ADD_NOISE = True        # Add lowercased/unpunctuated variations
ADD_REPAIR_TASKS = True # Add synthetic "Fix this code" examples

# --- ERROR INJECTION HELPERS (Synthetic Repairs) ---

def corrupt_asp_code(code):
    """
    Takes valid ASP code and returns a LIST of all possible corrupted versions
    paired with their specific Clingo error messages.
    Includes distinct error patterns based on common errors and also based on
    errors that can be expected of models that we mostly trained on other more
    popular languages.
    """
    corruptions = []
    
    # --- 1. TERMINATION ---
    if code.strip().endswith('.'):
        broken = code.strip()[:-1]
        corruptions.append((broken, "error: syntax error, unexpected <EOF>, expecting ."))

    # --- 2. SAFETY (Unsafe Variables) ---
    if ":-" in code and "." in code:
        parts = code.split(":-")
        head = parts[0].strip()
        if re.search(r'\b[A-Z][a-zA-Z0-9_]*\b', head):
            broken = head + "."
            var = re.search(r'\b[A-Z][a-zA-Z0-9_]*\b', head).group(0)
            msg = f"error: unsafe variables in: {broken}\nnote: '{var}' is unsafe"
            corruptions.append((broken, msg))

    # --- 3. RULE OPERATORS ---
    if ":-" in code:
        # Mistake: Colon only (Python dictionary style)
        broken = code.replace(":-", ":")
        corruptions.append((broken, "error: syntax error, unexpected :, expecting . or ; or :-"))
        
        # Mistake: "if" keyword (Python style)
        broken_if = code.replace(":-", " if ")
        corruptions.append((broken_if, "error: syntax error, unexpected <IDENTIFIER> 'if', expecting . or ; or :-"))
        
        # Mistake: Implication arrows (Math style)
        broken_arrow = code.replace(":-", " <= ")
        corruptions.append((broken_arrow, "error: syntax error, unexpected <=, expecting . or ; or :-"))

    # --- 4. NEGATION ---
    if " not " in code:
        # Mistake: C-style bang
        broken = code.replace(" not ", " ! ")
        corruptions.append((broken, "error: lexer error, unexpected !"))
        
        # Mistake: Tilde (Bitwise/Math)
        broken_tilde = code.replace(" not ", " ~ ")
        corruptions.append((broken_tilde, "error: lexer error, unexpected ~"))

    # --- 5. COMPARISON OPERATORS ---
    if "!=" in code:
        broken = code.replace("!=", "\\=") # Prolog style
        corruptions.append((broken, "error: syntax error, unexpected \\="))
        
        broken_sql = code.replace("!=", "<>") # SQL style
        corruptions.append((broken_sql, "error: syntax error, unexpected <>, expecting !="))

    if " = " in code:
        broken = code.replace(" = ", " == ") # Python style
        corruptions.append((broken, "error: syntax error, unexpected ==, expecting ="))
        
        broken_assign = code.replace(" = ", " := ") # Pascal/Go style
        corruptions.append((broken_assign, "error: syntax error, unexpected :="))

    # --- 6. LOGIC OPERATORS ---
    if ", " in code:
        # Mistake: "and" keyword
        broken = code.replace(", ", " and ", 1)
        corruptions.append((broken, "error: syntax error, unexpected <IDENTIFIER> 'and'"))

    if ", " in code and ":-" in code:
        parts = code.split(":-")
        if ", " in parts[1]:
            # Mistake: "or" keyword in body
            broken = parts[0] + ":-" + parts[1].replace(", ", " or ", 1)
            corruptions.append((broken, "error: syntax error, unexpected <IDENTIFIER> 'or'"))

    # --- 7. AGGREGATES & SETS ---
    # Mistake: Double Braces {{ }} (F-string hallucination)
    if "{" in code and "}" in code and "{{" not in code:
        broken = code.replace("{", "{{").replace("}", "}}")
        corruptions.append((broken, "error: syntax error, unexpected {"))

    # Mistake: Set-Builder Pipe "|" instead of Colon ":"
    if ":" in code and "{" in code:
        broken = code.replace(":", "|")
        corruptions.append((broken, "error: syntax error, unexpected |"))

    # Mistake: Missing #count/#sum keyword
    if "#count" in code:
        broken = code.replace("#count", "")
        corruptions.append((broken, "error: syntax error, unexpected {"))
    if "#sum" in code:
        broken = code.replace("#sum", "")
        corruptions.append((broken, "error: syntax error, unexpected {"))

    # --- 8. ARITHMETIC ---
    # Mistake: Modulo keyword
    if "\\" in code:
        broken = code.replace("\\", " mod ")
        corruptions.append((broken, "error: syntax error, unexpected <IDENTIFIER> 'mod'"))

    # Mistake: Hallucinated Operator =#
    if re.search(r'\b[A-Z0-9]+\s*=\s*[A-Z0-9]+', code):
        broken = code.replace("=", "=#", 1)
        corruptions.append((broken, "error: lexer error, unexpected #"))

    # --- 9. PRIMITIVES ---
    # Mistake: Single Quotes for strings (Python/SQL style)
    if '"' in code:
        broken = code.replace('"', "'")
        corruptions.append((broken, "error: lexer error, unexpected '"))

    # Mistake: Wildcard * instead of _
    if "_" in code:
        broken = code.replace("_", "*")
        corruptions.append((broken, "error: syntax error, unexpected *"))

    # Mistake: C-Style Comments
    if "%" in code:
        broken = code.replace("%", "//")
        corruptions.append((broken, "error: lexer error, unexpected /"))

    # Mistake: "minimize {...}" instead of "#minimize {...}"
    directives = ["minimize", "maximize", "show", "const"]
    for d in directives:
        target = f"#{d}"
        if target in code:
            broken = code.replace(target, d)
            corruptions.append((broken, f"error: syntax error, unexpected <IDENTIFIER> '{d}'"))

    return corruptions

# --- AUGMENTATION HELPERS ---

def get_local_variable_map(code_snippet):
    """
    Finds ASP variables and maps them to Abstract (X, Y) or Canonical (V0, V1) names.
    Purely structural, NO semantic domain terms to prevent overfitting.
    """
    code_no_strings = re.sub(r'".*?"', '', code_snippet)
    vars_found = set(re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', code_no_strings))
    if not vars_found: return {}

    canonical_pool = [f"V{i}" for i in range(20)]
    alphabet_pool = list("XYZABCDEFGHIJKLMNOPQRSTUVW")
    combined_pool = canonical_pool + alphabet_pool
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

# --- CORE PARSING ---

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

# --- CORE PROCESSING LOGIC ---

def process_single_problem(nl_path, asp_path, include_repairs=True):
    examples = []
    try:
        with open(nl_path, 'r', encoding='utf-8') as f: nl_content = f.read().strip()
        with open(asp_path, 'r', encoding='utf-8') as f: asp_content = f.read().strip()
    except Exception as e:
        print(f"Error reading {nl_path}: {e}")
        return []

    steps = parse_asp_to_steps(asp_content)
    if not steps: return []

    sys_msg = f"You are an expert Clingo programmer. You will be given a sequence of instructions that together form a program. Output only the ASP code for the current instruction. Do not repeat earlier rules."

    # 1. ORIGINAL (Multi-Turn)
    examples.append(create_conversation(sys_msg, steps))

    # 2. RENAMED (Multi-Turn)
    renamed_steps = []
    for ir, code in steps:
        mapping = get_local_variable_map(code) 
        new_code = apply_variable_renaming(code, mapping)
        renamed_steps.append((ir, new_code))
    examples.append(create_conversation(sys_msg, renamed_steps))

    # 3. NOISY (Multi-Turn)
    if ADD_NOISE:
        examples.append(create_conversation(sys_msg, steps, apply_noise=True))
        
    # 4. REPAIR TRAINING (Independent Single-Turn Examples)
    if ADD_REPAIR_TASKS and include_repairs:
        debug_sys_msg = "You are an expert Clingo code debugger. Fix the syntax errors in the following ASP rules. Output only valid clingo code. No explanation."
        for ir, code in steps:
            possible_corruptions = corrupt_asp_code(code)
            for bad_code, error_msg in possible_corruptions:
                repair_example = {
                    "messages": [
                        {"role": "system", "content": debug_sys_msg},
                        {"role": "user", "content": f"The following code has an error:\n{bad_code}\n\nClingo Error:\n{error_msg}\n\nFix the code. Output only valid clingo code. No explanation."},
                        {"role": "assistant", "content": code}
                    ]
                }
                examples.append(repair_example)

    return examples

# --- MAIN ---

def generate_asp_dataset(data_root, output_dir, split_ratio=0.8):
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root not found at: {data_root}")

    print(f"Traversing {data_root} recursively...")

    problem_groups = {}

    # Recursive Walk to find ALL problems regardless of folder depth
    for root, dirs, files in os.walk(data_root):
        if "NL.txt" in files and "ASP.txt" in files:
            nl_path = os.path.join(root, "NL.txt")
            asp_path = os.path.join(root, "ASP.txt")
            folder_name = os.path.basename(root)

            # Identify Base Name (remove _variant_...)
            if "_variant" in folder_name:
                base_name = folder_name.split("_variant")[0]
            else:
                base_name = folder_name
            
            # Unique ID for grouping (Parent path + Base Name)
            # This ensures "Routing/TSP" and "Manually/TSP" are treated distinct if desired,
            # but variants like "Routing/TSP_variant_1" stay glued to "Routing/TSP"
            parent_dir = os.path.dirname(root)
            group_id = os.path.join(parent_dir, base_name)
            
            if group_id not in problem_groups:
                problem_groups[group_id] = []
            
            problem_groups[group_id].append({
                "folder_name": folder_name,
                "nl_path": nl_path,
                "asp_path": asp_path
            })

    all_groups = list(problem_groups.values())
    print(f"Found {len(all_groups)} unique logical problems (spanning {sum(len(g) for g in all_groups)} total folder variants).")

    # Shuffle GROUPS
    random.seed(42)
    random.shuffle(all_groups)

    # Split Groups
    split_idx = int(len(all_groups) * split_ratio)
    train_groups = all_groups[:split_idx]
    valid_groups = all_groups[split_idx:]

    print(f"Splitting: {len(train_groups)} Training Groups, {len(valid_groups)} Validation Groups.")

    def process_groups(groups):
        dataset = []
        for group in groups:
            for item in group:
                folder_name = item["folder_name"]
                # Only generate repairs for the ORIGINAL folder
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

    # Final Shuffle
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
        t, v = generate_asp_dataset(data_in, data_out)
        print(f"Done. Final Dataset Size -> Train: {t} examples, Valid: {v} examples")
    except Exception as e:
        print(f"Error: {e}")