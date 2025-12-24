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

# --- ERROR INJECTION HELPERS (REALISTIC CLINGO ERRORS) ---

def corrupt_asp_code(code):
    """
    Takes valid ASP code and introduces a common syntax error based on 
    real-world LLM hallucinations and Clingo constraints.
    """
    options = []
    
    # 1. Missing Period (Termination)
    if code.strip().endswith('.'):
        broken = code.strip()[:-1]
        options.append((broken, "error: syntax error, unexpected <EOF>, expecting ."))

    # 2. Unsafe Variable (Safety)
    # Removing the body of a rule often makes the head variables unsafe.
    if ":-" in code and "." in code:
        parts = code.split(":-")
        head = parts[0].strip()
        # Only do this if the head actually has variables
        if re.search(r'\b[A-Z][a-zA-Z0-9_]*\b', head):
            broken = head + "."
            var = re.search(r'\b[A-Z][a-zA-Z0-9_]*\b', head).group(0)
            msg = f"error: unsafe variables in: {broken}\nnote: '{var}' is unsafe"
            options.append((broken, msg))

    # 3. Invalid Operator ( : instead of :- )
    if ":-" in code:
        broken = code.replace(":-", ":")
        options.append((broken, "error: syntax error, unexpected :, expecting . or ; or :-"))

    # 4. Invalid Negation (! instead of not)
    # LLMs often use C-style logic
    if " not " in code:
        broken = code.replace(" not ", " ! ")
        options.append((broken, "error: lexer error, unexpected !"))

    # 5. Invalid Comparison (\= instead of !=) -> From Example A
    if "!=" in code:
        broken = code.replace("!=", "\\=")
        options.append((broken, "error: syntax error, unexpected \\="))

    # 6. Invalid Comparison (:= instead of =) -> From Repair Rules
    if " = " in code:
        broken = code.replace(" = ", " := ")
        options.append((broken, "error: syntax error, unexpected :="))

    # 7. Double Braces in Aggregates ({{ }}) -> From Example B / Rule 2
    # LLMs often confuse this with Jinja templating
    if "{" in code and "}" in code and "{{" not in code:
        broken = code.replace("{", "{{").replace("}", "}}")
        options.append((broken, "error: syntax error, unexpected {"))

    # 8. Invalid Modulo (mod instead of \) -> From Example E
    if "\\" in code:
        broken = code.replace("\\", " mod ")
        options.append((broken, "error: syntax error, unexpected <IDENTIFIER> 'mod'"))

    # 9. 'or' in Body -> From Rule 8
    # Valid: a :- b. a :- c.  ->  Broken: a :- b or c.
    # This is hard to simulate on a single line safely, but we can look for comma separation
    if "," in code and ":-" in code:
        # Naive replacement of first comma in body
        parts = code.split(":-")
        if "," in parts[1]:
            broken = parts[0] + ":-" + parts[1].replace(",", " or ", 1)
            options.append((broken, "error: syntax error, unexpected <IDENTIFIER> 'or'"))

    if not options:
        return None
        
    return random.choice(options)

# --- AUGMENTATION HELPERS ---

def extract_domain_terms(nl_text):
    matches = re.findall(r'\b[A-Z][a-z]{2,}\b', nl_text)
    domain_terms = list(set([m for m in matches if m not in STOPWORDS]))
    if len(domain_terms) < 3:
        return ["Item", "Node", "Object", "Value", "Element"]
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

# --- MAIN ---

def process_single_problem(nl_path, asp_path):
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
    
    # 1. System Prompt with Expert Persona
    sys_msg = f"You are an expert Clingo programmer. Translate the following problem description into syntactically correct Answer Set Programming (ASP) code step-by-step.\n\n### Problem Description:\n{nl_content}"

    # 2. Original
    examples.append(create_conversation(sys_msg, steps))

    # 3. Shuffled
    if len(steps) > 2:
        for _ in range(NUM_SHUFFLES):
            shuffled_steps = steps[:] 
            first = shuffled_steps.pop(0) 
            random.shuffle(shuffled_steps)
            shuffled_steps.insert(0, first)
            examples.append(create_conversation(sys_msg, shuffled_steps))

    # 4. Renamed
    renamed_steps = []
    for ir, code in steps:
        mapping = get_local_variable_map(code, domain_pool)
        new_code = apply_variable_renaming(code, mapping)
        renamed_steps.append((ir, new_code))
    examples.append(create_conversation(sys_msg, renamed_steps))

    # 5. Noisy
    if ADD_NOISE:
        examples.append(create_conversation(sys_msg, steps, apply_noise=True))
        
    # 6. Repair Training (Synthetic Errors)
    if ADD_REPAIR_TASKS:
        # Specialized System Prompt for Debugging
        repair_conv = [{"role": "system", "content": "You are an expert Clingo code debugger. Fix the syntax errors in the following ASP rules."}]
        for ir, code in steps:
            result = corrupt_asp_code(code)
            if result:
                bad_code, error_msg = result
                # Simulate the Feedback Loop Prompt
                repair_conv.append({"role": "user", "content": f"The following code has an error:\n{bad_code}\n\nClingo Error:\n{error_msg}\n\nFix the code."})
                repair_conv.append({"role": "assistant", "content": code})
        
        if len(repair_conv) > 1:
            examples.append({"messages": repair_conv})

    return examples

def generate_asp_dataset_4(data_root, output_dir, split_ratio=0.8):
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root not found at: {data_root}")

    print(f"Traversing {data_root}...")
    problem_paths = []

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
                    problem_paths.append((nl_file, asp_file))

    print(f"Found {len(problem_paths)} unique problem instances.")
    random.seed(42)
    random.shuffle(problem_paths)

    split_idx = int(len(problem_paths) * split_ratio)
    train_paths = problem_paths[:split_idx]
    valid_paths = problem_paths[split_idx:]

    print(f"Splitting: {len(train_paths)} Train, {len(valid_paths)} Valid.")

    train_data = []
    for nl, asp in train_paths:
        train_data.extend(process_single_problem(nl, asp))

    valid_data = []
    for nl, asp in valid_paths:
        valid_data.extend(process_single_problem(nl, asp))

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
        t, v = generate_asp_dataset_4(data_in, data_out)
        print(f"Done. Final Dataset Size -> Train: {t} examples, Valid: {v} examples")
    except Exception as e:
        print(f"Error: {e}")