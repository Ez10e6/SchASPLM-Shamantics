import os
import json
import time
import random
import re
import glob
import clingo
from dotenv import load_dotenv
from google import genai
from google.genai import types
from utils_ft import to_relative

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-3-flash-preview" 

# Split Ratio
SPLIT_RATIO = 0.8 

# If True, Gemini will try to fix invalid Clingo code found in the dataset
ENABLE_AUTO_REPAIR = True

# Probability to generate a "broken -> fixed" training example for robustness
REPAIR_TASK_PROB = 0.2 

# Set to True to force structure regeneration even if cache exists
FORCE_REGEN = False 

# --- TONE VARIATIONS ---
TONES = [
    "Academic/Formal: Use precise mathematical terminology, passive voice, and logical quantifiers.",
    "Casual Developer: Speak like a programmer writing quick comments. Use slang like 'sanity check'.",
    "Business Requirements: Speak like a project manager. Focus on 'rules', 'policies', and 'compliance'.",
    "Succinct/Telegraphic: Use as few words as possible. Robot-like.",
    "Didactic/Instructional: Explain it like a teacher to a student."
]

# --- PROMPTS ---

EXTRACTION_PROMPT = """
You are an expert in Answer Set Programming (specifically Clingo 5+ syntax). 
I will give you a full problem description (NL) and a full ASP solution.

Your task is to DECONSTRUCT this solution into the specific structural components required for a scheduler system.
Return the result as a valid JSON object.

### TONE INSTRUCTION
For the 'description' fields in the JSON, you must write the natural language requirement using this specific tone:
**{selected_tone}**

### SYNTAX INSTRUCTION
Ensure all extracted 'asp_rule' fields contain VALID Clingo code.
- Aggregates must use {{ }} syntax (e.g., #count {{ X : p(X) }}).
- Choice rules must use {{ }} (e.g., {{ a; b }} = 1).
- Do not output generic Prolog syntax.
- Ensure every rule ends with a period.

---
Input NL:
{nl_content}

Input ASP:
{asp_content}

---
JSON OUTPUT REQUIREMENTS:
Structure the JSON exactly like this:
{{
  "instance_template": "The ASP code block defining predicates/domains. NO rules, just domain definitions.",
  "generator": "The ASP choice rule(s) that generate the search space.",
  "hard_constraints": [
    {{
      "description": "A description of this specific constraint written in the requested TONE.",
      "asp_rule": "The specific ASP rule(s) implementing this constraint."
    }},
    ... (repeat for all hard constraints)
  ],
  "soft_constraints": [
    {{
      "description": "A description of this preference/optimization written in the requested TONE.",
      "asp_rule": "The weak constraint or #minimize statement."
    }},
    ... (repeat for all soft constraints)
  ]
}}
"""

REPAIR_PROMPT = """
You are an expert Clingo debugger. The following ASP code snippet is syntactically invalid or incomplete.
Your task is to fix it based on the provided natural language description.

Context Description: {description}
Invalid Code: {code}

Requirements:
1. Fix syntax errors (missing dots, wrong aggregate brackets, etc.).
2. Ensure it matches the description.
3. Output ONLY the corrected Clingo code. No markdown, no explanations.
"""

# Training System Prompts
SYS_PROMPT_INSTANCE = (
    "You are an expert in Answer Set Programming. "
    "Task: Convert natural language instance descriptions into ASP facts and domain definitions. "
    "Output: Provide ONLY the valid Clingo code. Do not provide any explanation, comments, or markdown formatting."
)

SYS_PROMPT_GENERATOR = (
    "You are an expert in Answer Set Programming. "
    "Task: Generate the choice rules (generators) for the given problem. "
    "Output: Provide ONLY the valid Clingo code. Do not provide any explanation, comments, or markdown formatting."
)

SYS_PROMPT_HARD = (
    "You are an expert in Answer Set Programming. "
    "Task: Generate a hard constraint based on the description. "
    "CRITICAL: You must use EXACTLY the predicates defined in the <<instance_template>>. Do not invent new predicates. "
    "Output: Provide ONLY valid Clingo code."
)

SYS_PROMPT_SOFT = (
    "You are an expert in Answer Set Programming. "
    "Task: Generate a soft constraint (optimization). "
    "CRITICAL: You must use EXACTLY the predicates defined in the <<instance_template>>. Do not invent new predicates. "
    "Output: Provide ONLY valid Clingo code."
)

SYS_PROMPT_REPAIR_TASK = "You are an expert Clingo debugger. Fix the syntax errors in the provided ASP code. Output: Provide ONLY the corrected code."

# --- HELPER FUNCTIONS ---

def is_valid_clingo(code_snippet):
    """Checks syntax using Clingo library."""
    if not code_snippet or not code_snippet.strip(): return False
    cleaned = code_snippet.replace("```clingo", "").replace("```asp", "").replace("```", "").strip()
    if not cleaned.endswith('.'): return False
    try:
        ctl = clingo.Control()
        ctl.add("base", [], cleaned)
        return True
    except:
        return False

def clean_gemini_code(text):
    """Removes markdown fences if Gemini adds them."""
    return text.replace("```clingo", "").replace("```asp", "").replace("```", "").strip()

def repair_with_gemini(code, description):
    """Calls Gemini to fix invalid code."""
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[REPAIR_PROMPT.format(description=description, code=code)],
            config=types.GenerateContentConfig(temperature=0.2) # Low temp for precision
        )
        return clean_gemini_code(response.text)
    except Exception as e:
        print(f"    [Repair Error] {e}")
        return code

def corrupt_asp_code(code):
    """
    Takes valid ASP code and returns a LIST of all possible corrupted versions
    paired with their specific Clingo error messages.
    
    CRITICAL: Error messages here must match the Clingo Python API output 
    seen in scheduler.py logs exactly.
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
            # Note: Safety errors usually DO include the variable name in the 'note' section
            # Log: "error: unsafe variables in: ... note: 'X' is unsafe"
            var = re.search(r'\b[A-Z][a-zA-Z0-9_]*\b', head).group(0)
            msg = f"error: unsafe variables in: {broken}\nnote: '{var}' is unsafe"
            corruptions.append((broken, msg))

    # --- 3. RULE OPERATORS ---
    if ":-" in code:
        # Mistake: Colon only (Python style)
        broken = code.replace(":-", ":")
        corruptions.append((broken, "error: syntax error, unexpected :, expecting . or ; or :-"))
        
        # Mistake: "if" keyword
        # Log shows: unexpected <IDENTIFIER> (it usually hides 'if')
        broken_if = code.replace(":-", " if ")
        corruptions.append((broken_if, "error: syntax error, unexpected <IDENTIFIER>, expecting . or ; or :-"))
        
        # Mistake: Implication arrows
        broken_arrow = code.replace(":-", " <= ")
        corruptions.append((broken_arrow, "error: syntax error, unexpected <=, expecting . or ; or :-"))

    # --- 4. NEGATION ---
    if " not " in code:
        # Mistake: C-style bang
        broken = code.replace(" not ", " ! ")
        corruptions.append((broken, "error: lexer error, unexpected !"))
        
        # Mistake: Tilde
        broken_tilde = code.replace(" not ", " ~ ")
        corruptions.append((broken_tilde, "error: lexer error, unexpected ~"))
        
        # Mistake: Prolog style (\+)
        # Log showed: error: syntax error, unexpected \
        broken_prolog = code.replace(" not ", " \\+ ")
        corruptions.append((broken_prolog, "error: syntax error, unexpected \\"))

    # --- 5. COMPARISON OPERATORS ---
    if "!=" in code:
        broken = code.replace("!=", "\\=") 
        corruptions.append((broken, "error: syntax error, unexpected \\"))
        
        broken_sql = code.replace("!=", "<>") 
        corruptions.append((broken_sql, "error: syntax error, unexpected <>, expecting !="))

    if " = " in code:
        # Mistake: ==
        broken = code.replace(" = ", " == ") 
        corruptions.append((broken, "error: syntax error, unexpected ==, expecting ="))
        
        # Mistake: CLP/Prolog Arithmetic (#=)
        # Log showed: error: lexer error, unexpected #
        broken_clp = code.replace(" = ", " #= ")
        corruptions.append((broken_clp, "error: lexer error, unexpected #"))

    # --- 6. LOGIC OPERATORS ---
    if ", " in code:
        # Mistake: "and" keyword
        broken = code.replace(", ", " and ", 1)
        # Log: unexpected <IDENTIFIER> (Specifics hidden)
        corruptions.append((broken, "error: syntax error, unexpected <IDENTIFIER>"))

    if ", " in code and ":-" in code:
        parts = code.split(":-")
        if ", " in parts[1]:
            # Mistake: "or" keyword in body
            broken = parts[0] + ":-" + parts[1].replace(", ", " or ", 1)
            # Log: unexpected <IDENTIFIER>
            corruptions.append((broken, "error: syntax error, unexpected <IDENTIFIER>"))

    # --- 7. AGGREGATES & SETS ---
    # Mistake: Double Braces {{ }}
    if "{" in code and "}" in code and "{{" not in code:
        broken = code.replace("{", "{{").replace("}", "}}")
        corruptions.append((broken, "error: syntax error, unexpected {"))

    # Mistake: Set-Builder Pipe "|"
    # Log showed: error: syntax error, unexpected |
    if ":" in code and "{" in code:
        broken = code.replace(":", "|")
        corruptions.append((broken, "error: syntax error, unexpected |"))

    # Mistake: Missing #count/#sum keyword
    if "#count" in code:
        broken = code.replace("#count", "count")
        # Log: unexpected <IDENTIFIER> ('count' acts as a function name here)
        corruptions.append((broken, "error: syntax error, unexpected <IDENTIFIER>"))

    # --- 8. PRIMITIVES ---
    # Mistake: Single Quotes
    if '"' in code:
        broken = code.replace('"', "'")
        corruptions.append((broken, "error: lexer error, unexpected '"))

    return corruptions

# --- CORE LOGIC ---

def get_structure_data(folder_path, nl_text, asp_text):
    """Loads JSON from cache OR queries Gemini + Auto-Repairs invalid code."""
    cache_path = os.path.join(folder_path, "structure.json")
    
    # 1. Load Cache
    if os.path.exists(cache_path) and not FORCE_REGEN:
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data, False # False = No API call made
        except:
            pass

    # 2. Extract Structure (API)
    print(f"  [Extraction] Querying Gemini for structure...")
    current_tone = random.choice(TONES)
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[EXTRACTION_PROMPT.format(nl_content=nl_text, asp_content=asp_text, selected_tone=current_tone)],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = json.loads(response.text)
    except Exception as e:
        print(f"  [Error] Extraction failed: {e}")
        return None, False

    # 3. Auto-Repair Invalid Code (Distillation)
    if ENABLE_AUTO_REPAIR:
        modified = False
        
        # Check Generator
        if not is_valid_clingo(data.get("generator", "")):
            print(f"    [Repairing] Generator...")
            fixed = repair_with_gemini(data["generator"], "Generate the choice rule/generator for this problem.")
            if is_valid_clingo(fixed):
                data["generator"] = fixed
                modified = True

        # Check Hard Constraints
        for hc in data.get("hard_constraints", []):
            if not is_valid_clingo(hc.get("asp_rule", "")):
                print(f"    [Repairing] Hard Constraint...")
                fixed = repair_with_gemini(hc["asp_rule"], hc["description"])
                if is_valid_clingo(fixed):
                    hc["asp_rule"] = fixed
                    modified = True
        
        # Check Soft Constraints
        for sc in data.get("soft_constraints", []):
            if not is_valid_clingo(sc.get("asp_rule", "")):
                print(f"    [Repairing] Soft Constraint...")
                fixed = repair_with_gemini(sc["asp_rule"], sc["description"])
                if is_valid_clingo(fixed):
                    sc["asp_rule"] = fixed
                    modified = True
        
        if modified:
            print(f"    [Success] Data repaired via Distillation.")

    # 4. Save to Cache (Permanent Fix)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return data, True # True = API call made

def process_single_folder(folder_path):
    examples = []
    
    try:
        if not os.path.exists(os.path.join(folder_path, "NL.txt")) or \
           not os.path.exists(os.path.join(folder_path, "ASP.txt")):
            return []

        with open(os.path.join(folder_path, "NL.txt"), 'r', encoding='utf-8') as f: nl_text = f.read()
        with open(os.path.join(folder_path, "ASP.txt"), 'r', encoding='utf-8') as f: asp_text = f.read()

        data, used_api = get_structure_data(folder_path, nl_text, asp_text)
        if not data: return []
        
        if used_api: time.sleep(1) # Rate limit

        inst_temp = data.get("instance_template", "")
        gen_rule = data.get("generator", "")
        
        # --- NEW CONTEXT BLOCK STRUCTURE ---
        # We prepare the context block, but we DON'T add the specific instruction yet.
        context_block = (
            f"CONTEXT:\n"
            f"<<problem_description>>\n{nl_text}\n\n"
            f"<<instance_template>>\n{inst_temp}\n\n"
            f"<<generator>>\n{gen_rule}"
        )

        def add_example(sys, user, code):
            if is_valid_clingo(code):
                # 1. Standard Translation Task
                examples.append({"messages": [{"role": "system", "content": sys}, {"role": "user", "content": user}, {"role": "assistant", "content": code}]})
                
                # 2. Synthetic Repair Task
                if random.random() < REPAIR_TASK_PROB:
                    corruptions = corrupt_asp_code(code)
                    if corruptions:
                        bad_code, err = random.choice(corruptions)
                        # For repair, we put the broken code first, context is implicit or less relevant
                        repair_prompt = f"The following code has an error:\n{bad_code}\n\nClingo Error:\n{err}\n\nFix the code."
                        examples.append({"messages": [{"role": "system", "content": SYS_PROMPT_REPAIR_TASK}, {"role": "user", "content": repair_prompt}, {"role": "assistant", "content": code}]})

        # --- 1. Instance Generation ---
        # Prompt: Context -> Request
        inst_user_msg = f"Problem Description:\n{nl_text}\n\n### TASK\nCreate the instance template (define predicates and domains)."
        add_example(SYS_PROMPT_INSTANCE, inst_user_msg, inst_temp)

        # --- 2. Generator Generation ---
        gen_user_msg = f"Problem Description:\n{nl_text}\n\n<<instance_template>>\n{inst_temp}\n\n### TASK\nCreate the generator (choice rules)."
        add_example(SYS_PROMPT_GENERATOR, gen_user_msg, gen_rule)

        # --- 3. Hard Constraints ---
        for hc in data.get("hard_constraints", []):
            desc = hc.get('description', 'Constraint')
            user_msg = f"{context_block}\n\n### TARGET CONSTRAINT\nImplement this hard constraint:\n{desc}"
            add_example(SYS_PROMPT_HARD, user_msg, hc.get('asp_rule', ''))

        # --- 4. Soft Constraints ---
        for sc in data.get("soft_constraints", []):
            desc = sc.get('description', 'Optimization')
            user_msg = f"{context_block}\n\n### TARGET OPTIMIZATION\nImplement this soft constraint:\n{desc}"
            add_example(SYS_PROMPT_SOFT, user_msg, sc.get('asp_rule', ''))
            
        return examples

    except Exception as e:
        print(f"Failed to process {folder_path}: {e}")
        return []

def main(data_root, output_dir):
    print(f"Scanning {to_relative(data_root)}...")
    problem_groups = {}
    
    for root, dirs, files in os.walk(data_root):
        if "NL.txt" in files and "ASP.txt" in files:
            folder_name = os.path.basename(root)
            parent_dir = os.path.dirname(root)
            base_name = folder_name.split("_variant")[0] if "_variant" in folder_name else folder_name
            group_id = os.path.join(parent_dir, base_name)
            
            if group_id not in problem_groups: problem_groups[group_id] = []
            problem_groups[group_id].append(root)

    group_keys = list(problem_groups.keys())
    random.seed(42)
    random.shuffle(group_keys)
    
    split_idx = int(len(group_keys) * SPLIT_RATIO)
    train_keys = group_keys[:split_idx]
    valid_keys = group_keys[split_idx:]
    
    train_data = []
    valid_data = []

    print(f"\n--- Processing Training Set ({len(train_keys)} Groups) ---")
    for key in train_keys:
        for folder in problem_groups[key]:
            print(f"> {os.path.basename(folder)}")
            examples = process_single_folder(folder)
            train_data.extend(examples)

    print(f"\n--- Processing Validation Set ({len(valid_keys)} Groups) ---")
    for key in valid_keys:
        for folder in problem_groups[key]:
            print(f"> {os.path.basename(folder)}")
            examples = process_single_folder(folder)
            valid_data.extend(examples)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
        for entry in train_data: f.write(json.dumps(entry) + '\n')
    with open(os.path.join(output_dir, "valid.jsonl"), 'w', encoding='utf-8') as f:
        for entry in valid_data: f.write(json.dumps(entry) + '\n')

    print(f"\nDone! Train: {len(train_data)} | Valid: {len(valid_data)}")
