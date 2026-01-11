import os
import sys
from ..utils_ft import to_relative, get_prompts_path
import json
import time
import random
import re
import glob
import clingo
from dotenv import load_dotenv
from google import genai
from google.genai import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

# --- CONFIGURATION ---
load_dotenv()
print("GOOGLE_API_KEY present?", bool(os.getenv("GOOGLE_API_KEY")))
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-3-flash-preview" 

# Split Ratio
SPLIT_RATIO = 0.8 

# If True, Gemini will try to fix invalid Clingo code found in the dataset
ENABLE_AUTO_REPAIR = True

# Probability to generate a "broken -> fixed" training example for robustness
REPAIR_TASK_PROB = 0.5 

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

# --- LOAD PROMPTS FROM FILES ---

def load_prompt(filename):
    """Reads a prompt file from the centralized prompts directory."""
    path = os.path.join(get_prompts_path(), filename)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Critical error: Prompt file not found at {path}")

EXTRACTION_PROMPT = load_prompt("extraction.txt")
REPAIR_PROMPT = load_prompt("repair_gemini.txt")

# Training System Prompts
SYS_PROMPT_INSTANCE = load_prompt("instance.txt")
SYS_PROMPT_GENERATOR = load_prompt("generator.txt")
SYS_PROMPT_HARD = load_prompt("hard_constraint.txt")
SYS_PROMPT_SOFT = load_prompt("soft_constraint.txt")
SYS_PROMPT_REPAIR_TASK = load_prompt("repair_task.txt")

# --- HELPER FUNCTIONS ---

def get_clingo_error(code_snippet, context=""):
    """
    Parses and grounds ASP code to find specific errors.
    Includes context (Instance + Generator) to validate variable safety.
    """
    if not isinstance(code_snippet, str):
        try:
            code_snippet = str(code_snippet)
        except:
            return "Error: Input is not a string."

    if not code_snippet or not code_snippet.strip():
        return "Error: Code is empty."
        
    cleaned = code_snippet.replace("```clingo", "").replace("```asp", "").replace("```", "").strip()
    if not cleaned.endswith('.'): 
        return "error: syntax error, unexpected <EOF>, expecting ."

    # COMBINE: We check the snippet in the presence of its dependencies
    full_program = context + "\n" + cleaned

    messages = []
    def logger(code, message):
        # Captures grounding errors (unsafe variables) and syntax errors
        if "RuntimeError" in str(code):
            messages.append(str(message))

    try:
        ctl = clingo.Control(logger=logger)
        ctl.add("base", [], full_program)
        # Grounding is necessary to find "Unsafe Variable" errors
        ctl.ground([("base", [])])
        
        if not messages:
            return None 
            
    except RuntimeError:
        if messages:
            return "\n".join(messages)
        return "error: general clingo runtime/syntax error"
    except Exception as e:
        return str(e)

    if messages:
        return "\n".join(messages)
        
    return None

def is_valid_clingo(code_snippet, context=""):
    """Boolean wrapper for easy logic checks with context."""
    return get_clingo_error(code_snippet, context) is None

def clean_gemini_code(text):
    """Robustly extracts raw ASP code from LLM responses."""
    text = text.strip()
    
    # 1. Handle cases where the LLM returns a JSON list (e.g. ["rule."])
    if text.startswith('[') and text.endswith(']'):
        try:
            items = json.loads(text)
            if isinstance(items, list) and len(items) > 0:
                text = items[0] # Take the first element
        except:
            pass # Not valid JSON, continue

    # 2. Remove Markdown fences
    text = text.replace("```clingo", "").replace("```asp", "").replace("```", "")
    
    # 3. Final cleaning of quotes that might have been wrapped by the list
    text = text.strip().strip('"').strip("'")
    
    return text.strip()

def repair_with_gemini(code, description, error_message, context=""):
    """Calls Gemini to fix invalid code, providing the specific error."""
    try:
        # We pass the error message into the prompt
        content = REPAIR_PROMPT.format(
            context=context,
            description=description, 
            code=code, 
            error=error_message
        )

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[content],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=1
            )
        )
        clean_code = clean_gemini_code(response.text)
        return clean_code
    except Exception as e:
        print(f"    [Repair Error] {e}")
        return code

def corrupt_asp_code(code, val_context=""):
    """
    Generates synthetic broken ASP code candidates and verifies them 
    against the real Clingo compiler.
    
    Mappings to Repair Rules:
    1. Termination: Removes '.'
    2. Aggregates: {{ }}, | instead of :
    3. Safety: Creates unsafe variables by removing bodies
    4. Ranges/Comparisons: \=, <>, ==, :=, mod
    5. Choice Rules: (Covered by general syntax checks)
    6. Negation: !, ~
    7. Markdown: Injects ``` code fences
    8. Operators: and, or (in body), if
    """
    if not code or not code.strip():
        return []

    candidates = []
    
    # --- RULE 1: TERMINATION & TOKENIZATION ---
    if code.strip().endswith('.'):
        candidates.append(code.strip()[:-1]) # Remove dot

    # --- RULE 2: AGGREGATES ---
    if "{" in code and "}" in code and "{{" not in code:
        candidates.append(code.replace("{", "{{").replace("}", "}}")) # Double braces
    if ":" in code and "{" in code:
        candidates.append(code.replace(":", "|")) # Pipe instead of colon

    # --- RULE 3: VARIABLE SAFETY ---
    if ":-" in code:
        parts = code.split(":-")
        head = parts[0].strip()
        # Removing the body usually makes head variables unsafe
        if re.search(r'\b[A-Z][a-zA-Z0-9_]*\b', head):
            candidates.append(head + ".")

    # --- RULE 4: COMPARISONS & ARITHMETIC ---
    if "!=" in code:
        candidates.append(code.replace("!=", "\\=")) # Prolog style
        candidates.append(code.replace("!=", "<>")) # SQL style
    if " = " in code:
        candidates.append(code.replace(" = ", " == ")) # Python style
        candidates.append(code.replace(" = ", " := ")) # Pascal style
    
    # (Matches Example E: "Day mod 14")
    if "\\" in code: 
        candidates.append(code.replace("\\", " mod ")) 

    # --- RULE 6: NEGATION ---
    if " not " in code:
        candidates.append(code.replace(" not ", " ! "))
        candidates.append(code.replace(" not ", " ~ "))

    # --- RULE 7: MARKDOWN ---
    # (Matches Example C)
    candidates.append(f"```clingo\n{code}\n```")

    # --- RULE 8: OPERATORS & LOGIC ---
    if ":-" in code:
        candidates.append(code.replace(":-", " if "))
        candidates.append(code.replace(":-", " : "))
        
        # "or" in body (Example F)
        head, body = code.split(":-", 1)
        if ", " in body:
            candidates.append(head + ":-" + body.replace(", ", " or ", 1))

    if ", " in code:
        candidates.append(code.replace(", ", " and ", 1))

    # --- RULE 9: COMMENTS & DIRECTIVES ---
    if "%" in code:
        candidates.append(code.replace("%", "//")) # C-style comments

    for directive in ["minimize", "maximize", "show", "const"]:
        target = f"#{directive}"
        if target in code:
            candidates.append(code.replace(target, directive)) # Missing #

    # --- VALIDATION LOOP ---
    final_corruptions = []
    
    for broken_code in candidates:
        actual_error = get_clingo_error(broken_code, context=val_context)
        
        # Only include if it actually breaks Clingo
        if actual_error:
            final_corruptions.append((broken_code, actual_error))

    return final_corruptions

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
    
    print(f"  [Extraction] Querying Gemini for structure...")
    current_tone = random.choice(TONES)

    try:
        # 1. Format String
        prompt_content = EXTRACTION_PROMPT.format(
            nl_content=nl_text, 
            asp_content=asp_text, 
            selected_tone=current_tone
        )
        
        # 2. Call API
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt_content], 
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=1
            )
        )
        print("Gemini CALL SUCCESS")
        # 3. Clean JSON
        cleaned_json = clean_gemini_code(response.text)
        data = json.loads(cleaned_json)
        
    except json.JSONDecodeError as je:
        print(f"  [Error] JSON Parse Failed.")
        return None, False
    except Exception as e:
        print(f"  [Error] API Call Failed: {e}")
        return None, False

    # 3. Auto-Repair Invalid Code (Distillation)
    if ENABLE_AUTO_REPAIR:
        modified = False

        # --- REPAIR 1: INSTANCE TEMPLATE (Added) ---
        inst_code = data.get("instance_template", "")
        inst_error = get_clingo_error(inst_code)
        
        if inst_error:
            print(f"    [Repairing] Instance Template. Error: {inst_error.splitlines()[0]}...")
            fixed = repair_with_gemini(inst_code, "Create the instance template (define predicates and domains).", inst_error)

            if is_valid_clingo(fixed):
                data["instance_template"] = fixed
                modified = True
            else:
                print(f"    [Failed] Instance repair failed.")
        
        # --- REPAIR 2: GENERATOR ---
        inst_ctx = data.get("instance_template", "")
        gen_code = data.get("generator", "")
        gen_error = get_clingo_error(gen_code, inst_ctx)
        
        if gen_error:
            print(f"    [Repairing] Generator. Error: {gen_error.splitlines()[0]}...") 
            fixed = repair_with_gemini(gen_code, "Generate the choice rule/generator for this problem.", gen_error, inst_ctx)
            
            if is_valid_clingo(fixed, inst_ctx):
                data["generator"] = fixed
                modified = True
            else:
                print(f"    [Failed] Generator repair failed.")

        val_context = data.get("instance_template", "") + "\n" + data.get("generator", "")

        # --- REPAIR 3: HARD CONSTRAINTS ---
        for hc in data.get("hard_constraints", []):
            rule_code = hc.get("asp_rule", "")
            rule_error = get_clingo_error(rule_code, val_context)
            
            if rule_error:
                print("-"*30)
                print(f"code with error: {rule_code}")
                print(f"    [Repairing] Hard Constraint. Error: {rule_error.splitlines()[0]}...")
                fixed = repair_with_gemini(rule_code, hc.get("description", "Hard Constraint"), rule_error, val_context)
                print(f"Fixed code: {fixed}")
                if is_valid_clingo(fixed, val_context):
                    hc["asp_rule"] = fixed
                    modified = True
                else:
                    print(f"    [Failed] Hard Constraint repair failed.")
        
        # --- REPAIR 4: SOFT CONSTRAINTS ---
        for sc in data.get("soft_constraints", []):
            rule_code = sc.get("asp_rule", "")
            rule_error = get_clingo_error(rule_code, val_context)
            
            if rule_error:
                print("-"*30)
                print(f"rule code: {rule_code}")
                print(f"    [Repairing] Soft Constraint. Error: {rule_error.splitlines()[0]}...")
                fixed = repair_with_gemini(rule_code, sc.get("description", "Soft Constraint"), rule_error, val_context)
                print(f"Fixed code: {fixed}")
                if is_valid_clingo(fixed, val_context):
                    sc["asp_rule"] = fixed
                    modified = True
                else:
                    print(f"    [Failed] Soft Constraint repair failed.")
        
        if modified:
            print(f"    [Success] Data repaired via Distillation.")

    # 4. Save
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print("structure succesfully generated")
    print("-"*30)
    print("\n")

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

        # Context for checking Hard/Soft constraints
        validation_context = inst_temp + "\n" + gen_rule

        def add_example(sys, user, code, context_for_val=""):
            if is_valid_clingo(code, context_for_val):
                # 1. Standard Translation Task
                examples.append({"messages": [{"role": "system", "content": sys}, {"role": "user", "content": user}, {"role": "assistant", "content": code}]})
                
                # 2. Synthetic Repair Task
                if random.random() < REPAIR_TASK_PROB:
                    corruptions = corrupt_asp_code(code, context_for_val)
                    if corruptions:
                        bad_code, err = random.choice(corruptions)
                        # For repair, we put the broken code first, context is implicit or less relevant
                        repair_prompt = f"The following code has an error:\n{bad_code}\n\nClingo Error:\n{err}\n\nFix the code."
                        examples.append({"messages": [{"role": "system", "content": SYS_PROMPT_REPAIR_TASK.format(description=nl_text)}, {"role": "user", "content": repair_prompt}, {"role": "assistant", "content": code}]})

        # --- 1. Instance Generation ---
        # Prompt: Context -> Request
        inst_user_msg = f"Problem Description:\n{nl_text}\n\n### TASK\nCreate the instance template (define predicates and domains)."
        add_example(SYS_PROMPT_INSTANCE, inst_user_msg, inst_temp)

        # --- 2. Generator Generation ---
        gen_user_msg = f"Problem Description:\n{nl_text}\n\n<<instance_template>>\n{inst_temp}\n\n### TASK\nCreate the generator (choice rules)."
        add_example(SYS_PROMPT_GENERATOR, gen_user_msg, gen_rule, inst_temp)

        # --- 3. Hard Constraints ---
        for hc in data.get("hard_constraints", []):
            desc = hc.get('description', 'Constraint')
            user_msg = f"{context_block}\n\n### TARGET CONSTRAINT\nImplement this hard constraint:\n{desc}"
            add_example(SYS_PROMPT_HARD, user_msg, hc.get('asp_rule', ''), validation_context)

        # --- 4. Soft Constraints ---
        for sc in data.get("soft_constraints", []):
            desc = sc.get('description', 'Optimization')
            user_msg = f"{context_block}\n\n### TARGET OPTIMIZATION\nImplement this soft constraint:\n{desc}"
            add_example(SYS_PROMPT_SOFT, user_msg, sc.get('asp_rule', ''), validation_context)
            
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

if __name__ == "__main__":
    import argparse

    # 1. Setup paths relative to the script location
    # SCRIPT_DIR is defined at the top of your file already
    default_data_in = os.path.join(PROJECT_ROOT, "Finetuning", "data", "Benchmark Data")
    default_data_out = os.path.join(PROJECT_ROOT, "Finetuning", "data")

    parser = argparse.ArgumentParser(description="Convert ASP Benchmark data to JSONL for Fine-tuning.")
    parser.add_argument("--input", default=default_data_in, help="Path to Benchmark Data folder")
    parser.add_argument("--output", default=default_data_out, help="Path to save train.jsonl and valid.jsonl")

    args = parser.parse_args()

    # 2. Run the main processing loop
    print(f"🚀 Starting Data Conversion")
    print(f"📂 Input:  {to_relative(args.input)}")
    print(f"📂 Output: {to_relative(args.output)}")
    print("-" * 30)

    main(args.input, args.output)