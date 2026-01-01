import sys
import os
import shutil
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 1. SETUP PATHS & IMPORTS ---

current_script_dir = os.path.dirname(os.path.abspath(__file__))
finetuning_dir = os.path.dirname(current_script_dir)
sys.path.append(finetuning_dir)

try:
    from utils_ft import get_root_path
except ImportError:
    print("Warning: utils_ft not found. Using local path calculation.")
    def get_root_path():
        return os.path.abspath(os.path.join(finetuning_dir, '..'))

# --- 2. CONFIGURATION ---

PROJECT_ROOT = get_root_path()
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')
DATA_ROOT = os.path.join(PROJECT_ROOT, "Finetuning", "data", "Benchmark Data")

if os.path.exists(ENV_PATH):
    # print(f"Loading .env from: {ENV_PATH}") # Optional: comment out to reduce noise
    load_dotenv(ENV_PATH)

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY not found in .env file.")

client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-3-flash-preview"

# --- 3. PROMPT STYLES ---

PROMPTS = {
    "natural": (
        "You are a creative writer. Rewrite the following logic problem description as a natural, "
        "flowing narrative or scenario. Use full sentences and paragraphs. Avoid bullet points, "
        "lists, or strict constraint formats. Describe the situation as if explaining it to a "
        "colleague in plain English, but strictly preserve all logical rules, numbers, and constraints."
    ),
    "instructional": (
        "You are a formal systems architect. Rewrite the following problem description into a "
        "precise, structured specification. Use clear imperative language, bullet points for "
        "constraints, and formal terminology. Strictly preserve all logic and numbers."
    ),
    "academic": (
        "You are a computer science professor. Rewrite the following problem description as a "
        "formal textbook exercise. Use academic tone, precise vocabulary (e.g., 'entities', "
        "'conditions', 'subject to constraints'), and maintain strict logical equivalence."
    )
}

# --- 4. CORE LOGIC ---

def paraphrase_file(folder_path, style="natural"):
    """
    Augments the data using the specified style.
    Skips generation if a variant of this style already exists.
    """
    if style not in PROMPTS:
        print(f"Error: Unknown style '{style}'. Available: {list(PROMPTS.keys())}")
        return

    nl_path = os.path.join(folder_path, "NL.txt")
    asp_path = os.path.join(folder_path, "ASP.txt")

    if not os.path.exists(nl_path):
        return

    # --- CHECK FOR EXISTING VARIANT ---
    base_name = os.path.basename(folder_path)
    parent_dir = os.path.dirname(folder_path)
    
    # We look for any folder that starts with the specific pattern for this style
    # e.g., "Problem1_variant_gemini_natural"
    prefix_to_check = f"{base_name}_variant_gemini_{style}"
    
    existing_variants = [
        d for d in os.listdir(parent_dir) 
        if d.startswith(prefix_to_check) and os.path.isdir(os.path.join(parent_dir, d))
    ]

    if existing_variants:
        # Found an existing folder for this style -> SKIP
        # print(f"Skipping {base_name}: Style '{style}' already exists.") 
        return

    # --- GENERATION ---
    
    with open(nl_path, "r", encoding="utf-8") as f:
        nl_content = f.read().strip()
    
    # Define Output Path (Always use suffix _1 since we ensure uniqueness above)
    variant_name = f"{base_name}_variant_gemini_{style}_1"
    new_folder = os.path.join(parent_dir, variant_name)
    
    print(f"Generating [{style}]: {base_name} -> {variant_name}...")

    try:
        # Call Gemini
        response = client.models.generate_content(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                temperature=0.85, 
                system_instruction=PROMPTS[style]
            ),
            contents=[nl_content]
        )
        
        new_nl = response.text

        # Save
        os.makedirs(new_folder, exist_ok=True)
        
        with open(os.path.join(new_folder, "NL.txt"), "w", encoding="utf-8") as f:
            f.write(new_nl)
        
        if os.path.exists(asp_path):
            shutil.copy(asp_path, os.path.join(new_folder, "ASP.txt"))
            print(f" -> Success.")
        else:
            print(f" -> Warning: ASP.txt missing, copied only NL.")

    except Exception as e:
        print(f" -> Failed: {e}")