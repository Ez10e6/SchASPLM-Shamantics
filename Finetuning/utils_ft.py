import os
import sys
from dotenv import load_dotenv

def get_finetuning_root():
    """Returns the absolute path to the 'Finetuning' directory."""
    # Assuming this file is located at Finetuning/utils_ft.py
    return os.path.dirname(os.path.abspath(__file__))

def get_project_root():
    """Returns the absolute path to the repository root (one level up from Finetuning)."""
    return os.path.dirname(get_finetuning_root())

def to_relative(path, start=None):
    """
    Converts an absolute path to a relative path for cleaner logging/display.
    Defaults to relative to the project root.
    """
    if start is None:
        start = get_project_root()
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path

def setup_env():
    """Loads environment variables from .env in the project root."""
    root = get_project_root()
    env_path = os.path.join(root, '.env')
    load_dotenv(env_path)
    return os.getenv('HF_KEY')

# --- Path Helpers ---

def get_benchmark_data_path():
    """Returns path to: Finetuning/data/Benchmark Data"""
    return os.path.join(get_finetuning_root(), "data", "Benchmark Data")

def get_processed_data_path():
    """Returns path to: Finetuning/data"""
    return os.path.join(get_finetuning_root(), "data")

def get_scripts_path():
    """Returns path to: Finetuning/scripts"""
    return os.path.join(get_finetuning_root(), "scripts")

def get_adapters_path(model_type, framework):
    """Returns path to: Finetuning/adapters/{model_type}_{framework}"""
    return os.path.join(get_finetuning_root(), "adapters", f"{model_type}_{framework}")

# --- Model Paths (Preserving existing logic but making it robust) ---

def get_model_path(model_type="llama"):
    root = get_project_root()
    # Assuming models are stored in the root's local_models folder
    local_dir = os.path.join(root, 'local_models')
    
    # Mapping based on your folder structure
    registry = {
        "llama": {
            "local": os.path.join(local_dir, "meta-llama", "Meta-Llama-3-8B-Instruct-bfloat16"),
            "hub": "meta-llama/Meta-Llama-3-8B-Instruct"
        },
        "qwen": {
            "local": os.path.join(local_dir, "Qwen", "Qwen2.5-7B-Instruct-bfloat16"),
            "hub": "Qwen/Qwen2.5-7B-Instruct"
        }
    }
    
    paths = registry.get(model_type.lower())
    if paths and os.path.exists(paths["local"]):
        return paths["local"]
    return paths["hub"] if paths else None

def get_output_dir(model_type, framework):
    root = get_project_root()
    return os.path.join(root, 'local_models', f"ft_{model_type}_{framework}")