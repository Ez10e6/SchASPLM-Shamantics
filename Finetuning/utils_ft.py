import os
from dotenv import load_dotenv

def get_root_path():
    # Returns the absolute path to the SchASPLM directory
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def to_relative(path):
    """
    Converts an absolute path to a relative path based on the project root.
    Used for secure logging/printing in notebooks.
    """
    try:
        return os.path.relpath(path, get_root_path())
    except ValueError:
        return path # Fallback if path is on a different drive (Windows)
    
def setup_env():
    root = get_root_path()
    load_dotenv(os.path.join(root, '.env'))
    return os.getenv('HF_KEY')

def get_model_path(model_type="llama"):
    root = get_root_path()
    local_dir = os.path.join(root, 'local_models')
    
    # Mapping based on your provided folder structure
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
    root = get_root_path()
    return os.path.join(root, 'local_models', f"ft_{model_type}_{framework}")