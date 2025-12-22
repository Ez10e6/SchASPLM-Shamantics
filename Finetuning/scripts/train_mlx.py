import sys
import os
import subprocess
import yaml
from utils_ft import to_relative

def run_mlx_training(model_path, data_folder, adapter_path, iters=300):
    """
    Runs MLX LoRA training by generating a YAML config and calling the CLI.
    Since mlx_lm.lora uses trust_remote_code=true when loading the tokenizer
    it is important to only run this with trusted models.
    """
    print("--- Initializing MLX Fine-tuning (CLI Mode) ---")
    print(f"--- Model: ./{to_relative(model_path)} ---")
    print(f"--- Data Folder: ./{to_relative(data_folder)} ---")

    # Ensure adapter directory exists
    os.makedirs(adapter_path, exist_ok=True)
    
    # 1. Construct the Configuration Dictionary
    config_data = {
        "model": model_path,
        "train": True,
        "data": data_folder,
        "fine_tune_type": "lora",
        "adapter_path": adapter_path,
        
        # Training Hyperparameters
        "iters": iters,
        "batch_size": 1,
        "grad_accumulation_steps": 4,  
        "grad_checkpoint": True,       
        "max_seq_length": 2048,
        
        # Optimization
        "optimizer": "adamw",
        "lr_schedule": {
            "name": "cosine_decay",  
            "arguments": [5e-5, iters]
        },
        
        # Logging & Saving
        "steps_per_report": 10,
        "steps_per_eval": 20,
        "val_batches": 5,
        "save_every": 100,
        "seed": 42,
        
        # LoRA Architecture
        # num_layers: 16 fine-tunes the top 16 layers (memory efficient).
        "num_layers": 16, 
        
        "lora_parameters": {
            "rank": 16,
            "dropout": 0.05,
            "scale": 10.0
        }
    }

    # 2. Save Config to YAML
    config_path = os.path.join(adapter_path, "train_config.yaml")
    print(f"Generating configuration file at: ./{to_relative(config_path)}")
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    # 3. Build and Execute Command
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", config_path
    ]

    print("Executing MLX CLI training...")
    try:
        # Check=True will raise CalledProcessError if it fails
        subprocess.run(cmd, check=True)
        
        if os.path.exists(os.path.join(adapter_path, "adapters.safetensors")):
            print("✓ Training completed successfully. Adapters saved.")
        else:
            print("! Warning: Training finished but adapters.safetensors not found.")
            
    except subprocess.CalledProcessError as e:
        print(f"Training Failed with exit code {e.returncode}")
        raise

def fuse_model(base_model, adapter_path, save_path):
    """
    Fuses the LoRA adapter weights back into the base model using the official CLI.
    """
    print("Fusing LoRA adapters into base model (CLI Mode)...")
    print(f"--- Base Model: ./{to_relative(base_model)} ---")
    print(f"--- Adapter Path: ./{to_relative(adapter_path)} ---")
    print(f"--- Save Path: ./{to_relative(save_path)} ---")
    
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", base_model,
        "--adapter-path", adapter_path,
        "--save-path", save_path,
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Success! Fused model saved to: ./{to_relative(save_path)}")
    except subprocess.CalledProcessError as e:
        print(f"Fusing Failed with exit code {e.returncode}")
        if e.returncode == 1:
            print("Tip: If fusing failed due to memory, close other apps or try reducing the base model precision.")
        raise