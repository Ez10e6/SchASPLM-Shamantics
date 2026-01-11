import sys
import os
import subprocess
import yaml
import shutil
import tempfile
from utils_ft import to_relative, get_project_root

def run_mlx_training(model_path, data_folder, adapter_path, iters=1000, num_layers=16):
    """
    Runs MLX LoRA training via CLI.
    """
    cwd = os.getcwd()

    # Maak paden relatief t.o.v. waar je het script draait.
    # Dit zorgt ervoor dat de interne logs van MLX er schoon uitzien.
    rel_model_path = os.path.relpath(model_path, cwd)
    rel_data_folder = os.path.relpath(data_folder, cwd)
    rel_adapter_path = os.path.relpath(adapter_path, cwd)

    print("--- Initializing MLX Fine-tuning (CLI Mode) ---")
    print(f"--- Model: ./{rel_model_path} ---")
    print(f"--- Data Folder: ./{rel_data_folder} ---")

    os.makedirs(rel_adapter_path, exist_ok=True)

    # This ratio needs to be correct
    grad_accumulation_steps=4
    decay_steps=iters//grad_accumulation_steps
    
    # Configuration
    config_data = {
        "model": rel_model_path,
        "train": True,
        "data": rel_data_folder,
        "fine_tune_type": "lora",
        "adapter_path": rel_adapter_path,
        
        # Hyperparameters
        "iters": iters,
        "batch_size": 1,
        "grad_accumulation_steps": grad_accumulation_steps,  
        "grad_checkpoint": True,       
        "max_seq_length": 4096,        
        
        # Optimizer
        "optimizer": "adamw",
        "lr_schedule": {
            "name": "cosine_decay",  
            "arguments": [1e-5, decay_steps],
        },
        
        # Logging & Saving
        "steps_per_report": 20,
        "steps_per_eval": 100,
        "val_batches": -1,
        "save_every": 100,
        "seed": 42,
        
        # LoRA Config
        "num_layers": num_layers, 
        # "lora_parameters": {
        #     "rank": 16,
        #     "dropout": 0.1,
        #     "scale": 2.0
        # },

        "mask_prompt": True
    }

    rel_config_path = os.path.join(rel_adapter_path, "train_config.yaml")
    print(f"Generating configuration file at: ./{rel_config_path}")
    with open(rel_config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", rel_config_path]

    print("Executing MLX CLI training...")
    try:
        subprocess.run(cmd, check=True)
        print("✓ Training completed.")
    except subprocess.CalledProcessError as e:
        print(f"Training Failed with exit code {e.returncode}")
        raise

def fuse_model(base_model, adapter_path, save_path, checkpoint_step=None):
    """
    Fuses adapters into the base model.
    """
    print(f"Fusing LoRA adapters (Step: {checkpoint_step if checkpoint_step else 'Final'})...")
    
    # Paths for display
    rel_save_path = to_relative(save_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        
        if checkpoint_step:
            src_weights = os.path.join(adapter_path, f"{int(checkpoint_step):07d}_adapters.safetensors")
            if not os.path.exists(src_weights):
                raise FileNotFoundError(f"Checkpoint not found: {src_weights}")
        else:
            src_weights = os.path.join(adapter_path, "adapters.safetensors")

        # Destination must be 'adapters.safetensors' for MLX fuse command
        dest_weights = os.path.join(temp_dir, "adapters.safetensors")
        
        print(f"--- Staging checkpoint from: {os.path.basename(src_weights)} ---")
        shutil.copy2(src_weights, dest_weights)
        
        # Copy config files
        for file in ["adapter_config.json", "train_config.yaml"]:
            src = os.path.join(adapter_path, file)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(temp_dir, file))

        # Use relative paths for the fuse command as well
        # Note: temp_dir is absolute, which is fine
        cmd = [
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model", base_model,
            "--adapter-path", temp_dir,
            "--save-path", rel_save_path,
        ]
        
        try:
            # Run from Root so relative model/save paths work
            subprocess.run(cmd, check=True, cwd=get_project_root())
            print(f"Success! Fused model saved to: ./{rel_save_path}")
        except subprocess.CalledProcessError as e:
            print(f"Fusing Failed with exit code {e.returncode}")
            raise