import subprocess
import sys
import os
import time
from utils_ft import to_relative

def run_mlx_training(model_path, data_folder, adapter_path, iters=300):
    """
    Runs MLX-native LoRA training using the stable CLI via subprocess.
    This method is preferred as it automatically generates report.json
    and handles complex object initialization internally.
    """
    print("--- Initializing MLX Fine-tuning ---")
    print(f"--- Model: ./{to_relative(model_path)} ---")
    print(f"--- Data Folder: ./{to_relative(data_folder)} ---")

    # Ensure adapter directory exists
    os.makedirs(adapter_path, exist_ok=True)
    
    # Standard MLX CLI command
    # Note: MLX looks for train.jsonl and valid.jsonl inside the --data folder
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", data_folder,
        "--iters", str(iters),
        "--batch-size", "1",
        "--adapter-path", adapter_path,
        "--learning-rate", "5e-5",
        "--steps-per-report", "10",  # Log training loss to terminal/JSON every 10 iters
        "--steps-per-eval", "10"      # Run validation every 10 iters to populate report.json
    ]
    
    try:
        print("Executing MLX LoRA training command...")
        subprocess.run(cmd, check=True)
        
        # Give the OS a moment to flush the file to disk
        print("Finalizing logs...")
        time.sleep(2)
        
        report_path = os.path.join(adapter_path, "report.json")
        if os.path.exists(report_path):
            print(f"✓ Success: report.json generated at ./{to_relative(report_path)}")
        else:
            print("! Error: report.json was still not generated. Checking for training_log.json...")
            # Some versions use this alternative name
            alt_path = os.path.join(adapter_path, "training_log.json")
            if os.path.exists(alt_path):
                os.rename(alt_path, report_path)
                print(f"✓ Found alternative log and renamed it to report.json")

    except subprocess.CalledProcessError as e:
        print(f"MLX Training Failed: {e}")
        raise

def fuse_model(base_model, adapter_path, save_path):
    """
    Fuses the LoRA adapter weights back into the base model.
    """
    print("Fusing LoRA adapters into base model...")
    print(f"--- Target Path: ./{to_relative(save_path)} ---")
    
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", base_model,
        "--adapter-path", adapter_path,
        "--save-path", save_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Success! Fused model saved to: ./{to_relative(save_path)}")
    except subprocess.CalledProcessError as e:
        print(f"Fusing Failed: {e}")
        raise