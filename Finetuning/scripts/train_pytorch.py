import os
import torch
import json
import gc
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from utils_ft import to_relative

def plot_loss(report_path, output_dir):
    """Generates a loss plot from the trainer log history."""
    with open(report_path, "r") as f:
        history = json.load(f)
    
    train_loss = [x["loss"] for x in history if "loss" in x]
    train_steps = [x["step"] for x in history if "loss" in x]
    eval_loss = [x["eval_loss"] for x in history if "eval_loss" in x]
    eval_steps = [x["step"] for x in history if "eval_loss" in x]

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label="Training Loss", color="blue")
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="Validation Loss", color="red", marker="o")
    
    plt.title("Training and Validation Loss (MLX-Aligned)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to: {plot_path}")

def train_pytorch(model_id, dataset_dir, output_dir, hf_token):
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()
    
    print(f"--- Initializing PyTorch Training (MLX-Aligned) ---")
    print(f"--- Device: {'MPS' if is_mps else 'CUDA' if is_cuda else 'CPU'} ---")

    # 1. Unified Model Loading
    # Integration: 'device' is set during init to avoid redundant CPU-to-device transfers.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, # Match MLX native bf16 training
        token=hf_token,
        device="mps" if is_mps else None, # Integrated MPS loading
        device_map="auto" if is_cuda else None,
        attn_implementation="sdpa" if is_cuda else "eager"
    )

    # 2. Dynamic Layer Masking (Top 16 Layers)
    # Aligns with MLX "num_layers: 16"
    num_total_layers = model.config.num_hidden_layers
    target_layers = list(range(num_total_layers - 16, num_total_layers))

    # 3. Aligned LoRA Config
    # Scale = alpha/r = 160/16 = 10.0 (Matches MLX 'scale: 10')
    peft_config = LoraConfig(
        r=16,
        lora_alpha=160, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # MLX default
        layers_to_transform=target_layers,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. Standard Tokenizer & Dataset Setup
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 
    dataset_train = load_dataset("json", data_files=os.path.join(dataset_dir, "train.jsonl"), split="train")
    dataset_valid = load_dataset("json", data_files=os.path.join(dataset_dir, "valid.jsonl"), split="train")

    # 5. Training Args (Standard AdamW to match MLX)
    sft_config = SFTConfig(
        output_dir="./tmp_pytorch_results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=300,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=100,
        optim="adamw_torch", 
        bf16=True, 
        report_to="none",
        dataset_text_field="messages",
        max_length=2048,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        peft_config=peft_config,
        processing_class=tokenizer, 
        args=sft_config
    )

    print(f"Starting training on {len(target_layers)} layers...")
    trainer.train()
    
    # 6. Save Logs & Visualizations
    report_path = os.path.join(output_dir, "report.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    
    plot_loss(report_path, output_dir)
    
    trainer.model.save_pretrained(output_dir) 
    tokenizer.save_pretrained(output_dir)

    # 7. CLEANUP & MERGE
    print("--- Starting Merge Process ---")
    del model, trainer
    if is_cuda: torch.cuda.empty_cache()
    if is_mps: torch.mps.empty_cache()
    gc.collect()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device="mps" if is_mps else None, # Integrated MPS reload
        device_map="auto" if is_cuda else None,
        token=hf_token
    )
    
    model_to_merge = PeftModel.from_pretrained(base_model, output_dir)
    merged_model = model_to_merge.merge_and_unload()
    
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("Merge complete.")