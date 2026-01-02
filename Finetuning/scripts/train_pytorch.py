import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset

def train_pytorch(model_path, data_folder, adapter_path, save_path, hf_token, iters=12000, num_layers=16):
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()
    
    print(f"--- Initializing PyTorch Training ---")
    print(f"--- Device: {'MPS' if is_mps else 'CUDA' if is_cuda else 'CPU'} ---")

    # 1. Unified Model Loading
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        token=hf_token,
        device_map="auto" if is_cuda else None,
        attn_implementation="sdpa" if (is_cuda or is_mps) else "eager"
    )

    if is_mps:
        print("Moving model to MPS...")
        model.to("mps")

    # 2. Dynamic Layer Masking (Top 16 Layers)
    # Aligns with MLX "num_layers: 16"
    num_total_layers = model.config.num_hidden_layers
    target_layers = list(range(num_total_layers - num_layers, num_total_layers))

    # 3. Aligned LoRA Config
    #     # Scale = alpha/r = 512/16 = 32.0 (Matches MLX 'scale: 32.0')
    peft_config = LoraConfig(
        r=16,
        lora_alpha=512,
        layers_to_transform=target_layers,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. Standard Tokenizer & Dataset Setup
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    dataset_train = load_dataset("json", data_files=os.path.join(data_folder, "train.jsonl"), split="train")
    dataset_valid = load_dataset("json", data_files=os.path.join(data_folder, "valid.jsonl"), split="train")
    # Match MLX val_batches=128 (batch_size=1) by limiting eval samples
    if len(dataset_valid) > 128: dataset_valid = dataset_valid.select(range(128))

    # 5. Training Args (Standard AdamW to match MLX)
    sft_config = SFTConfig(
        output_dir=adapter_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_steps=iters//16,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=600,
        save_strategy="steps",
        save_steps=600,
        optim="adamw_torch", 
        bf16=True, 
        report_to="none",
        dataset_text_field="messages",
        max_length=4096,
        neftune_noise_alpha=5,
        seed=42,
        gradient_checkpointing=True,
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
    report_path = os.path.join(adapter_path, "report.json")
    os.makedirs(adapter_path, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    
    
    trainer.model.save_pretrained(adapter_path) 
    tokenizer.save_pretrained(adapter_path)

    # 7. SIMPLE MERGE & SAVE (In-Place)
    print("--- Starting Simplified Merge ---")

    # Finalize Weights
    print("Merging adapters into base model...")
    merged_model = trainer.model.merge_and_unload()

    # Save to the path
    print(f"Saving final model to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"--- Fused model ready at {save_path} ---")
