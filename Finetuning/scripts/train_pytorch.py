import os
import torch
import json
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from utils_ft import to_relative

def train_pytorch(model_id, dataset_dir, output_dir, hf_token):
    """
    Performs LoRA fine-tuning using PyTorch.
    Optimized for CUDA (Windows) and MPS (macOS).
    """
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()
    
    print(f"--- Initializing PyTorch Training ---")
    print(f"--- Device: {'CUDA' if is_cuda else 'MPS' if is_mps else 'CPU'} ---")

    # 1. Hardware-specific Configuration (Quantization)
    bnb_config = None
    if is_cuda:
        # bitsandbytes is only supported on CUDA
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # 2. Load Model
    # Note: device_map="auto" is essential for bitsandbytes on CUDA
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        token=hf_token,
        device_map="auto" if is_cuda else None,
        trust_remote_code=True
    )
    
    # Fallback for Mac (MPS)
    if not is_cuda and is_mps:
        model.to("mps")

    # 3. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fixed for specific Llama-3/Qwen behaviors

    # 4. Define LoRA (PEFT) Config
    # target_modules covers the linear layers in both Llama and Qwen architectures
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 5. Load Datasets (Train and Valid)
    dataset_train = load_dataset("json", data_files=os.path.join(dataset_dir, "train.jsonl"), split="train")
    dataset_valid = load_dataset("json", data_files=os.path.join(dataset_dir, "valid.jsonl"), split="train")

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir="./tmp_pytorch_results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        evaluation_strategy="epoch",  # Run validation at the end of every epoch
        save_strategy="no",           # Don't save checkpoints, we merge at the end
        # Use paged optimizer for CUDA to save VRAM; standard for Mac
        optim="paged_adamw_32bit" if is_cuda else "adamw_torch",
        report_to="none"
    )

    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=1024,
        dataset_text_field="messages", # Required for the ChatML format we created
    )

    # 8. Execute Training
    print("Starting training process...")
    train_result = trainer.train()
    
    # 9. Save standardized report for the dashboard
    report_path = os.path.join(output_dir, "report.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    
    # 10. Merge and Finalize
    print(f"Saving merged model to: ./{to_relative(output_dir)}")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)