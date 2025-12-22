# Fine-tuning LLMs for Answer Set Programming

This directory contains the pipeline for Supervised Fine-Tuning (SFT) of Large Language Models (LLMs) to generate Answer Set Programming (ASP) code.

## Approach

The pipeline utilizes **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA (Low-Rank Adaptation)** to adapt base models (Meta-Llama-3, Qwen-2.5) using specific ASP synthesis datasets.

### 1. Dual-Backend Architecture
To support different hardware environments, we implement two distinct training paths:

*   **PyTorch (NVIDIA CUDA / Generic):**
    *   Uses Hugging Face `transformers`, `trl` (SFTTrainer), and `peft`.
    *   Loads base models in `bfloat16` precision.
    *   Applies LoRA adapters with high scaling (`alpha=160`, `r=16`) to the top 16 transformer layers to optimize memory usage during training.
    *   Uses the AdamW optimizer with a cosine learning rate scheduler.

*   **MLX (Apple Silicon):**
    *   Uses the `mlx` framework and `mlx-lm` library.
    *   Executes training via the CLI using dynamically generated YAML configuration files.
    *   Supports local fine-tuning and model fusion directly on macOS devices.

### 2. Data Processing Pipeline
The training data is processed using `data_converter.py`, which performs the following:
1.  Ingests raw text files containing Natural Language (NL) or Intermediate Representation (IR) problem descriptions and their corresponding ASP solutions.
2.  Formats these pairs into the standard **ChatML JSONL format**:
    ```json
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    ```
3.  Splits the data into training and validation sets (80/20 split).

### 3. Model Fusion
Post-training, the `fuse_model` functions merge the trained LoRA adapters back into the base model weights. This results in a standalone model that can be loaded without external adapter files for inference.

## Repository Structure

```
Finetuning/
├── data/                   # Processed JSONL datasets (Train/Validation splits)
├── adapters/               # Checkpoints and saved LoRA adapters
├── notebooks/
│   ├── data_converter.ipynb   # Script to convert raw benchmarks to JSONL
│   ├── ft_pytorch_qlora.ipynb # Training notebook for PyTorch/CUDA
│   └── ft_mlx_lora.ipynb      # Training notebook for MLX/macOS
├── scripts/
│   ├── train_pytorch.py       # SFTTrainer implementation using PyTorch
│   ├── train_mlx.py           # Wrapper for mlx_lm.lora CLI
│   └── data_converter.py      # Dataset parsing and formatting logic
└── utils_ft.py                # Environment setup and path management
```