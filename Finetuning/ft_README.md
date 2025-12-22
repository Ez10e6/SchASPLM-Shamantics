# Fine-tuning LLMs for Answer Set Programming

This directory contains the pipeline for Supervised Fine-Tuning (SFT) of Large Language Models (LLMs) to generate Answer Set Programming (ASP) code from natural language descriptions.

---

## Approach

The pipeline utilizes **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA (Low-Rank Adaptation)** to adapt base models (such as Meta-Llama-3 or Qwen-2.5) to the rigid syntax and logic of ASP.

### 1. Dual-Backend Architecture
The framework supports two distinct training paths based on available hardware:
*   **PyTorch Backend (NVIDIA CUDA / Generic):** Uses Hugging Face `trl` (SFTTrainer) and `peft`. Models are loaded in `bfloat16` precision. LoRA is applied to the top 16 transformer layers to balance adaptation quality with memory efficiency.
*   **MLX Backend (Apple Silicon):** Uses the Apple MLX framework for high-performance training on macOS. Training is handled via CLI wrappers and dynamic YAML configuration generation.

### 2. Data Processing Pipeline
The training data is ingested via `data_converter.py`. This script parses raw text triplets (Natural Language, Intermediate Representation, and the target ASP code) from the benchmark folders and formats them into a standardized **ChatML JSONL** format.

### 3. Model Fusion
The pipeline concludes with a fusion step. Trained LoRA adapter weights are merged directly into the base model weights, creating a standalone "fused" model that can be loaded for inference without requiring separate adapter files.

---

## How to Use

### 1. Setup
Install the specific fine-tuning dependencies:
```bash
pip install -r requirements_ft.txt
```
Ensure your `.env` file in the project root contains a valid `HF_KEY` if using gated models (like Llama-3).

### 2. Prepare the Dataset
1.  Place your raw problem files in `Finetuning/data/Benchmark Data/`. Each problem folder should contain `NL.txt`, `IR.txt`, and `ASP.txt`.
2.  Open and run `notebooks/data_converter.ipynb`. 
3.  This will generate `train.jsonl` and `valid.jsonl` in the `Finetuning/data/` folder.

### 3. Run Fine-tuning

#### Option A: Using PyTorch (NVIDIA GPU / Windows / Linux)
1.  Open `notebooks/ft_pytorch_qlora.ipynb`.
2.  Set the `MODEL_TYPE` (e.g., "llama" or "qwen").
3.  Execute the cells to initialize the `SFTTrainer`. The script will train for the specified iterations, save the adapters to `Finetuning/adapters/`, and automatically save a fused version of the model to `local_models/`.

#### Option B: Using MLX (Apple Silicon macOS)
1.  Open `notebooks/ft_mlx_lora.ipynb`.
2.  Set your training parameters (iterations, learning rate).
3.  Execute the cells to run the MLX CLI training. 
4.  Run the final "Fusion" cell to merge the adapters into a new model directory in `local_models/`.

### 4. Inference with the Fine-tuned Model
Once training and fusion are complete, you can use the new model in the main application:
1.  Go to the main `1_ASP_Scheduler.ipynb` notebook.
2.  Update the `CHECKPOINT` path to point to your new fused model (e.g., `./local_models/ft_llama_pytorch`).
3.  Run the scheduler to generate ASP programs using your specialized model.

---

## Folder Structure

*   **`data/`**: Contains the generated `train.jsonl` and `valid.jsonl` files.
*   **`adapters/`**: Storage for LoRA weight checkpoints during and after training.
*   **`notebooks/`**: Interactive jupyter notebooks for data conversion and training.
*   **`scripts/`**: Python implementations for the PyTorch and MLX training logic.
*   **`utils_ft.py`**: Shared utility functions for pathing and environment management.

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