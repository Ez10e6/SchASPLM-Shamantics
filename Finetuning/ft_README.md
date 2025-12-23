# Fine-tuning LLMs for Answer Set Programming

This directory contains the pipeline for Supervised Fine-Tuning (SFT) of Large Language Models (LLMs) to generate Answer Set Programming (ASP) code from natural language descriptions.

---

## Approach & Training Configuration

The pipeline utilizes **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA (Low-Rank Adaptation)**. This allows the model to learn the specific syntax and logical structures of ASP without the massive computational overhead of full-parameter fine-tuning.

### 1. LoRA Configuration
*   **Task Type (`CAUSAL_LM`):** The PyTorch backend specifically uses the `CAUSAL_LM` task type.
    *   *Reason:* This is required for autoregressive models (like Llama and Qwen) to ensure the LoRA adapters are correctly integrated into the language modeling head for next-token prediction.
*   **Rank ($r=16$) and Alpha ($\alpha=160$):** We use a rank of 16 to provide sufficient capacity for logic rules. The alpha is set to 160 to achieve a high scaling factor ($\frac{\alpha}{r} = 10$).
    *   *Reason:* A high scaling factor ensures that the learned ASP weights have a significant impact on the model's final output, "forcing" the model to prioritize the rigid ASP structure over its default conversational style.
*   **Dropout ($0.05$):** A dropout rate of 5% is applied to the LoRA layers.
    *   *Reason:* Since the benchmark dataset is relatively small, dropout is essential to prevent the model from simply memorizing the training examples, encouraging it to generalize the underlying principles of ASP syntax instead.
*   **Selective Layer Targeting:** We tune only the **top 16 transformer layers**.
    *   *Reason:* Lower layers typically handle general language understanding, while higher layers manage complex reasoning and specific output formats. Targeting the top layers focuses the training on ASP syntax mapping while preserving foundational linguistic abilities and reducing VRAM usage.

### 2. Optimization & Precision
*   **Optimizer (AdamW):** We use AdamW with a **Cosine Learning Rate Scheduler**.
    *   *Reason:* AdamW provides effective weight decay for transformers. The cosine scheduler starts with a higher learning rate to explore the solution space and gradually decays to zero, allowing the model to "settle" into the specific ASP syntax without overshooting.
*   **Precision (bfloat16):** Models are loaded and trained in `bfloat16`.
    *   *Reason:* `bf16` offers a better dynamic range than standard `float16`, preventing gradient overflows and ensuring stable convergence on logic-heavy code datasets.
*   **Gradients:** Training uses a micro-batch size of 1 with **Gradient Accumulation** set to 4.
    *   *Reason:* This setup simulates a stable batch size of 4, providing necessary gradient stability for small datasets while staying within the memory limits of consumer-grade GPUs.

### 3. Data Processing
The `data_converter.py` script transforms raw benchmark data into the **ChatML JSONL** format.
*   **Format:** Each entry includes a "system" role (persona of an ASP expert), a "user" role (the NL/IR problem), and an "assistant" role (the gold-standard ASP code).
*   **Splitting:** Data is automatically split 80/20 into training and validation sets.

---

## How to Use

### 1. Setup
Install the specific fine-tuning dependencies:
```bash
pip install -r requirements_ft.txt
```
Ensure your `.env` file in the project root contains a valid `HF_KEY` if using gated models (like Llama-3).

### 2. Prepare the Dataset
1.  Place your raw problem files in `Finetuning/data/Benchmark Data/`. Each problem folder should contain `NL.txt` OR `IR.txt`, and `ASP.txt`.
2.  Open and run `notebooks/data_converter.ipynb`. 
3.  This will generate `train.jsonl` and `valid.jsonl` in the `Finetuning/data/` folder.

### 3. Run Fine-tuning

#### Option A: Using PyTorch (NVIDIA GPU / Windows / Linux)
1.  Open `notebooks/ft_pytorch_qlora.ipynb`.
2.  Set the `MODEL_TYPE` (e.g., "llama" or "qwen").
3.  Set the amount of iterations.
3.  Execute the training cell to start training. The script will train for the specified iterations, save the adapters to `Finetuning/adapters/`, and automatically save a fused version of the model to `local_models/`.

#### Option B: Using MLX (Apple Silicon macOS)
1.  Open `notebooks/ft_mlx_lora.ipynb`.
2.  Set the `MODEL_TYPE` (e.g., "llama" or "qwen").
3.  Set the amount of iterations.
4.  Execute the cells to run the MLX CLI training. 
5.  Run the final "Fusion" cell to merge the adapters into a new model directory in `local_models/`.

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