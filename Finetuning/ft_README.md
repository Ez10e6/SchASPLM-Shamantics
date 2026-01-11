# ASP-Scheduler Fine-Tuning Pipeline

This directory contains the implementation for fine-tuning Large Language Models (LLMs) to act as specialized **ASP Knowledge Engineers**. The pipeline is designed to transform Natural Language (NL) descriptions into syntactically valid and semantically correct Answer Set Programming (ASP) code.

## 1. Overview: Modular Distillation
Rather than training a model to generate a full program at once, we fine-tune models to handle the modular components used by the `ASP_Scheduler`:
*   **Instance Templates:** Mapping NL objects to ASP predicates.
*   **Generators:** Defining search spaces (Choice rules).
*   **Constraints:** Implementing logic for Hard (Integrity) and Soft (Penalty) constraints.
*   **Syntax Repair:** Correcting erroneous code based on compiler feedback.

## 2. Data Pipeline (`scripts/data_converter.py`)
The training data is generated using a "Teacher-Student" distillation approach with **Gemini-1.5-Flash** as the teacher and the **Clingo Compiler** as the ground-truth validator.

### Distillation Process
1.  **Extraction:** Gemini deconstructs existing ASP benchmarks into structured JSON components (Instance, Generator, Constraints).
2.  **Tone Variation:** To improve generalization, the NL descriptions are rewritten into five distinct tones: *Academic, Casual, Business, Succinct,* and *Didactic*.
3.  **Auto-Repair:** Every extracted ASP snippet is passed through `clingo`. If the teacher (Gemini) generates invalid code, it is sent back for a "Distillation Repair" until it passes syntax checks.

### Synthetic Corruption (The "Repair Task")
To support the Scheduler's self-healing capabilities, we generate "Broken -> Fixed" training pairs:
*   **Corruption:** We programmatically break valid ASP code by removing periods, using double braces `{{ }}`, creating unsafe variables, or using non-ASCII characters.
*   **Feedback Integration:** We pair the broken code with the actual Clingo error message.
*   **Task:** The model is trained to output the original valid code when presented with the error message and the broken snippet.

## 3. Training Framework (`scripts/train_mlx.py`)
We utilize the **Apple MLX** framework for efficient LoRA (Low-Rank Adaptation) on Apple Silicon.

### Training Specifications
*   **Base Models:** `Qwen2.5-7B-Instruct` and `Llama-3-8B-Instruct`.
*   **Method:** LoRA (Low-Rank Adaptation).
*   **Hyperparameters:**
    *   **Rank:** 16 (default) / 32 (recommended for logic).
    *   **Learning Rate:** $1e-5$ with Cosine Decay.
    *   **Batch Size:** 1 with Gradient Accumulation (4 steps).
    *   **Max Seq Length:** 4096 tokens.
*   **Masking:** Prompt masking is enabled to ensure the model only learns from the ASP completions, not the user prompts.

## 4. Usage

### Step 1: Data Preparation
Run the data converter to process the raw benchmarks into JSONL format.
```bash
python -m Finetuning.scripts.data_converter --input "data/Benchmark Data" --output "data"
```

### Step 2: Training
Use the provided notebooks or the training script to begin fine-tuning.
*   `notebooks/data_converter.ipynb`: Visualizing the extraction and corruption process.
*   `notebooks/ft_mlx_lora.ipynb`: Executing the training run and monitoring loss.

### Step 3: Model Fusing
Once training is complete, fuse the LoRA adapters into the base model for deployment.
```python
from Finetuning.scripts.train_mlx import fuse_model
fuse_model(base_model, adapter_path, save_path)
```

## 5. Directory Structure
*   `adapters/`: Stores LoRA weights and training configurations.
*   `data/`: Contains `train.jsonl` and `valid.jsonl` files.
*   `prompts/`: System prompts used for data extraction and task-specific fine-tuning.
*   `scripts/`: Core logic for conversion and training.
*   `utils_ft.py`: Shared utilities for path management and environment setup.
