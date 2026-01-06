# LLM Fine-Tuning for ASP Generation

This directory contains the pipeline for fine-tuning Large Language Models (LLMs) to generate Answer Set Programming (ASP) code from Natural Language descriptions. The pipeline leverages Apple MLX for efficient LoRA fine-tuning on Apple Silicon, and Google Gemini for structural distillation.

## Project Structure

```text
Finetuning/
├── adapters/              # Stores LoRA adapters during/after training
├── data/
│   ├── Benchmark Data/    # Raw source problems (folders with NL.txt & ASP.txt)
│   ├── train.jsonl        # Processed training dataset (Chat format)
│   └── valid.jsonl        # Processed validation dataset
├── local_models/          # Location for fused models (base + adapters)
├── notebooks/             # Jupyter notebooks for interactive execution
│   ├── data_converter.ipynb
│   └── ft_mlx_lora.ipynb
├── prompts/               # System prompts used for data distillation
│   ├── extraction.txt     # Extracts JSON structure from NL/ASP pairs
│   ├── generator.txt      # Prompt for generating choice rules
│   ├── hard_constraint.txt
│   ├── repair_gemini.txt  # Auto-repair logic prompt
│   └── ...
├── scripts/               # Core logic scripts
│   ├── data_converter.py  # Converts raw data to training format
│   ├── train_mlx.py       # MLX training wrapper
├── requirements_ft.txt    # Python dependencies
└── utils_ft.py            # Path and environment helpers
```

## Prerequisites

### Hardware
This pipeline is optimized for Apple Silicon (M1/M2/M3) using the `mlx-lm` library.

### Environment Setup
1. **Install Dependencies:**
   ```bash
   pip install -r requirements_ft.txt
   ```

2. **Environment Variables:**
   Create a `.env` file in the project root containing:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here  # Required for data processing/distillation
   HF_KEY=your_huggingface_token            # Required for downloading Llama/Qwen models
   ```

## How to Use

Follow these steps to go from raw data to a fully fine-tuned ASP generation model.

### 1. Prepare the Data
Before training, you must process the benchmark data into the JSONL format required by MLX.

1. Open `notebooks/data_converter.ipynb`.
2. Run all cells in the notebook.
3. This will:
   * Scan the `data/Benchmark Data` folder.
   * Use Google Gemini to distill the ASP code into logical components.
   * Generate synthetic repair tasks for robustness.
   * Output `data/train.jsonl` and `data/valid.jsonl`.

### 2. Fine-Tune the Model
Once the data is ready, you can start the training process.

1. Open `notebooks/ft_mlx_lora.ipynb`.
2. Locate the configuration cell and select your base model (e.g., `MODEL_TYPE = "llama"` or `MODEL_TYPE = "qwen"`).
3. Run the notebook.
4. The script will:
   * Download the base model from Hugging Face if not present.
   * Train LoRA adapters using the parameters defined in `scripts/train_mlx.py`.
   * Fuse the adapters into the base model.
   * Save the final model to `local_models/ft_[model]_[framework]`.

### 3. Run Inference
You can test your fine-tuned model using the MLX command line interface or Python.

**Using CLI:**
```bash
python -m mlx_lm generate \
    --model local_models/ft_qwen_mlx \
    --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert in Answer Set Programming. Task: Generate the choice rules and ONLY the absolutely essential auxiliary logic. Output: Provide ONLY the valid Clingo code.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nProblem Description: Create a graph coloring instance with 3 nodes. Adjacent nodes cannot have the same color. Available colors are red, green, and blue.\n\n### TASK\nCreate the generator.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" \
    --max-tokens 512
```

## Workflow

The workflow is divided into two distinct stages, managed via the Jupyter Notebooks described above.

### 1. Data Conversion & Distillation
**Notebook:** `notebooks/data_converter.ipynb`

This step converts raw text/code pairs into a structured JSONL dataset suitable for instruction tuning.
*   **Logic:** Uses `scripts/data_converter.py`.
*   **Features:**
    *   **Structure Extraction:** Uses Gemini to reverse-engineer the ASP solution into a JSON structure (Instance Template, Generator, Hard/Soft Constraints).
    *   **Auto-Repair:** If the extracted ASP structure contains syntax errors, Gemini attempts to fix the Clingo code automatically.
    *   **Synthetic Repair Tasks:** Randomly injects "broken" code examples into the training set to teach the model how to fix syntax errors (robustness training).
*   **Output:** Generates `data/train.jsonl` and `data/valid.jsonl`.

### 2. LoRA Fine-Tuning (MLX)
**Notebook:** `notebooks/ft_mlx_lora.ipynb`

Performs Parameter-Efficient Fine-Tuning (PEFT) using LoRA.
*   **Logic:** Uses `scripts/train_mlx.py`.
*   **Supported Models:** Llama 3 (8B) and Qwen 2.5 (7B).
*   **Process:**
    1.  Converts the JSONL data into MLX-compatible format.
    2.  Trains adapters using LoRA (Low-Rank Adaptation).
    3.  **Fusing:** Merges the trained adapters back into the base model.
*   **Output:** A fine-tuned model saved in `local_models/ft_[model]_[framework]`.

## Prompt Engineering Strategy

The `prompts/` directory contains specific instructions used during the data conversion phase to break down complex ASP programs into manageable logic chunks.

*   **`extraction.txt`**: Deconstructs a full ASP solution into JSON components.
*   **`repair_gemini.txt`**: Used when the Python syntax checker detects invalid Clingo code; asks the LLM to fix it.
*   **`repair_task.txt`**: Used to generate synthetic training data where the model acts as a debugger.

## Dataset Format

The final training data (`train.jsonl`) follows a standard Chat format:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert in Answer Set Programming..."},
    {"role": "user", "content": "Problem Description: ..."},
    {"role": "assistant", "content": "loc(1..4). edge(1,2). ..."}
  ]
}
```

## License
This project uses benchmark data derived from various ASP competitions. Ensure compliance with original data licenses when distributing models.