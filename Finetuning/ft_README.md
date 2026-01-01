# Fine-tuning LLMs for Answer Set Programming

This directory contains the pipeline for Supervised Fine-Tuning (SFT) of Large Language Models (LLMs) to generate Answer Set Programming (ASP) code. The methodology is designed to overcome the "Knowledge Acquisition Bottleneck" by training models not just to translate, but to reason iteratively and repair their own syntax errors.

---

## 1. Data Methodology: "Step-by-Step" Reasoning

Standard translation approaches often train models to generate an entire code file from a single prompt. However, this increases the risk of hallucination and variable inconsistency in complex logic programs.

**Our Approach:** We structure the training data as **Multi-Turn Conversations**.
*   **Structure:**
    *   **System:** Expert Persona + Full Natural Language (NL) Problem Description.
    *   **User:** A specific constraint instruction (extracted from comments in the Intermediate Representation).
    *   **Assistant:** The corresponding single ASP logic rule.
    *   *(Repeats for every rule in the problem)*
*   **Reason:** This aligns the training objective with the inference time "Feedback Loop." It teaches the model to generate code that is consistent with the *history* of what it has already generated while focusing on one logic task at a time.

---

## 2. Advanced Data Augmentation

Given the scarcity of high-quality ASP datasets, we employ a sophisticated augmentation pipeline to prevent overfitting and force generalization.

### A. Algorithmic Augmentation (Rule-Based)
Implemented in `scripts/data_converter.py`.

1.  **Local Alpha-Renaming:**
    *   *Description:* Variables in ASP rules are renamed *per rule* using a dynamic pool of **Canonical** (`V0`, `V1`) and **Abstract** (`X`, `Y`) terms.
    *   *Reason:* This prevents "Rote Memorization." It teaches the model that capitalized terms are local placeholders (quantifiers) and forces it to learn the *abstract logical structure* of the rule rather than memorizing specific strings like "Course" or "Time".
2.  **Input Noise:**
    *   *Description:* We generate variations where the user instructions are lowercased and stripped of punctuation (e.g., "Implement: Constraint X != Y" $\to$ "implement constraint x y").
    *   *Reason:* This forces the model to rely on semantic reasoning and context to infer operators and intent, rather than just translating symbols one-to-one.
3.  **Synthetic Repair Training:**
    *   *Description:* We algorithmically corrupt valid code (e.g., removing periods, creating unsafe variables, breaking operators) to mimic Clingo compiler errors. The model is trained on independent turns mapping `[Broken Code + Clingo Error]` $\to$ `[Fixed Code]`.
    *   *Reason:* This directly trains the model for the "Solver-in-the-Loop" architecture, turning syntax correction into a learned skill rather than relying on zero-shot capabilities.

### B. Synthetic Knowledge Distillation (LLM-Based)
Implemented in `scripts/data_augmentor.py` using **Google Gemini 3 Flash**.

1.  **Stylistic Paraphrasing:**
    *   *Description:* Rewriting problem descriptions in "Natural," "Instructional," and "Academic" tones.
    *   *Reason:* Ensures the model is robust to different user prompting styles and vocabularies.

---

## 3. Training Architecture (PEFT)

We use **Low-Rank Adaptation (LoRA)** to adapt the base model (`Meta-Llama-3-8B` or `Qwen2.5-7B`).

### Hyperparameters & Configuration
*   **Rank ($r=16$) and Alpha ($\alpha=160$):** High scaling factor (10x) to prioritize learned ASP syntax.
*   **Target Layers (Top 16):** We apply adapters only to the top 16 transformer layers, specializing high-level reasoning while preserving pre-trained linguistic knowledge in lower layers.
*   **Iterations:** ~550 (approx 2 epochs for ~1,000 examples) to prevent overfitting.
*   **Context Length:** 4096 tokens to accommodate full ASP programs.
*   **Precision:** bfloat16.

---

## 4. How to Use

### Step 1: Setup
Install dependencies:
```bash
pip install -r requirements_ft.txt
```
Ensure your `.env` file contains `HF_KEY` (HuggingFace) and `GOOGLE_API_KEY` (Gemini).

### Step 2: Generate Data
1.  Run `notebooks/data_augmentation.ipynb`. This uses Gemini to create Paraphrased variants of your base benchmarks.
2.  Run `notebooks/data_converter.ipynb`. This parses all data into the JSONL format, applying Renaming, Noise, and generating Repair Tasks.

### Step 3: Train (Choose One)

#### Option A: PyTorch (Linux / Windows / CUDA)
Use `notebooks/ft_pytorch_qlora.ipynb`.

*   **Engine:** Hugging Face `trl` + `peft`.
*   **Features:** Includes **NEFTune ($\alpha=5$)** noise embeddings for better generalization.
*   **Process:**
    1.  Set `MODEL_TYPE` (e.g., `"llama"`).
    2.  Execute training cell.
    3.  Script automatically fuses the adapter into `local_models/ft_llama_pytorch`.

#### Option B: MLX (macOS Apple Silicon)
Use `notebooks/ft_mlx_lora.ipynb`.

*   **Engine:** Apple `mlx-lm` framework (highly optimized for MacOS).
*   **Process:**
    1.  Set `MODEL_TYPE` (e.g., `"qwen"`).
    2.  Execute training cell.
    3.  Run the **Fuse** cell at the end. *Note: You can select a specific checkpoint (e.g., step 700) if the final model overfits.*
    4.  Final model is saved to `local_models/ft_qwen_mlx`.


Finetuning/
├── data/                   # Processed JSONL datasets
├── adapters/               # LoRA checkpoints
├── notebooks/
│   ├── data_augmentation.ipynb # Gemini-based Paraphrasing
│   ├── data_converter.ipynb    # Conversion + Algorithmic Augmentation
│   ├── ft_pytorch_qlora.ipynb  # PyTorch Training Notebook
│   └── ft_mlx_lora.ipynb       # MLX (macOS) Training Notebook
├── scripts/
│   ├── train_pytorch.py        # SFTTrainer logic
│   ├── train_mlx.py            # MLX CLI wrapper
│   ├── data_converter.py       # Regex parsing, Alpha-Renaming, Repairs
│   └── data_augmentor.py       # LLM Paraphrasing Logic
└── utils_ft.py                 # Path & Env utilities