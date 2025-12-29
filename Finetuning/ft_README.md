# Fine-tuning LLMs for Answer Set Programming

This directory contains the pipeline for Supervised Fine-Tuning (SFT) of Large Language Models (LLMs) to generate Answer Set Programming (ASP) code. The methodology is designed to overcome the "Knowledge Acquisition Bottleneck" by training models not just to translate, but to reason iteratively and repair their own syntax errors.

---

## 1. Data Methodology: "Step-by-Step" Reasoning

Standard translation approaches often train models to generate an entire code file from a single prompt. However, this increases the risk of hallucination and variable inconsistency in complex logic programs.

**Our Approach:** We structure the training data as **Multi-Turn Conversations**.
*   **Structure:**
    *   **System:** Expert Persona + Full Natural Language (NL) Problem Description.
    *   **User:** A specific constraint instruction (extracted from the Intermediate Representation).
    *   **Assistant:** The corresponding single ASP logic rule.
    *   *(Repeats for every rule in the problem)*
*   **Reason:** This aligns the training objective with the inference time "Feedback Loop." It teaches the model to generate code that is consistent with the *history* of what it has already generated (e.g., reusing the correct variable names defined in previous steps) while focusing on one logic task at a time.

---

## 2. Advanced Data Augmentation

Given the scarcity of high-quality ASP datasets, we employ a sophisticated augmentation pipeline to prevent overfitting and force generalization.

### A. Algorithmic Augmentation (Rule-Based)
Implemented in `scripts/data_converter.py`.

1.  **Constraint Permutation (Shuffling):**
    *   *Description:* We generate random permutations of the constraint ordering within the conversation history.
    *   *Reason:* ASP is a declarative language; the order of rules does not affect the semantics. This prevents the model from overfitting to specific sequences (e.g., assuming "hard constraints" always follow "generators").
2.  **Local Alpha-Renaming:**
    *   *Description:* Variables in ASP rules are renamed *per rule* using a dynamic pool of **Canonical** (`V0`, `V1`), **Abstract** (`X`, `Y`), and **Semantic** terms (extracted from the NL text, e.g., `Nurse`, `Shift`).
    *   *Reason:* This prevents "Rote Memorization." It teaches the model that capitalized terms are local placeholders (quantifiers) and forces it to learn the *abstract logical structure* of the rule rather than memorizing specific strings.
3.  **Synthetic Repair Training:**
    *   *Description:* We algorithmically corrupt valid code (e.g., removing periods, creating unsafe variables, breaking operators) to mimic Clingo compiler errors. The model is trained on independent turns mapping `[Broken Code + Clingo Error]` $\to$ `[Fixed Code]`.
    *   *Reason:* This directly trains the model for the "Solver-in-the-Loop" architecture, turning syntax correction into a learned skill rather than relying on zero-shot capabilities.

### B. Synthetic Knowledge Distillation (LLM-Based)
Implemented in `scripts/data_augmentor.py` using **Google Gemini 3 Flash/Pro**.

1.  **Stylistic Paraphrasing:**
    *   *Description:* Rewriting problem descriptions in "Natural," "Instructional," and "Academic" tones.
    *   *Reason:* Ensures the model is robust to different user prompting styles and vocabularies.
2.  **Logic Morphing (Domain Reskinning):**
    *   *Description:* Asking the LLM to "reskin" a problem (e.g., Nurse Scheduling $\to$ Factory Machine Allocation) by renaming all predicates and constants while keeping the logical operators identical. Syntax is validated via Clingo.
    *   *Reason:* This decouples **Logic** from **Domain**. It teaches the model to apply scheduling logic to *any* context, preventing it from overfitting to specific scenarios like "Nurses."

---

## 3. Training Architecture (PEFT)

We use **Low-Rank Adaptation (LoRA)** via the PyTorch `trl` library.

*   **Rank ($r=16$) and Alpha ($\alpha=160$):**
    *   *Reason:* This results in a high scaling factor ($\frac{\alpha}{r} = 10$). We intentionally "force" the model to prioritize the learned ASP patterns over its pre-trained natural language weights, essentially converting it into a domain specialist.
*   **Target Layers (Top 16):**
    *   *Reason:* We apply adapters only to the top 16 transformer layers. Lower layers retain general linguistic understanding, while higher abstraction layers are specialized for the complex reasoning required by ASP.
*   **Dropout ($0.05$):**
    *   *Reason:* Essential regularization to prevent the model from memorizing the specific training examples in our medium-sized dataset (~1,400 examples).

---

## 4. Hyperparameters & Configuration

*   **Iterations: 700 (~2 Epochs):**
    *   *Reason:* With the augmented dataset size (~1,400), 700 steps (batch size 1, grad accum 4) ensures the model sees every logical pattern roughly twice. Less is underfitting; more risks overfitting.
*   **Context Length: 4096:**
    *   *Reason:* ASP files can be dense. Increasing from 2048 prevents truncation of the final logical constraints, ensuring the model learns to write complete, valid programs.
*   **NEFTune Noise ($\alpha=5$):**
    *   *Reason:* We inject noise into the embedding vectors during training. This is a proven technique for improving generalization and instruction-following robustness when fine-tuning on synthetic or repetitive datasets.
*   **Precision: bfloat16:**
    *   *Reason:* Provides better dynamic range than float16, preventing gradient underflow/overflow during the training of logic-heavy sequences.

---

## 5. How to Use

### 1. Setup
Install dependencies:
```bash
pip install -r requirements_ft.txt
```

Ensure your `.env` file contains `HF_KEY` (HuggingFace) and `GOOGLE_API_KEY` (Gemini).

### 2. Generate Data
1.  Run `notebooks/data_augmentation.ipynb`. This uses Gemini to create Paraphrased and Morphed variants of your base benchmarks.
2.  Run `notebooks/data_converter.ipynb`. This parses all data into the JSONL format, applying Shuffling, Renaming, and generating Repair Tasks.

### 3. Train
1.  Open `notebooks/ft_pytorch_qlora.ipynb`.
2.  Set `MODEL_TYPE` (e.g., `"llama"`).
3.  Execute training. The script will:
    *   Load the base model in 4-bit/bf16.
    *   Apply the LoRA configuration.
    *   Train for 700 steps.
    *   Fuse the adapters into the base model automatically.
4.  The final model is saved to `local_models/ft_llama_pytorch`.


Finetuning/
├── data/                   # Processed JSONL datasets
├── adapters/               # LoRA checkpoints
├── notebooks/
│   ├── data_augmentation.ipynb # Gemini-based Synthetic Data Generation
│   ├── data_converter.ipynb    # Conversion + Algorithmic Augmentation
│   ├── ft_pytorch_qlora.ipynb  # Main Training Notebook
│   └── ft_mlx_lora.ipynb       # Alternative macOS Training
├── scripts/
│   ├── train_pytorch.py        # SFTTrainer logic (NEFTune, Callbacks)
│   ├── data_converter.py       # Regex parsing, Alpha-Renaming, Shuffling
│   └── data_augmentor.py       # Logic Morphing & Paraphrasing via Gemini
└── utils_ft.py                 # Path & Env utilities