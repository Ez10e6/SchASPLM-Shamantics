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

### 3. Data Preparation & Augmentation
The pipeline employs a sophisticated data generation strategy designed to overcome the limitations of small datasets and prevent overfitting to specific variable names or problem structures.

#### A. Step-by-Step Reasoning (Multi-Turn)
Instead of training the model to generate a full program in one pass, `data_converter.py` parses annotated ASP files to create **multi-turn conversations**:
*   **System Context:** Contains the expert persona and the full Natural Language (NL) problem description.
*   **User Turn:** A specific constraint instruction (extracted from comments in the Intermediate Representation).
*   **Assistant Turn:** The corresponding single ASP logic rule.
This aligns the training objective with the iterative nature of the inference feedback loop.

#### B. Structural Augmentation
To force the model to learn abstract logic structures (ASTs) rather than memorizing token sequences, we apply algorithmic augmentations during conversion:
1.  **Step Shuffling:** Exploiting the declarative nature of ASP, we generate permutations of the constraint ordering within the conversation history.
2.  **Variable Renaming (Alpha-Conversion):** Variables in ASP rules are locally renamed per rule. We use a dynamic pool of **Canonical** (`V0`, `V1`), **Abstract** (`X`, `Y`), and **Semantic** terms (e.g., `Course`, `Time` extracted from the NL description). This teaches the model that variable names are local placeholders.
3.  **Input Noise:** Creating variations with lowercased or unpunctuated user prompts to improve robustness against imperfect inputs.

#### C. Synthetic Repair Training
The pipeline generates specific **Debugging Tasks** where valid ASP code is algorithmically corrupted (e.g., removing periods, creating unsafe variables, breaking operators) to mimic realistic Clingo compiler errors. The model is explicitly trained to map `[Broken Code + Clingo Error]` $\to$ `[Fixed Code]`, directly optimizing it for the solver-in-the-loop workflow.

#### D. Synthetic Data Generation (LLM Paraphrasing)
The `paraphraser_gemini.py` script uses **Google Gemini 3 Flash** to rewrite problem descriptions in distinct styles (**Natural**, **Instructional**, **Academic**) while preserving logic. This expands the linguistic diversity of the input space and prevents stylistic overfitting.

#### E. Leakage Prevention
Data splitting is performed at the **Problem ID level**. All variations (Original, Shuffled, Renamed, Paraphrased) of a specific problem are kept together in either the Train or Validation set to ensure the validation metrics reflect true generalization.

---

## How to Use

### 1. Setup
Install the specific fine-tuning dependencies:
```bash
pip install -r requirements_ft.txt