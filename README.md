# Talking a Rejection Into an Accept? 🎭
### Confidence Framing and the Model-Specific Style Sensitivity of LLM Reviewers

🏆 **Accepted at the NeurIPS 2026 Workshop on AI-Native Academia** 

**Authors:** Prabhjot Singh, Somnath Luitel, Manmeet Singh

---

## 📌 Overview

Large language models already review real submissions at real venues. Prior work treats "the AI reviewer" as a single actor and reports pooled bias estimates. We ask a different question: **is style sensitivity a general property of LLM reviewers, or is it specific to particular models?**

We take **100 real abstracts** from ICLR 2026 and ICML 2026 (70 rejected, 30 accepted), rewrite each into **five controlled tones**, and score every version with **9 open-weight reviewer models**.

| Tone | Axis |
|---|---|
| Hedged | Confidence |
| Assertive | Confidence |
| Overclaiming | Confidence |
| Native-fluent | Register |
| L2-style | Register |

Each rewrite is produced by an LLM instructed to preserve every factual claim.

---

## 🔍 Key Findings

- 🧯 **Hedging is penalized by every model.** Scores drop in all 9 of 9 reviewers, the only effect that replicates without exception.
- ↔️ **Overclaiming splits models by sign.** It is penalized in some models (down to Cohen's d = −1.64) and rewarded in others (up to d = +0.695).
- 🤖 **Model choice matters more than tone.** Which reviewer model is used shifts a paper's score (between-model SD ≈ 0.91) by more than any tone manipulation we test.
- 🚫 **No reliable evidence that tone talks a rejection into an accept.** After FDR correction, no model–tone pair shows significant net movement toward accept. The few significant movements run toward rejection, and raw reject→accept rates are small-sample and comparable to a native-fluent control rewrite.
- 📉 **Agreement with human outcomes is weak and uneven.** Per-model correlation with human scores ranges from −0.078 to +0.299 on abstract-only input.

⚠️ **Scope:** these results hold for small open-weight models (at most 32B parameters) and abstract-only input. They should not be assumed to transfer to frontier systems or to human program committees.

---

## 📂 Repository Structure

```
├── analysis_and_figures/   # Final parsed datasets, statistical summaries, and figures
├── code/                   # Python scripts for the full pipeline
│   ├── make_abstracts.py   # 1. Generates the stylistic rewrites
│   ├── score_review.py     # 2. Prompts the reviewer models to score all variants
│   └── analyze_code.py     # 3. Runs the statistics and generates tables/figures
├── datasets/               # Original ICLR and ICML abstracts (70 rejected, 30 accepted)
└── result/                 # Raw outputs and LLM scores from the evaluation scripts
```

---

## 🚀 How to Run

Run the scripts in this order.

### 1️⃣ Generate the tonal variants
```bash
python code/make_abstracts.py
```

### 2️⃣ Score the abstracts with the reviewer models
```bash
python code/score_review.py
```
Each abstract-variant-model combination is sampled 3 times at temperature 0.7. This step needs access to the nine reviewer models listed below.

### 3️⃣ Analyze the results
```bash
python code/analyze_code.py
```
This runs the mixed-effects model, paired Wilcoxon tests, McNemar tests with Benjamini–Hochberg FDR correction, Verdict Switch Rate, and the manipulation and content-invariance checks.

### 📦 Python dependencies
```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib openpyxl google-genai torch transformers huggingface_hub
```
The input and output paths are set at the top of each script, so adjust them if your folder layout differs.

---

## 🤖 Reviewer Models

| Family | Models |
|---|---|
| Llama | Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct |
| Qwen | Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct |
| Gemma | gemma-4-E4B-it, gemma-4-31B-it |
| Ministral | Ministral-3-8B-Instruct-2512, Ministral-3-14B-Instruct-2512 |

Tonal rewrites were generated with `gemini-3.7-flash`.

---

## 📝 Notes

- 🏷️ **Naming:** the code and some data files use `l2_grounded`. In the paper, this condition is called **L2-style**, because it is a simulated register that was not validated against corpus or human evidence.
- 🔬 **Content invariance** is enforced by instruction and checked only lexically (TF-IDF cosine similarity). We did not manually audit the full set of 500 rewrites.
- 🖥️ **Compute:** all experiments ran on a single NVIDIA GH200 (120GB) GPU via the Texas Advanced Computing Center (TACC), taking roughly 50–60 hours in total.
