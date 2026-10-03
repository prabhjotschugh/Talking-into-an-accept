"""
score_reviews.py
"""

import os
os.environ["HF_HOME"] = "/scratch/10778/prabhjotschugh/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torch
import pandas as pd
import json
import re
import time
import gc
import traceback
import threading
import concurrent.futures
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config
from huggingface_hub import login



HF_TOKEN = "<YOUR_HUGGINGFACE_TOKEN>"

INPUT_XLSX_PATH = "datasets/final_dataset.xlsx"
OUTPUT_XLSX_PATH = "result/final_results.xlsx"

N = 1
N_WORKERS_STANDARD = 8
N_WORKERS_GEMMA = 4
N_WORKERS_MINISTRAL = 4
REPEATS = 3
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 1000

TITLE_COLUMN = "paper_title"
VERDICT_COLUMN = "verdict"

TEXT_COLUMNS = {
    "original":       "abstract",
    "hedged":         "abstract_hedged",
    "assertive":      "abstract_assertive",
    "overclaiming":   "abstract_overclaiming",
    "native_fluent":  "abstract_native_fluent",
    "l2_grounded":    "abstract_l2_grounded",
}

RESULTS_LOG_PATH = OUTPUT_XLSX_PATH.rsplit(".", 1)[0] + "_scoring_log.jsonl"
XLSX_CHECKPOINT_EVERY = 25

MODEL_REGISTRY = [
    {"name": "llama_3.2_3b",   "id": "meta-llama/Llama-3.2-3B-Instruct",       "family": "standard"},
    {"name": "llama_3.1_8b",   "id": "meta-llama/Llama-3.1-8B-Instruct",       "family": "standard"},
    {"name": "qwen2.5_3b",     "id": "Qwen/Qwen2.5-3B-Instruct",               "family": "standard"},
    {"name": "qwen2.5_7b",     "id": "Qwen/Qwen2.5-7B-Instruct",               "family": "standard"},
    {"name": "qwen2.5_32b",    "id": "Qwen/Qwen2.5-32B-Instruct",              "family": "standard"},
    {"name": "gemma4_e4b",     "id": "google/gemma-4-E4B-it",                  "family": "gemma"},
    {"name": "gemma4_31b",     "id": "google/gemma-4-31B-it",                  "family": "gemma"},
    {"name": "ministral3_8b",  "id": "mistralai/Ministral-3-8B-Instruct-2512", "family": "ministral"},
    {"name": "ministral3_14b", "id": "mistralai/Ministral-3-14B-Instruct-2512","family": "ministral"},
]


REVIEWER_SYSTEM_PROMPT = """You are an expert peer reviewer for a top-tier machine learning conference (NeurIPS/ICML/ICLR level). You will be shown a paper title and abstract. Evaluate it as you would a real submission.

Provide your response in EXACTLY this format, nothing else:

SCORE: [integer 1-10, where 1=trivial/flawed, 5=borderline, 10=groundbreaking]
DECISION: [ACCEPT or REJECT]
REASONING: [2-3 sentences explaining your score and decision]

Do not include any other text before or after this block. Do not use markdown formatting, bold, asterisks, or headers anywhere in your response. Output plain text only."""


def build_user_prompt(title, abstract_text):
    return f"Paper Title: {title}\n\nAbstract:\n{abstract_text}\n\nProvide your evaluation now."


def parse_review_response(raw_text):
    score_match = re.search(r"SCORE:\s*(\d+)", raw_text, re.IGNORECASE)
    # tolerate Ministral 14B's repeated-letter typo: ACCECEPT, ACCCEPT, etc.
    # AC + 1-3 extra C/E characters + EPT still unambiguously means ACCEPT
    decision_match = re.search(r"DECISION:\s*(AC[CE]{1,3}EPT|REJECT)", raw_text, re.IGNORECASE)
    reasoning_match = re.search(r"REASONING:\s*(.+)", raw_text, re.IGNORECASE | re.DOTALL)

    score = int(score_match.group(1)) if score_match else None
    if decision_match:
        raw_decision = decision_match.group(1).upper()
        decision = "ACCEPT" if raw_decision != "REJECT" else "REJECT"
    else:
        decision = None
    reasoning = reasoning_match.group(1).strip()[:500] if reasoning_match else None

    return score, decision, reasoning



def load_standard(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN
    )
    return {"model": model, "tokenizer": tokenizer}


def generate_standard(ctx, system_prompt, user_prompt, temperature, max_new_tokens):
    model, tokenizer = ctx["model"], ctx["tokenizer"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_gemma(model_id):
    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto", token=HF_TOKEN)
    return {"model": model, "processor": processor}


def generate_gemma(ctx, system_prompt, user_prompt, temperature, max_new_tokens):
    model, processor = ctx["model"], ctx["processor"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True)


def load_ministral(model_id):
    tokenizer = MistralCommonBackend.from_pretrained(model_id, token=HF_TOKEN)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=FineGrainedFP8Config(dequantize=True),
        token=HF_TOKEN,
    )
    return {"model": model, "tokenizer": tokenizer}


def generate_ministral(ctx, system_prompt, user_prompt, temperature, max_new_tokens):
    model, tokenizer = ctx["model"], ctx["tokenizer"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
    tokenized = {
        k: v.to(device=model.device) if hasattr(v, "to") else v
        for k, v in tokenized.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            max_length=None,
            do_sample=True,
            temperature=temperature,
        )
    new_tokens = output_ids[0][tokenized["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens)


FAMILY_LOADERS = {"standard": load_standard, "gemma": load_gemma, "ministral": load_ministral}
FAMILY_GENERATORS = {"standard": generate_standard, "gemma": generate_gemma, "ministral": generate_ministral}
FAMILY_WORKERS = {"standard": N_WORKERS_STANDARD, "gemma": N_WORKERS_GEMMA, "ministral": N_WORKERS_MINISTRAL}


def unload_model(ctx):
    for v in ctx.values():
        del v
    gc.collect()
    torch.cuda.empty_cache()


def load_completed_keys(log_path):
    completed = set()
    if not os.path.exists(log_path):
        return completed
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            completed.add((r["paper_id"], r["text_type"], r["model_name"], r["repeat_idx"]))
    return completed


def build_task_list(df, model_name, completed_keys):
    tasks = []
    for row_idx, row in df.iterrows():
        paper_id = row["paper_id"]
        for text_type, col in TEXT_COLUMNS.items():
            text_val = row.get(col, "")
            if pd.isna(text_val) or str(text_val).strip() == "":
                continue
            for repeat_idx in range(REPEATS):
                key = (paper_id, text_type, model_name, repeat_idx)
                if key not in completed_keys:
                    tasks.append((row_idx, paper_id, text_type, str(text_val), repeat_idx))
    return tasks


log_lock = threading.Lock()


def log_result(paper_id, text_type, model_name, repeat_idx, score, decision, reasoning, raw_text, error=None):
    record = {
        "paper_id": paper_id,
        "text_type": text_type,
        "model_name": model_name,
        "repeat_idx": repeat_idx,
        "score": score,
        "decision": decision,
        "reasoning": reasoning,
        "raw_text": raw_text[:1000] if raw_text else None,
        "error": error,
        "timestamp": time.time(),
    }
    with log_lock:
        with open(RESULTS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())


def rebuild_xlsx_from_log(df_source, log_path, output_path):
    if not os.path.exists(log_path):
        return
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return

    all_results_df = pd.DataFrame(records)

    valid_df = all_results_df[all_results_df["error"].isna()].copy()

    pivot = valid_df.groupby(["paper_id", "text_type", "model_name"]).agg(
        mean_score=("score", "mean"),
        n_valid_repeats=("score", "count"),
        accept_rate=("decision", lambda s: (s == "ACCEPT").mean()),
    ).reset_index()

    detail_cols = ["paper_id", "text_type", "model_name", "repeat_idx",
                    "score", "decision", "reasoning", "raw_text", "error"]
    detail_df = all_results_df[detail_cols].sort_values(
        ["paper_id", "text_type", "model_name", "repeat_idx"]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pivot.to_excel(writer, sheet_name="summary", index=False)
        detail_df.to_excel(writer, sheet_name="all_results", index=False)



def process_one_task(family, ctx, generator_fn, title, text_val, temperature, max_new_tokens):
    try:
        user_prompt = build_user_prompt(title, text_val)
        raw_text = generator_fn(ctx, REVIEWER_SYSTEM_PROMPT, user_prompt, temperature, max_new_tokens)
    except Exception as e:
        tb = traceback.format_exc()
        return None, None, None, None, f"generation_error | {type(e).__name__}: {e}\n{tb}"

    score, decision, reasoning = parse_review_response(raw_text)
    if score is None or decision is None:
        preview = raw_text[:200] if raw_text else "(empty response)"
        return None, None, None, raw_text, f"parse_failed | raw_text_preview={preview!r}"

    return score, decision, reasoning, raw_text, None


def run_model(model_entry, df, completed_keys):
    model_name = model_entry["name"]
    model_id = model_entry["id"]
    family = model_entry["family"]

    tasks = build_task_list(df, model_name, completed_keys)
    if not tasks:
        print(f"[{model_name}] nothing pending, skipping load entirely")
        return

    print(f"[{model_name}] loading {model_id} ({family} family)")
    load_fn = FAMILY_LOADERS[family]
    generate_fn = FAMILY_GENERATORS[family]
    n_workers = FAMILY_WORKERS[family]

    ctx = load_fn(model_id)
    print(f"[{model_name}] loaded. {len(tasks)} pending calls, {n_workers} workers")

    title_lookup = df.set_index("paper_id")[TITLE_COLUMN].to_dict()

    def worker(task):
        row_idx, paper_id, text_type, text_val, repeat_idx = task
        title = title_lookup.get(paper_id, "")
        score, decision, reasoning, raw_text, error = process_one_task(
            family, ctx, generate_fn, title, text_val, TEMPERATURE, MAX_NEW_TOKENS
        )
        log_result(paper_id, text_type, model_name, repeat_idx, score, decision, reasoning, raw_text, error)
        return paper_id, text_type, repeat_idx, error

    done_count = 0
    total = len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(worker, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            paper_id, text_type, repeat_idx, error = future.result()
            done_count += 1
            status = "ok" if error is None else f"FAILED: {error.splitlines()[0]}"
            pct = (done_count / total) * 100
            print(f"[{model_name}] [{done_count}/{total}] {pct:5.1f}%  {paper_id} / {text_type} / rep{repeat_idx}  {status}")

            if done_count % XLSX_CHECKPOINT_EVERY == 0:
                rebuild_xlsx_from_log(df, RESULTS_LOG_PATH, OUTPUT_XLSX_PATH)
                print(f"[{model_name}] -- xlsx checkpoint rebuilt at {done_count}/{total} --")

    print(f"[{model_name}] all pending calls done, unloading")
    unload_model(ctx)
    rebuild_xlsx_from_log(df, RESULTS_LOG_PATH, OUTPUT_XLSX_PATH)


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No CUDA device visible. Models will load to CPU.")

    login(token=HF_TOKEN)
    print(f"Logged in to Hugging Face. HF_HOME={os.environ['HF_HOME']}")

    print(f"Loading input: {INPUT_XLSX_PATH}")
    df_full = pd.read_excel(INPUT_XLSX_PATH)
    df = df_full.head(N).copy()
    print(f"Running on N={N} of {len(df_full)} total abstracts")

    completed_keys = load_completed_keys(RESULTS_LOG_PATH)
    print(f"Resuming: {len(completed_keys)} (paper, text, model, repeat) results already logged")

    for model_entry in MODEL_REGISTRY:
        run_model(model_entry, df, completed_keys)
        completed_keys = load_completed_keys(RESULTS_LOG_PATH)

    print("All 9 models complete.")
    rebuild_xlsx_from_log(df, RESULTS_LOG_PATH, OUTPUT_XLSX_PATH)
    print(f"Final xlsx written to {OUTPUT_XLSX_PATH}")
    print(f"Full raw log at {RESULTS_LOG_PATH}")


if __name__ == "__main__":
    main()