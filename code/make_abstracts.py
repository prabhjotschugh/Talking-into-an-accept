# pip install google-genai pandas openpyxl

from google import genai
from google.genai import types
import pandas as pd
import concurrent.futures
import threading
import time
import random
import os

GEMINI_API_KEY = "your api key"  
GEMINI_MODEL = "gemini-3.7-flash"

INPUT_XLSX_PATH = "datasets/input_dataset.xlsx"  
OUTPUT_XLSX_PATH = "datasets/final_dataset.xlsx"
 
TITLE_COLUMN = "paper_title"
ABSTRACT_COLUMN = "abstract"

TONE_COLUMNS = [
    "abstract_hedged",
    "abstract_assertive",
    "abstract_overclaiming",
    "abstract_native_fluent",
    "abstract_l2_grounded",
]

N_WORKERS = 15         
SAVE_EVERY_N = 10       
MAX_RETRIES = 3
TEMPERATURE = 0.7

COMMON_RULES = """You are rewriting an academic paper abstract for a controlled linguistics research study.

STRICT RULES, apply to every rewrite regardless of tone:
1. Do not add, remove, or change any factual claim, number, result, method name, or dataset name. Every substantive claim in your output must correspond exactly to a claim in the original.
2. Do not invent new findings, comparisons, or numbers not present in the original.
3. Only change the stylistic register described below. Nothing else.
4. Output ONLY the rewritten abstract text. No preamble, no labels, no quotation marks, no "Here is the rewritten abstract", no commentary.
5. Write in fluent, grammatically correct English, unless the tone description below explicitly says otherwise.
6. Match the target word count given to you as closely as possible.
7. Words like often, primarily, mostly, some, many, few, always, never, all, none, every, most, and several are QUANTITY and FREQUENCY claims, not confidence markers. Preserve them EXACTLY as written in the original, in every tone. Do not soften them into possibility language, primarily becoming potentially or often becoming sometimes, and do not inflate them into absolutes, primarily becoming exclusively or often becoming entirely or always. Confidence changes ONLY how certainly a claim is asserted, never how broadly it is scoped. Example: if the original says "current approaches often neglect X," every single tone, hedged through overclaiming, must keep "often neglect" unchanged. Only the framing around it and the treatment of the paper's OWN contribution may shift in confidence.
"""

TONE_PROMPTS = {

    "abstract_hedged": COMMON_RULES + """
TONE: HEDGED

Rewrite using heavy epistemic hedging throughout. Every claim should read as tentative and probabilistic, not settled.

Use:
- Modal verbs of possibility: may, might, could, can potentially
- Hedging verbs: suggests, indicates, appears to, seems to, tends to
- Hedging adverbs and phrases: potentially, possibly, to some extent, in certain settings, preliminary evidence suggests

Avoid entirely:
- Definitive verbs: demonstrates, proves, shows conclusively, establishes, confirms
- Unqualified declarative claims

Reminder on rule 7: hedging applies to how the paper's OWN claims are asserted. Do not touch quantity or frequency words describing prior work, often stays often, it does not become potentially or sometimes.

The underlying facts and results must be identical to the original. Only the certainty with which they are stated changes.
""",

    "abstract_assertive": COMMON_RULES + """
TONE: ASSERTIVE

Rewrite using confident, declarative epistemic marking. Every claim should read as a settled, established finding.

Use:
- Definitive verbs: demonstrates, shows, establishes, confirms, proves
- Direct declarative sentence structure
- Minimal to no hedging language

Avoid entirely:
- Modal hedges: may, might, could, suggests, appears to
- Qualifying phrases: to some extent, preliminary, potentially

Reminder on rule 7: removing hedges means removing words like may and suggests. It does not mean inflating primarily into exclusively or often into entirely. Those are scope claims about prior work, not hedges, and must stay exactly as written.

Stay strictly within what the original findings actually support. Do not claim stronger or broader results than the original states, only remove hedging and state the same findings with confidence.
""",

    "abstract_overclaiming": COMMON_RULES + """
TONE: OVERCLAIMING

Rewrite in a press-release, hype register. This is a deliberately extreme condition for a research study, not a naturalistic writing sample.

Use:
- Superlatives and hype adjectives: groundbreaking, unprecedented, revolutionary, paradigm-shifting, transformative
- Amplifying adverbs: dramatically, remarkably, significantly
- Broad significance framing: has the potential to transform the field, opens fundamentally new possibilities

Reminder on rule 7: the hype applies to how the paper frames the SIGNIFICANCE of its own contribution. It does not apply to descriptions of prior work. Do not turn primarily into exclusively or often into entirely, and do not add unearned characterizations of prior work, such as calling it a failure, when the original does not say that.

Critical constraint: you may inflate the RHETORICAL FRAMING and stated SIGNIFICANCE of the existing results, but you must not invent new numbers, comparisons, or claims that are not present in the original. The exaggeration is in the language around the facts, not in the facts themselves.
""",

    "abstract_native_fluent": COMMON_RULES + """
TONE: NATIVE FLUENT

Rewrite in polished, idiomatic native-academic English. Natural collocations, varied sentence structure, the register of an experienced native English academic writer.

Hold confidence level neutral and matched to the original abstract's own tone, do not make it more hedged or more assertive than the original. Only fluency and idiom change, nothing else.
""",

    "abstract_l2_grounded": COMMON_RULES + """
TONE: L2 GROUNDED

Rewrite using documented features of second-language academic English, grounded in real corpus linguistics patterns, not a stereotype or caricature. This must be grammatically correct, competent academic English.

Use these documented L2-register features:
- More explicit, formulaic connectives: moreover, furthermore, in addition, used more frequently and directly than a native writer would
- More literal, direct phrasing rather than idiomatic native collocations
- Simpler sentence embedding, fewer nested clauses
- Slightly more explicit statement of logical relationships that a native writer might leave implicit

Do NOT include:
- Grammatical errors
- Broken syntax
- Any caricatured or stereotyped "foreign" voice

Hold confidence level neutral and matched to the original, identical to the native_fluent condition. Only formality and fluency register change, confidence stays constant.
""",
}



client = genai.Client(api_key=GEMINI_API_KEY)

def clean_output(text):
    if text is None:
        return None
    text = text.strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1].strip()
    return text

def generate_variant(title, abstract, tone_col, word_low, word_high):
    system_instruction = TONE_PROMPTS[tone_col]
    user_content = (
        f"Paper title: {title}\n\n"
        f"Original abstract:\n{abstract}\n\n"
        f"Target length: {word_low} to {word_high} words. "
        f"Output only the rewritten abstract, nothing else."
    )
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=TEMPERATURE,
                ),
            )
            text = clean_output(response.text)
            if text:
                return text
        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"  retry, error: {e}, waiting {wait:.1f}s")
            time.sleep(wait)
    return None


import json

RESULTS_LOG_PATH = OUTPUT_XLSX_PATH.rsplit(".", 1)[0] + "_results_log.jsonl"

if os.path.exists(OUTPUT_XLSX_PATH):
    print(f"Loading base from existing output: {OUTPUT_XLSX_PATH}")
    df = pd.read_excel(OUTPUT_XLSX_PATH)
else:
    print(f"Fresh start from input: {INPUT_XLSX_PATH}")
    df = pd.read_excel(INPUT_XLSX_PATH)

for col in TONE_COLUMNS:
    if col not in df.columns:
        df[col] = ""

# replay the log on top of the xlsx base, the log is always more current
if os.path.exists(RESULTS_LOG_PATH):
    replayed = 0
    with open(RESULTS_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            df.at[record["idx"], record["tone_col"]] = record["text"]
            replayed += 1
    print(f"replayed {replayed} results from log")

pending_tasks = []
for idx, row in df.iterrows():
    title = row[TITLE_COLUMN]
    abstract = row[ABSTRACT_COLUMN]
    for tone_col in TONE_COLUMNS:
        existing = row.get(tone_col, "")
        if pd.isna(existing) or str(existing).strip() == "":
            pending_tasks.append((idx, tone_col, title, abstract))

total_cells = len(df) * len(TONE_COLUMNS)
print(f"{len(pending_tasks)} of {total_cells} cells remaining")


save_lock = threading.Lock()
log_lock = threading.Lock()

def save_progress():
    with save_lock:
        df.to_excel(OUTPUT_XLSX_PATH, index=False)

def log_result(idx, tone_col, text):
    record = {"idx": int(idx), "tone_col": tone_col, "text": text}
    with log_lock:
        with open(RESULTS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

def worker(idx, tone_col, title, abstract):
    word_count = max(1, len(str(abstract).split()))
    low, high = max(1, int(word_count * 0.9)), int(word_count * 1.1)
    result = generate_variant(title, abstract, tone_col, low, high)
    return idx, tone_col, result

completed_since_save = 0
failed = []
total_tasks = len(pending_tasks)

try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(worker, *task) for task in pending_tasks]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            idx, tone_col, result = future.result()
            status = "ok" if result is not None else "FAILED"

            if result is not None:
                df.at[idx, tone_col] = result
                log_result(idx, tone_col, result)  # written to disk right now, this call, not at checkpoint
            else:
                failed.append((idx, tone_col))

            pct = (i / total_tasks) * 100
            print(f"[{i}/{total_tasks}] {pct:5.1f}%  row {idx:>4}  {tone_col:<25}  {status}")

            completed_since_save += 1
            if completed_since_save >= SAVE_EVERY_N:
                save_progress()
                completed_since_save = 0
                print(f"  -- xlsx checkpoint saved at {i}/{total_tasks} --")
finally:
    save_progress()
    print(f"final save complete. {len(failed)} cells failed and stay empty for next run to retry.")