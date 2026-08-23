"""
analyze_results.py

Full analysis pipeline for the confidence-framing peer review study.
Run once completed-output_scoring_log.jsonl has all 16,200 lines.

pip install pandas numpy scipy statsmodels scikit-learn matplotlib openpyxl
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, ttest_rel, shapiro
from scipy.spatial.distance import pdist
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower
import statsmodels.formula.api as smf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=FutureWarning)


INPUT_XLSX_PATH = "results/final_results.xlsx"
RESULTS_LOG_PATH = "completed-output_scoring_log.jsonl"
OUTPUT_DIR = "analysis_output"
FDR_ALPHA = 0.05

TEXT_COLUMNS = {
    "original":      "abstract",
    "hedged":        "abstract_hedged",
    "assertive":     "abstract_assertive",
    "overclaiming":  "abstract_overclaiming",
    "native_fluent": "abstract_native_fluent",
    "l2_grounded":   "abstract_l2_grounded",
}
TONE_ONLY = {k: v for k, v in TEXT_COLUMNS.items() if k != "original"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_venue(paper_id):
    m = re.match(r"^([A-Za-z]+)_", str(paper_id))
    return m.group(1).upper() if m else "unknown"


def load_ground_truth(path):
    df = pd.read_excel(path)
    df["venue"] = df["paper_id"].apply(extract_venue)
    df["quality_tier"] = df["verdict"].str.strip().str.lower().map(
        {"reject": "reject_tier", "accept": "accept_tier"}
    )
    n_unknown_venue = (df["venue"] == "unknown").sum()
    if n_unknown_venue > 0:
        print(f"  WARNING: {n_unknown_venue} paper_id values did not match the expected venue prefix pattern")

    def mean_human_score(s):
        try:
            vals = [float(x) for x in str(s).split(",") if x.strip() != ""]
            return np.mean(vals) if vals else np.nan
        except Exception:
            return np.nan

    df["mean_human_score"] = df["review_rating"].apply(mean_human_score)
    return df


def load_scoring_log(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def build_cell_level_df(log_df, ground_truth_df):
    """One row per (paper_id, text_type, model_name): mean_score, majority decision."""
    valid = log_df[log_df["error"].isna()].copy()

    def majority_decision(decisions):
        decisions = [d for d in decisions if d in ("ACCEPT", "REJECT")]
        if not decisions:
            return None
        accept_frac = decisions.count("ACCEPT") / len(decisions)
        return "ACCEPT" if accept_frac >= 0.5 else "REJECT"

    agg = valid.groupby(["paper_id", "text_type", "model_name"]).agg(
        mean_score=("score", "mean"),
        n_valid_repeats=("score", "count"),
        accept_fraction=("decision", lambda s: (s == "ACCEPT").mean()),
    ).reset_index()

    decisions = valid.groupby(["paper_id", "text_type", "model_name"])["decision"] \
        .apply(lambda s: majority_decision(list(s))).reset_index(name="majority_decision")

    cell_df = agg.merge(decisions, on=["paper_id", "text_type", "model_name"])
    cell_df = cell_df.merge(
        ground_truth_df[["paper_id", "venue", "verdict", "quality_tier", "mean_human_score"]],
        on="paper_id", how="left"
    )
    return cell_df


def report_data_quality(log_df):
    errored = log_df[log_df["error"].notna()].copy()
    print(f"  Total logged calls: {len(log_df)}")
    print(f"  Errored calls: {len(errored)} ({100*len(errored)/max(len(log_df),1):.2f}%)")
    if len(errored) > 0:
        errored["error_type"] = errored["error"].str.split("|").str[0].str.strip()
        by_model = errored.groupby(["model_name", "error_type"]).size().reset_index(name="count")
        print("  Errors by model and type:")
        for _, row in by_model.iterrows():
            print(f"    {row['model_name']:<20} {row['error_type']:<20} {row['count']}")
    return errored

HEDGE_PATTERN = re.compile(
    r"\b(may|might|could|potentially|possibly|suggests?|indicates?|appears? to|seems? to|tends? to|preliminary)\b",
    re.IGNORECASE)
ASSERTIVE_PATTERN = re.compile(
    r"\b(demonstrates?|shows?|establishes?|confirms?|proves?|conclusively)\b", re.IGNORECASE)
HYPE_PATTERN = re.compile(
    r"\b(groundbreaking|unprecedented|revolutionary|paradigm-shifting|transformative|dramatically|remarkably|"
    r"transforms? the field|fundamentally new)\b", re.IGNORECASE)


def marker_density(text, pattern):
    text = str(text)
    n_words = max(len(text.split()), 1)
    return (len(pattern.findall(text)) / n_words) * 100


def run_manipulation_check(ground_truth_df):
    rows = []
    for _, row in ground_truth_df.iterrows():
        for tone_name, col in TEXT_COLUMNS.items():
            text = row.get(col, "")
            if pd.isna(text) or str(text).strip() == "":
                continue
            rows.append({
                "paper_id": row["paper_id"],
                "tone": tone_name,
                "hedge_density": marker_density(text, HEDGE_PATTERN),
                "assertive_density": marker_density(text, ASSERTIVE_PATTERN),
                "hype_density": marker_density(text, HYPE_PATTERN),
            })
    marker_df = pd.DataFrame(rows)
    summary = marker_df.groupby("tone")[["hedge_density", "assertive_density", "hype_density"]] \
        .agg(["mean", "std"]).round(3)

    checks = {}
    pivot = marker_df.pivot_table(index="paper_id", columns="tone", values="hedge_density")
    if "hedged" in pivot.columns and "original" in pivot.columns:
        paired = pivot[["original", "hedged"]].dropna()
        if len(paired) >= 3:
            stat, p = wilcoxon(paired["hedged"], paired["original"])
            checks["hedged_vs_original_hedge_density"] = {
                "n": len(paired), "mean_original": paired["original"].mean(),
                "mean_hedged": paired["hedged"].mean(), "wilcoxon_p": p
            }

    pivot_hype = marker_df.pivot_table(index="paper_id", columns="tone", values="hype_density")
    if "overclaiming" in pivot_hype.columns and "original" in pivot_hype.columns:
        paired = pivot_hype[["original", "overclaiming"]].dropna()
        if len(paired) >= 3:
            stat, p = wilcoxon(paired["overclaiming"], paired["original"])
            checks["overclaiming_vs_original_hype_density"] = {
                "n": len(paired), "mean_original": paired["original"].mean(),
                "mean_overclaiming": paired["overclaiming"].mean(), "wilcoxon_p": p
            }

    return marker_df, summary, checks

def run_content_invariance_check(ground_truth_df):
    rows = []
    for _, row in ground_truth_df.iterrows():
        original = str(row.get("abstract", ""))
        if not original.strip():
            continue
        for tone_name, col in TONE_ONLY.items():
            variant = str(row.get(col, ""))
            if not variant.strip():
                continue
            try:
                vec = TfidfVectorizer().fit([original, variant])
                vectors = vec.transform([original, variant])
                sim = cosine_similarity(vectors[0], vectors[1])[0][0]
            except Exception:
                sim = np.nan
            rows.append({"paper_id": row["paper_id"], "tone": tone_name, "cosine_similarity": sim})
    sim_df = pd.DataFrame(rows)
    summary = sim_df.groupby("tone")["cosine_similarity"].agg(["mean", "std", "min", "max"]).round(4)
    return sim_df, summary


def krippendorff_alpha_interval(matrix):
    matrix = np.array(matrix, dtype=float)
    n_units = matrix.shape[0]

    do_diffs = []
    for u in range(n_units):
        vals = matrix[u, :]
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 2:
            do_diffs.extend(pdist(vals.reshape(-1, 1), metric="sqeuclidean"))
    if len(do_diffs) == 0:
        return np.nan
    Do = np.mean(do_diffs)

    all_vals = matrix[~np.isnan(matrix)]
    if len(all_vals) < 2:
        return np.nan
    De = np.mean(pdist(all_vals.reshape(-1, 1), metric="sqeuclidean"))
    if De == 0:
        return np.nan

    return 1 - (Do / De)


def run_baseline_reliability(cell_df):
    original_only = cell_df[cell_df["text_type"] == "original"]
    matrix = original_only.pivot_table(index="paper_id", columns="model_name", values="mean_score")
    alpha = krippendorff_alpha_interval(matrix.values)

    # face-validity: does baseline LLM score correlate with real human reviewer score at all
    corr_rows = []
    for model in matrix.columns:
        model_scores = original_only[original_only["model_name"] == model][["paper_id", "mean_score"]]
        merged = model_scores.merge(
            original_only[["paper_id", "mean_human_score"]].drop_duplicates(), on="paper_id"
        )
        merged = merged.dropna()
        if len(merged) >= 3:
            corr = merged["mean_score"].corr(merged["mean_human_score"])
        else:
            corr = np.nan
        corr_rows.append({"model_name": model, "n_papers": len(merged), "corr_with_human_score": corr})

    return alpha, matrix, pd.DataFrame(corr_rows)


def compute_verdict_switch_rate(cell_df):
    orig = cell_df[cell_df["text_type"] == "original"][
        ["paper_id", "model_name", "majority_decision", "quality_tier"]
    ].rename(columns={"majority_decision": "baseline_decision"})

    results = []
    for tone in TONE_ONLY.keys():
        tone_df = cell_df[cell_df["text_type"] == tone][["paper_id", "model_name", "majority_decision"]] \
            .rename(columns={"majority_decision": "tone_decision"})
        merged = orig.merge(tone_df, on=["paper_id", "model_name"])

        # headline: baseline correctly reads REJECT (matches real verdict), tone flips to ACCEPT
        reject_tier = merged[merged["quality_tier"] == "reject_tier"]
        baseline_correct = reject_tier[reject_tier["baseline_decision"] == "REJECT"]
        n_eligible = len(baseline_correct)
        n_switched = (baseline_correct["tone_decision"] == "ACCEPT").sum()
        switch_rate = n_switched / n_eligible if n_eligible > 0 else np.nan

        # per-model breakdown
        per_model = baseline_correct.groupby("model_name").apply(
            lambda g: pd.Series({
                "n_eligible": len(g),
                "n_switched": (g["tone_decision"] == "ACCEPT").sum(),
                "switch_rate": (g["tone_decision"] == "ACCEPT").mean() if len(g) > 0 else np.nan,
            })
        ).reset_index()

        results.append({
            "tone": tone, "n_eligible": n_eligible, "n_switched": n_switched,
            "verdict_switch_rate": switch_rate, "per_model": per_model
        })

    # baseline agreement rate: how often original-condition decision matches real venue verdict
    orig_with_verdict = orig.merge(
        cell_df[["paper_id", "verdict"]].drop_duplicates(), on="paper_id"
    )
    orig_with_verdict["verdict_upper"] = orig_with_verdict["verdict"].str.strip().str.upper()
    agreement = (orig_with_verdict["baseline_decision"] == orig_with_verdict["verdict_upper"]).mean()

    return results, agreement


def run_mcnemar_single(baseline, comparison):
    paired = [(b, c) for b, c in zip(baseline, comparison) if b in ("ACCEPT", "REJECT") and c in ("ACCEPT", "REJECT")]
    if len(paired) < 2:
        return None
    reject_to_accept = sum(1 for b, c in paired if b == "REJECT" and c == "ACCEPT")
    accept_to_reject = sum(1 for b, c in paired if b == "ACCEPT" and c == "REJECT")
    both_accept = sum(1 for b, c in paired if b == "ACCEPT" and c == "ACCEPT")
    both_reject = sum(1 for b, c in paired if b == "REJECT" and c == "REJECT")

    n_discordant = reject_to_accept + accept_to_reject
    if n_discordant == 0:
        return {"n_pairs": len(paired), "reject_to_accept": 0, "accept_to_reject": 0,
                "n_discordant": 0, "statistic": np.nan, "pvalue": 1.0, "odds_ratio": np.nan,
                "exact_used": True}

    use_exact = n_discordant < 25
    table = [[both_accept, accept_to_reject], [reject_to_accept, both_reject]]
    result = mcnemar(table, exact=use_exact, correction=not use_exact)
    odds_ratio = (reject_to_accept / accept_to_reject) if accept_to_reject > 0 else np.inf

    return {
        "n_pairs": len(paired), "reject_to_accept": reject_to_accept, "accept_to_reject": accept_to_reject,
        "n_discordant": n_discordant, "statistic": result.statistic, "pvalue": result.pvalue,
        "odds_ratio": odds_ratio, "exact_used": use_exact,
    }


def run_all_mcnemar_tests(cell_df):
    orig = cell_df[cell_df["text_type"] == "original"][
        ["paper_id", "model_name", "majority_decision", "quality_tier"]
    ].rename(columns={"majority_decision": "baseline_decision"})

    rows = []
    for tier_name, tier_filter in [("reject_tier", "reject_tier"), ("accept_tier", "accept_tier"), ("pooled", None)]:
        for model in cell_df["model_name"].unique():
            for tone in TONE_ONLY.keys():
                tone_df = cell_df[(cell_df["text_type"] == tone) & (cell_df["model_name"] == model)][
                    ["paper_id", "majority_decision"]
                ].rename(columns={"majority_decision": "tone_decision"})
                base = orig[orig["model_name"] == model]
                if tier_filter:
                    base = base[base["quality_tier"] == tier_filter]
                merged = base.merge(tone_df, on="paper_id")
                res = run_mcnemar_single(merged["baseline_decision"], merged["tone_decision"])
                if res:
                    res.update({"tier": tier_name, "model_name": model, "tone": tone})
                    rows.append(res)
    return pd.DataFrame(rows)


def run_paired_score_comparison(original_scores, tone_scores):
    paired = [(o, t) for o, t in zip(original_scores, tone_scores) if not (pd.isna(o) or pd.isna(t))]
    if len(paired) < 3:
        return None
    orig = np.array([p[0] for p in paired])
    tone = np.array([p[1] for p in paired])
    diffs = tone - orig

    shapiro_p = shapiro(diffs)[1] if len(np.unique(diffs)) > 1 else np.nan
    t_p = ttest_rel(tone, orig)[1] if np.std(diffs) > 0 else np.nan
    try:
        w_p = wilcoxon(tone, orig)[1] if np.any(diffs != 0) else 1.0
    except ValueError:
        w_p = np.nan

    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else np.nan
    cohens_d = mean_diff / sd_diff if sd_diff and sd_diff > 0 else np.nan

    return {
        "n_pairs": len(paired), "mean_original": np.mean(orig), "mean_tone": np.mean(tone),
        "mean_diff": mean_diff, "cohens_d": cohens_d, "shapiro_p": shapiro_p,
        "paired_t_p": t_p, "wilcoxon_p": w_p,
    }


def run_all_score_tests(cell_df):
    orig = cell_df[cell_df["text_type"] == "original"][["paper_id", "model_name", "mean_score", "quality_tier"]] \
        .rename(columns={"mean_score": "original_score"})

    rows = []
    for scope_name, scope_val in [("pooled", None)] + [(m, m) for m in cell_df["model_name"].unique()]:
        for tone in TONE_ONLY.keys():
            tone_df = cell_df[cell_df["text_type"] == tone][["paper_id", "model_name", "mean_score"]]
            merged = orig.merge(tone_df, on=["paper_id", "model_name"])
            if scope_val:
                merged = merged[merged["model_name"] == scope_val]
            res = run_paired_score_comparison(merged["original_score"], merged["mean_score"])
            if res:
                res.update({"scope": scope_name, "tone": tone})
                rows.append(res)
    return pd.DataFrame(rows)


def run_mixed_effects_model(cell_df):
    df = cell_df.dropna(subset=["mean_score", "quality_tier"]).copy()
    df["text_type"] = pd.Categorical(
        df["text_type"],
        categories=["original", "hedged", "assertive", "overclaiming", "native_fluent", "l2_grounded"]
    )

    try:
        model = smf.mixedlm(
            "mean_score ~ C(text_type, Treatment(reference='original')) + C(quality_tier)",
            data=df, groups=df["paper_id"],
            vc_formula={"model_name": "0 + C(model_name)"},
        )
        result = model.fit(reml=True)
        return {"success": True, "fallback_used": False, "result": result}
    except Exception as e1:
        print(f"  Full crossed-random-effects model failed to converge: {e1}")
        print("  Falling back to simpler model (paper_id random intercept only)...")
        try:
            model = smf.mixedlm(
                "mean_score ~ C(text_type, Treatment(reference='original')) + C(quality_tier)",
                data=df, groups=df["paper_id"],
            )
            result = model.fit(reml=True)
            return {"success": True, "fallback_used": True, "error_from_full_model": str(e1), "result": result}
        except Exception as e2:
            return {"success": False, "error": str(e2)}



def run_quality_tier_interaction(cell_df):
    orig = cell_df[cell_df["text_type"] == "original"][["paper_id", "model_name", "mean_score", "quality_tier"]] \
        .rename(columns={"mean_score": "original_score"})
    rows = []
    for tone in TONE_ONLY.keys():
        tone_df = cell_df[cell_df["text_type"] == tone][["paper_id", "model_name", "mean_score"]]
        merged = orig.merge(tone_df, on=["paper_id", "model_name"])
        for tier in ["reject_tier", "accept_tier"]:
            subset = merged[merged["quality_tier"] == tier]
            res = run_paired_score_comparison(subset["original_score"], subset["mean_score"])
            if res:
                res.update({"tone": tone, "quality_tier": tier})
                rows.append(res)
    return pd.DataFrame(rows)


def run_model_heterogeneity(score_tests_df, mcnemar_df):
    score_by_model = score_tests_df[score_tests_df["scope"] != "pooled"].rename(columns={"scope": "model_name"})
    mcnemar_pooled = mcnemar_df[mcnemar_df["tier"] == "reject_tier"]
    return score_by_model, mcnemar_pooled


def apply_fdr_correction(pvalue_records):
    """pvalue_records: list of dicts each containing a 'pvalue' key. Adds 'pvalue_fdr' and 'significant_fdr'."""
    valid_idx = [i for i, r in enumerate(pvalue_records) if r.get("pvalue") is not None and not pd.isna(r.get("pvalue"))]
    if not valid_idx:
        return pvalue_records
    pvals = [pvalue_records[i]["pvalue"] for i in valid_idx]
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=FDR_ALPHA, method="fdr_bh")
    for j, i in enumerate(valid_idx):
        pvalue_records[i]["pvalue_fdr"] = pvals_corrected[j]
        pvalue_records[i]["significant_fdr"] = bool(reject[j])
    return pvalue_records


def compute_observed_power(cohens_d, n_pairs, alpha=0.05):
    if cohens_d is None or pd.isna(cohens_d) or n_pairs < 2:
        return np.nan
    try:
        return TTestPower().power(effect_size=abs(cohens_d), nobs=n_pairs, alpha=alpha, alternative="two-sided")
    except Exception:
        return np.nan


def make_score_by_tone_figure(cell_df):
    order = ["original", "hedged", "assertive", "overclaiming", "native_fluent", "l2_grounded"]
    means = cell_df.groupby("text_type")["mean_score"].mean().reindex(order)
    sems = cell_df.groupby("text_type")["mean_score"].sem().reindex(order)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(order, means.values, yerr=sems.values, capsize=4, color="#4C72B0")
    ax.set_ylabel("Mean LLM reviewer score (1-10)")
    ax.set_xlabel("Condition")
    ax.set_title("Mean score by condition, pooled across all reviewer models")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure_score_by_tone.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_verdict_switch_by_model_figure(verdict_switch_results):
    rows = []
    for r in verdict_switch_results:
        for _, mr in r["per_model"].iterrows():
            rows.append({"tone": r["tone"], "model_name": mr["model_name"], "switch_rate": mr["switch_rate"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    pivot = df.pivot(index="model_name", columns="tone", values="switch_rate")

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Verdict switch rate (reject to accept)")
    ax.set_xlabel("Reviewer model")
    ax.set_title("Verdict switch rate by model and tone")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Tone", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure_verdict_switch_by_model.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_workbook(sheets_dict, path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets_dict.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.copy()
                    df.columns = ["_".join(str(c) for c in col if c) for col in df.columns]
                df.to_excel(writer, sheet_name=name[:31], index=False)
            except Exception as e:
                print(f"    WARNING: sheet '{name}' failed to write, skipping it: {e}")


def write_summary_report(context, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Results Summary\n\n")
        f.write("Auto-generated from analyze_results.py. Verify every number against the workbook before quoting it in the paper.\n\n")

        f.write("## Data quality\n\n")
        f.write(f"- Total logged calls: {context.get('n_total_calls', 'N/A')}\n")
        f.write(f"- Errored calls: {context.get('n_errors', 'N/A')}\n\n")

        f.write("## Baseline reliability\n\n")
        f.write(f"- Krippendorff's alpha (interval, across 9 models, original abstracts): {context.get('krippendorff_alpha', 'N/A'):.3f}\n")
        f.write(f"- Baseline decision agreement with real venue verdict: {context.get('baseline_agreement', 'N/A'):.3f}\n\n")

        f.write("## Headline result: Verdict Switch Rate\n\n")
        for r in context.get("verdict_switch_results", []):
            f.write(f"- **{r['tone']}**: {r['n_switched']}/{r['n_eligible']} correctly-identified rejections flipped to accept "
                    f"({r['verdict_switch_rate']*100:.1f}%)\n" if r["n_eligible"] > 0 else f"- **{r['tone']}**: no eligible papers\n")
        f.write("\n")

        f.write("## Manipulation check\n\n")
        for name, check in context.get("manipulation_checks", {}).items():
            f.write(f"- {name}: n={check['n']}, wilcoxon p={check['wilcoxon_p']:.4g}\n")
        f.write("\n")

        f.write("## Content invariance (TF-IDF cosine similarity to original)\n\n")
        f.write(context.get("content_invariance_summary_str", "N/A") + "\n\n")

        f.write("## McNemar's exact test (reject tier, decision-level)\n\n")
        f.write("See mcnemar_results sheet in the workbook for the full table, per model, per tone.\n\n")

        f.write("## Paired score tests (Wilcoxon, pooled across models)\n\n")
        f.write("See score_tests sheet in the workbook. Includes Cohen's d and observed power per tone.\n\n")

        f.write("## Mixed-effects model\n\n")
        mm = context.get("mixed_model_result")
        if mm and mm.get("success"):
            f.write(f"Fallback used: {mm.get('fallback_used', False)}\n\n")
            f.write("```\n")
            f.write(str(mm["result"].summary()))
            f.write("\n```\n\n")
        else:
            f.write(f"Mixed model failed: {mm.get('error', 'unknown') if mm else 'not run'}\n\n")



def main():
    print("=" * 60)
    print("STEP 1: Loading and merging data")
    print("=" * 60)
    ground_truth_df = load_ground_truth(INPUT_XLSX_PATH)
    log_df = load_scoring_log(RESULTS_LOG_PATH)
    print(f"  Ground truth papers: {len(ground_truth_df)}")
    print(f"  Logged calls: {len(log_df)}")
    if len(log_df) < 16200:
        print(f"  WARNING: expected 16200 calls, found {len(log_df)}. The SLURM job may still be running.")

    errored = report_data_quality(log_df)
    cell_df = build_cell_level_df(log_df, ground_truth_df)
    cell_df.to_csv(os.path.join(OUTPUT_DIR, "merged_cell_level.csv"), index=False)

    context = {"n_total_calls": len(log_df), "n_errors": len(errored)}

    print("\n" + "=" * 60)
    print("STEP 2: Manipulation check")
    print("=" * 60)
    try:
        marker_df, marker_summary, manip_checks = run_manipulation_check(ground_truth_df)
        context["manipulation_checks"] = manip_checks
        print("  Done.")
        for k, v in manip_checks.items():
            print(f"    {k}: n={v['n']}, wilcoxon p={v['wilcoxon_p']:.4g}")
    except Exception as e:
        print(f"  FAILED: {e}")
        marker_df, marker_summary, manip_checks = pd.DataFrame(), pd.DataFrame(), {}
        context["manipulation_checks"] = {}

    print("\n" + "=" * 60)
    print("STEP 3: Content invariance check")
    print("=" * 60)
    try:
        sim_df, sim_summary = run_content_invariance_check(ground_truth_df)
        context["content_invariance_summary_str"] = sim_summary.to_string()
        print(sim_summary.to_string())
    except Exception as e:
        print(f"  FAILED: {e}")
        sim_df, sim_summary = pd.DataFrame(), pd.DataFrame()
        context["content_invariance_summary_str"] = "FAILED"

    print("\n" + "=" * 60)
    print("STEP 4: Baseline reliability")
    print("=" * 60)
    try:
        alpha, reliability_matrix, human_corr_df = run_baseline_reliability(cell_df)
        context["krippendorff_alpha"] = alpha
        print(f"  Krippendorff's alpha (interval): {alpha:.3f}")
        print(human_corr_df.to_string())
    except Exception as e:
        print(f"  FAILED: {e}")
        alpha, human_corr_df = np.nan, pd.DataFrame()
        context["krippendorff_alpha"] = np.nan

    print("\n" + "=" * 60)
    print("STEP 5: Verdict Switch Rate (headline metric)")
    print("=" * 60)
    try:
        verdict_switch_results, baseline_agreement = compute_verdict_switch_rate(cell_df)
        context["verdict_switch_results"] = verdict_switch_results
        context["baseline_agreement"] = baseline_agreement
        print(f"  Baseline agreement with real venue verdict: {baseline_agreement:.3f}")
        for r in verdict_switch_results:
            if r["n_eligible"] > 0:
                print(f"  {r['tone']:<15} {r['n_switched']}/{r['n_eligible']} switched  ({r['verdict_switch_rate']*100:.1f}%)")
            else:
                print(f"  {r['tone']:<15} no eligible papers")
    except Exception as e:
        print(f"  FAILED: {e}")
        verdict_switch_results, baseline_agreement = [], np.nan
        context["verdict_switch_results"] = []
        context["baseline_agreement"] = np.nan

    print("\n" + "=" * 60)
    print("STEP 6: McNemar's exact test (RQ1)")
    print("=" * 60)
    try:
        mcnemar_df = run_all_mcnemar_tests(cell_df)
        mcnemar_records = mcnemar_df.to_dict("records")
        mcnemar_records = apply_fdr_correction(mcnemar_records)
        mcnemar_df = pd.DataFrame(mcnemar_records)
        print(f"  {len(mcnemar_df)} McNemar tests run across tiers, models, tones.")
        sig = mcnemar_df[(mcnemar_df["significant_fdr"] == True) & (mcnemar_df["tier"] == "reject_tier")]
        print(f"  Significant after FDR correction (reject tier): {len(sig)}")
    except Exception as e:
        print(f"  FAILED: {e}")
        mcnemar_df = pd.DataFrame()

    print("\n" + "=" * 60)
    print("STEP 7: Paired score tests (RQ2/RQ3)")
    print("=" * 60)
    try:
        score_tests_df = run_all_score_tests(cell_df)
        records = score_tests_df.rename(columns={"wilcoxon_p": "pvalue"}).to_dict("records")
        records = apply_fdr_correction(records)
        score_tests_df = pd.DataFrame(records).rename(columns={"pvalue": "wilcoxon_p"})
        score_tests_df["observed_power"] = score_tests_df.apply(
            lambda r: compute_observed_power(r["cohens_d"], r["n_pairs"]), axis=1
        )
        print(score_tests_df[score_tests_df["scope"] == "pooled"][
            ["tone", "n_pairs", "mean_diff", "cohens_d", "wilcoxon_p", "pvalue_fdr", "observed_power"]
        ].to_string(index=False))
    except Exception as e:
        print(f"  FAILED: {e}")
        score_tests_df = pd.DataFrame()

    print("\n" + "=" * 60)
    print("STEP 8: Mixed-effects model (confirmatory)")
    print("=" * 60)
    try:
        mixed_result = run_mixed_effects_model(cell_df)
        context["mixed_model_result"] = mixed_result
        if mixed_result["success"]:
            print(f"  Success (fallback used: {mixed_result.get('fallback_used', False)})")
            print(mixed_result["result"].summary())
        else:
            print(f"  FAILED: {mixed_result['error']}")
    except Exception as e:
        print(f"  FAILED: {e}")
        mixed_result = {"success": False, "error": str(e)}
        context["mixed_model_result"] = mixed_result

    print("\n" + "=" * 60)
    print("STEP 9: Quality tier interaction (RQ4)")
    print("=" * 60)
    try:
        interaction_df = run_quality_tier_interaction(cell_df)
        print(interaction_df[["tone", "quality_tier", "n_pairs", "mean_diff", "cohens_d"]].to_string(index=False))
    except Exception as e:
        print(f"  FAILED: {e}")
        interaction_df = pd.DataFrame()

    print("\n" + "=" * 60)
    print("STEP 10: Model heterogeneity (RQ5)")
    print("=" * 60)
    try:
        model_scores_df, model_mcnemar_df = run_model_heterogeneity(score_tests_df, mcnemar_df)
        print("  Done. See model_heterogeneity_scores / model_heterogeneity_mcnemar sheets.")
    except Exception as e:
        print(f"  FAILED: {e}")
        model_scores_df, model_mcnemar_df = pd.DataFrame(), pd.DataFrame()

    print("\n" + "=" * 60)
    print("STEP 11: Figures")
    print("=" * 60)
    try:
        p1 = make_score_by_tone_figure(cell_df)
        print(f"  Saved {p1}")
    except Exception as e:
        print(f"  Score-by-tone figure FAILED: {e}")
    try:
        p2 = make_verdict_switch_by_model_figure(verdict_switch_results)
        print(f"  Saved {p2}")
    except Exception as e:
        print(f"  Verdict-switch figure FAILED: {e}")

    print("\n" + "=" * 60)
    print("STEP 12: Writing outputs")
    print("=" * 60)
    try:
        vsr_flat = []
        for r in verdict_switch_results:
            row = {k: v for k, v in r.items() if k != "per_model"}
            vsr_flat.append(row)
        vsr_df = pd.DataFrame(vsr_flat)

        per_model_flat = []
        for r in verdict_switch_results:
            for _, mr in r["per_model"].iterrows():
                per_model_flat.append({"tone": r["tone"], **mr.to_dict()})
        vsr_per_model_df = pd.DataFrame(per_model_flat)

        sheets = {
            "verdict_switch_rate": vsr_df,
            "verdict_switch_per_model": vsr_per_model_df,
            "mcnemar_results": mcnemar_df,
            "score_tests": score_tests_df,
            "quality_tier_interaction": interaction_df,
            "model_heterogeneity_scores": model_scores_df,
            "model_heterogeneity_mcnemar": model_mcnemar_df,
            "manipulation_check_summary": marker_summary.reset_index() if not marker_summary.empty else pd.DataFrame(),
            "content_invariance_summary": sim_summary.reset_index() if not sim_summary.empty else pd.DataFrame(),
            "baseline_human_corr": human_corr_df,
        }
        write_workbook(sheets, os.path.join(OUTPUT_DIR, "results_workbook.xlsx"))
        print(f"  Wrote {os.path.join(OUTPUT_DIR, 'results_workbook.xlsx')}")

        write_summary_report(context, os.path.join(OUTPUT_DIR, "summary_report.md"))
        print(f"  Wrote {os.path.join(OUTPUT_DIR, 'summary_report.md')}")
    except Exception as e:
        print(f"  FAILED to write final outputs: {e}")

    print("\n" + "=" * 60)
    print("DONE. Check analysis_output/ for everything.")
    print("=" * 60)


if __name__ == "__main__":
    main()