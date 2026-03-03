#!/usr/bin/env python3
"""
Evaluate an LLM-based interactive preference predictor (g_theta) on one or more
synthetic datasets produced by generate_dataset_llm.py.

This evaluator uses:
- Semantic memory (LTM): an LLM-generated summary updated every Δ meals
- Episodic memory (EM): embedding retrieval over past episodes (cached)
- Working memory (WM): current context + revealed corrections so far

It simulates user corrections by comparing predictions to the dataset's ground truth
and revealing one randomly chosen incorrect preference dimension at a time.

We evaluate the 19-dim preference bundle defined by config.PREFERENCE_BUNDLE
(exclude bite_ordering_preference).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Ensure repo root is on sys.path (for importing config, generate_dataset_llm, etc.)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import feeding_deployment.preference_learning.config as root_config  # type: ignore

try:
    from openai import OpenAI
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = Exception  # type: ignore[misc, assignment]
    print(
        "Error: openai package not installed. Install it with: pip install openai",
        file=sys.stderr,
    )
    raise

T = TypeVar("T")


def _retry_on_rate_limit(
    fn: Callable[[], T],
    max_retries: int = 5,
    base_wait: float = 60.0,
) -> T:
    """Call fn(); on RateLimitError (429), wait and retry with exponential backoff."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except OpenAIRateLimitError as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
            wait = base_wait * (2 ** attempt)
            print(f"  [Rate limit] Waiting {wait:.0f}s before retry ({attempt + 1}/{max_retries}) ...", flush=True)
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


PREF_FIELDS: List[str] = [name for (name, _, _) in root_config.PREFERENCE_BUNDLE]  # 19 dims
PREF_OPTIONS: Dict[str, List[str]] = {name: opts for (name, _, opts) in root_config.PREFERENCE_BUNDLE}


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if "```" not in s:
        return s
    if "```json" in s:
        return s.split("```json", 1)[1].split("```", 1)[0].strip()
    return s.split("```", 1)[1].split("```", 1)[0].strip()


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


class EmbeddingCache:
    """Simple on-disk embedding cache keyed by sha256(text)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: Dict[str, List[float]] = {}
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, list):
                            self._cache[k] = [float(x) for x in v]
            except Exception:
                self._cache = {}

    def key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        return self._cache.get(self.key(text))

    def set(self, text: str, emb: List[float]) -> None:
        self._cache[self.key(text)] = emb

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)


def _resolve_api_key(cli_key: Optional[str]) -> str:
    if cli_key:
        return cli_key

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    # Fallback: use the key in generate_dataset_llm.py (user approved for experiments).
    try:
        import generate_dataset_llm  # type: ignore

        file_key = getattr(generate_dataset_llm, "OPENAI_API_KEY", None)
        if isinstance(file_key, str) and file_key.strip():
            return file_key.strip()
    except Exception:
        pass

    raise RuntimeError(
        "OpenAI API key not found. Set OPENAI_API_KEY env var or pass --api-key."
    )


def _get_meal_info(meal: str) -> Dict[str, Any]:
    known_meal = meal in root_config.MEAL_STRUCTURE
    info = root_config.MEAL_STRUCTURE.get(meal, {})
    dippable = list(info.get("dippable_items", []) or [])
    sauces = list(info.get("sauces", []) or [])
    return {
        "known_meal": known_meal,
        "dippable_items": dippable,
        "sauces": sauces,
        "has_dippable": len(dippable) > 0,
        "has_sauce": len(sauces) > 0,
    }


def _apply_hard_rules(prefs: Dict[str, str], meal: str, corrected: Dict[str, str]) -> Dict[str, str]:
    out = dict(prefs)
    meal_info = _get_meal_info(meal)
    if out.get("transfer_mode") == "inside mouth transfer" and "outside_mouth_distance" not in corrected:
        out["outside_mouth_distance"] = "near"
    if (
        "bite_dipping_preference" not in corrected
        and meal_info.get("known_meal", False)
        and ((not meal_info["has_dippable"]) or (not meal_info["has_sauce"]))
    ):
        out["bite_dipping_preference"] = "do not dip"
    return out


def _episode_text(day: int, context: Dict[str, Any], prefs: Dict[str, str]) -> str:
    ctx = (
        f"day={day}; meal={context.get('meal')}; setting={context.get('setting')}; "
        f"time_of_day={context.get('time_of_day')};"
    )
    pref_str = "; ".join(f"{k}={prefs.get(k,'')}" for k in PREF_FIELDS)
    return f"{ctx}\npreferences: {pref_str}"


def _query_text(context: Dict[str, Any], corrected: Dict[str, str]) -> str:
    ctx = (
        f"meal={context.get('meal')}; setting={context.get('setting')}; "
        f"time_of_day={context.get('time_of_day')};"
    )
    corr = "; ".join(f"{k}={v}" for k, v in sorted(corrected.items())) if corrected else "none"
    return f"{ctx}\ncorrected_so_far: {corr}"


def _summarize_ltm(
    client: OpenAI,
    model: str,
    physical_profile: str,
    previous_ltm_summary: str,
    past_episode_snippets: List[str],
) -> str:
    opts_block = "\n".join(f"- {field}: [{', '.join(PREF_OPTIONS[field])}]" for field in PREF_FIELDS)

    if not previous_ltm_summary.strip():
        # Cold start (boundary=0): generate initial defaults from physical_profile alone.
        prompt = f"""You are helping a feeding-assistance robot maintain a concise semantic memory of a user's dining preferences.\n\nUSER PHYSICAL CAPABILITY PROFILE:\n{physical_profile}\n\nThere are no past episodes yet. Propose reasonable initial defaults for this user's preferences based only on the physical capability profile.\n\nAllowed options per dimension (only use these exact strings when you mention concrete values):\n{opts_block}\n\nWrite a compact summary (6-10 bullet points max). Keep it short; this summary will be used as a prior for the next meal.\n"""
        max_tokens = 350
    else:
        # Update mode: incorporate new episodes into the previous summary.
        new_episodes_block = "\n\n".join(past_episode_snippets) if past_episode_snippets else "(no new episodes in this window)"
        prompt = f"""You are helping a feeding-assistance robot maintain a concise semantic memory of a user's dining preferences.\n\nUSER PHYSICAL CAPABILITY PROFILE:\n{physical_profile}\n\nPREVIOUS SUMMARY:\n{previous_ltm_summary}\n\nNEW EPISODES SINCE THAT SUMMARY (context + final accepted preferences):\n{new_episodes_block}\n\nTask: Update the summary to incorporate the new evidence above. When the same context produced different preference bundles across episodes, preserve both patterns as context-conditional variation — do not average or discard either. Flag dimensions where the user shows variability, as these likely depend on unobserved internal states. Retain everything from the previous summary unless directly contradicted by new evidence.\n\nAllowed options per dimension (only use these exact strings when you mention concrete values):\n{opts_block}\n\nKeep the summary to 6-10 bullet points normally; you may use up to ~12 if there are genuine conditional branches to preserve. This summary will be used as a prior for the next meal.\n"""
        max_tokens = 500

    def _call() -> Any:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write concise, faithful preference summaries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )

    resp = _retry_on_rate_limit(_call)
    return (resp.choices[0].message.content or "").strip()


def _predict_bundle_llm(
    client: OpenAI,
    model: str,
    physical_profile: str,
    ltm_summary: str,
    retrieved_episodes: List[str],
    context: Dict[str, Any],
    corrected: Dict[str, str],
) -> Dict[str, str]:
    opts_block = "\n".join(f"- {field}: [{', '.join(PREF_OPTIONS[field])}]" for field in PREF_FIELDS)
    retrieved_block = "\n\n".join(retrieved_episodes) if retrieved_episodes else "(none)"
    corrected_block = "\n".join(f"- {k}: {v}" for k, v in corrected.items()) if corrected else "(none)"

    prompt = f"""You are a personalized preference predictor for a feeding-assistance robot.\n\nUSER PHYSICAL CAPABILITY PROFILE (U^phys):\n{physical_profile}\n\nSEMANTIC MEMORY SUMMARY (LTM prior):\n{ltm_summary}\n\nRETRIEVED SIMILAR PAST EPISODES (Episodic memory):\n{retrieved_block}\n\nCURRENT CONTEXT (X_t):\n- meal: {context.get('meal')}\n- setting: {context.get('setting')}\n- time_of_day: {context.get('time_of_day')}\n\nREVEALED USER CORRECTIONS SO FAR (Working memory):\n{corrected_block}\n\nYou must predict the user's FINAL accepted preference bundle for THIS meal.\n\nAllowed options per dimension (you MUST choose exactly one of the strings for each field):\n{opts_block}\n\nHARD RULES (must satisfy):\n- If transfer_mode is \"inside mouth transfer\", outside_mouth_distance MUST be \"near\".\n- If the meal has no dippable items OR no sauces, bite_dipping_preference MUST be \"do not dip\".\n\nOutput ONLY valid JSON with exactly these 19 keys:\n{', '.join(PREF_FIELDS)}\n"""

    def _call() -> Any:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return JSON only. No extra text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=650,
        )

    resp = _retry_on_rate_limit(_call)

    raw = (resp.choices[0].message.content or "").strip()
    raw = _strip_code_fences(raw)
    data = _safe_json_load(raw) or {}

    out: Dict[str, str] = {}
    for field in PREF_FIELDS:
        val = str(data.get(field, "")).strip()
        if val in PREF_OPTIONS[field]:
            out[field] = val
        else:
            out[field] = corrected.get(field, PREF_OPTIONS[field][0])

    out = _apply_hard_rules(out, meal=str(context.get("meal", "")), corrected=corrected)

    # Defensive: never allow post-processing to override explicit corrections.
    if corrected:
        for k, v in corrected.items():
            out[k] = v

    for field in PREF_FIELDS:
        if out[field] not in PREF_OPTIONS[field]:
            out[field] = PREF_OPTIONS[field][0]

    return out


def _extract_truth_bundle(day_rec: Dict[str, Any]) -> Dict[str, str]:
    prefs = day_rec.get("preferences", {}) or {}
    out: Dict[str, str] = {}
    for field in PREF_FIELDS:
        val = prefs.get(field, {})
        if isinstance(val, dict):
            out[field] = str(val.get("choice", "")).strip()
        else:
            out[field] = ""
    return out


def _embed_text(client: OpenAI, embed_model: str, cache: EmbeddingCache, text: str) -> List[float]:
    cached = cache.get(text)
    if cached is not None:
        return cached

    def _call() -> Any:
        return client.embeddings.create(model=embed_model, input=text)

    resp = _retry_on_rate_limit(_call)
    emb = [float(x) for x in resp.data[0].embedding]
    cache.set(text, emb)
    return emb


def _retrieve_episodes(
    client: OpenAI,
    embed_model: str,
    cache: EmbeddingCache,
    history_texts: List[str],
    query: str,
    k: int,
) -> List[str]:
    if not history_texts or k <= 0:
        return []
    q_emb = _embed_text(client, embed_model, cache, query)
    scored: List[Tuple[float, str]] = []
    for txt in history_texts:
        e_emb = _embed_text(client, embed_model, cache, txt)
        scored.append((_cosine_sim(q_emb, e_emb), txt))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [t[1] for t in scored[:k]]


def _generate_plots(user_reports: List[Dict[str, Any]], report_dir: Path) -> None:
    """Generate five plots and save them in report_dir."""
    import numpy as np

    # Aggregate per-day data across users (for mean ± std)
    day_to_acc_m0: Dict[int, List[float]] = defaultdict(list)
    day_to_acc_m1: Dict[int, List[float]] = defaultdict(list)
    day_to_m_star: Dict[int, List[float]] = defaultdict(list)
    for ur in user_reports:
        for rec in ur.get("per_day_metrics", []):
            d = rec["day"]
            if "acc_m0" in rec:
                day_to_acc_m0[d].append(rec["acc_m0"])
            if "acc_m1" in rec:
                day_to_acc_m1[d].append(rec["acc_m1"])
            if "m_star" in rec:
                day_to_m_star[d].append(rec["m_star"])

    days = sorted(set(day_to_acc_m0.keys()) | set(day_to_acc_m1.keys()) | set(day_to_m_star.keys()))
    if not days:
        print("No per-day data for plots.")
        return

    # Plot A: Learning curve — Acc(t, m=0) vs day
    fig, ax = plt.subplots(figsize=(8, 5))
    acc_m0_means = [np.mean(day_to_acc_m0.get(d, [0])) for d in days]
    acc_m0_stds = [np.std(day_to_acc_m0.get(d, [0])) if len(day_to_acc_m0.get(d, [])) > 1 else 0 for d in days]
    ax.plot(days, acc_m0_means, "b-o", markersize=4, label="Acc(t, m=0)")
    ax.fill_between(days, np.subtract(acc_m0_means, acc_m0_stds), np.add(acc_m0_means, acc_m0_stds), alpha=0.2)
    ax.set_xlabel("Day")
    ax.set_ylabel("Accuracy (unrevealed dims)")
    ax.set_title("Plot A: Learning Curve — Acc(t, m=0) vs Day")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "plot_a_learning_curve.png", dpi=150)
    plt.close()

    # Plot B: Correction burden — m_t* vs day
    fig, ax = plt.subplots(figsize=(8, 5))
    m_star_means = [np.mean(day_to_m_star.get(d, [0])) for d in days]
    m_star_stds = [np.std(day_to_m_star.get(d, [0])) if len(day_to_m_star.get(d, [])) > 1 else 0 for d in days]
    ax.plot(days, m_star_means, "g-o", markersize=4, label="m_t* (corrections to stop)")
    ax.fill_between(days, np.subtract(m_star_means, m_star_stds), np.add(m_star_means, m_star_stds), alpha=0.2)
    ax.set_xlabel("Day")
    ax.set_ylabel("Mean corrections to stop")
    ax.set_title("Plot B: Correction Burden — m_t* vs Day")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "plot_b_correction_burden.png", dpi=150)
    plt.close()

    # Plot C: Autocomplete gain — Acc(t, m=0) and Acc(t, m=1) vs day
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(days, acc_m0_means, "b-o", markersize=4, label="Acc(t, m=0)")
    acc_m1_means = [np.mean(day_to_acc_m1.get(d, [0])) for d in days]
    ax.plot(days, acc_m1_means, "r-s", markersize=4, label="Acc(t, m=1)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Accuracy (unrevealed dims)")
    ax.set_title("Plot C: Autocomplete Gain — Acc(t, m=0) vs Acc(t, m=1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "plot_c_autocomplete_gain.png", dpi=150)
    plt.close()

    # Plot D: Accuracy by affective state
    state_to_acc: Dict[str, List[float]] = defaultdict(list)
    state_to_m_star: Dict[str, List[float]] = defaultdict(list)
    for ur in user_reports:
        for state, vals in ur.get("by_affective_state", {}).items():
            if "acc_m0" in vals:
                state_to_acc[state].append(vals["acc_m0"])
            if "mean_m_star" in vals:
                state_to_m_star[state].append(vals["mean_m_star"])
    states = sorted(set(state_to_acc.keys()) | set(state_to_m_star.keys()))
    if states:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        acc_means = [np.mean(state_to_acc.get(s, [0])) for s in states]
        m_star_means_d = [np.mean(state_to_m_star.get(s, [0])) for s in states]
        x = np.arange(len(states))
        w = 0.35
        ax1.bar(x - w / 2, acc_means, w, label="Acc(m=0)", color="steelblue")
        ax1.set_xticks(x)
        ax1.set_xticklabels(states, rotation=45, ha="right")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Acc(m=0) by Affective State")
        ax1.legend()
        ax2.bar(x + w / 2, m_star_means_d, w, label="Mean m*", color="coral")
        ax2.set_xticks(x)
        ax2.set_xticklabels(states, rotation=45, ha="right")
        ax2.set_ylabel("Mean corrections to stop")
        ax2.set_title("Mean m* by Affective State")
        ax2.legend()
        fig.suptitle("Plot D: Metrics by Affective State")
        fig.tight_layout()
        fig.savefig(report_dir / "plot_d_by_affective_state.png", dpi=150)
        plt.close()

    # Plot E: Per-dimension accuracy at m=0
    dim_acc: Dict[str, List[float]] = defaultdict(list)
    for ur in user_reports:
        for dim, acc in ur.get("per_dimension_m0_accuracy", {}).items():
            dim_acc[dim].append(acc)
    dim_final = {d: float(np.mean(v)) for d, v in dim_acc.items()}
    dims = list(dim_final.keys())
    if dims:
        fig, ax = plt.subplots(figsize=(10, 6))
        vals = [dim_final[d] for d in dims]
        ax.barh(dims, vals, color="teal", alpha=0.7)
        ax.set_xlabel("Accuracy at m=0")
        ax.set_title("Plot E: Per-Dimension Accuracy at m=0")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(report_dir / "plot_e_per_dimension_accuracy.png", dpi=150)
        plt.close()

    print(f"Saved plots to {report_dir}")


class _Tee:
    """Write to multiple file-like objects (e.g. stdout and a file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> None:
        for st in self._streams:
            st.write(s)
            st.flush()

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate LLM-based interactive predictor on synthetic datasets."
    )
    p.add_argument("--data-file", help="Path to one JSON dataset file.")
    p.add_argument("--data-dir", help="Directory containing JSON dataset files.")
    p.add_argument("--k-retrieve", type=int, default=10, help="Top-K episodes to retrieve (default: 10).")
    p.add_argument("--delta", type=int, default=5, help="Update semantic memory every Δ meals (default: 5).")
    p.add_argument("--max-corrections", type=int, default=19, help="Max simulated corrections per meal (default: 19).")
    p.add_argument(
        "--max-meals",
        type=int,
        default=0,
        help="Max meals per user to evaluate (0 = all, default: 0).",
    )
    p.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of Monte Carlo rollouts (different correction trajectories) per user (default: 1).",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for correction selection (default: 0).")
    p.add_argument("--openai-model", default="gpt-4o", help="Chat model (default: gpt-4o).")
    p.add_argument("--embed-model", default="text-embedding-3-small", help="Embedding model (default: text-embedding-3-small).")
    p.add_argument("--api-key", default = "", help="OpenAI API key (optional).")
    p.add_argument("--cache-dir", default=".cache/llm_memory_eval", help="Cache directory (default: .cache/llm_memory_eval).")
    p.add_argument(
        "--progress",
        choices=["none", "meal", "step"],
        default="step",
        help=(
            "Progress printing level (default: step). "
            "'meal' prints once per meal; 'step' prints per correction step."
        ),
    )
    p.add_argument(
        "--ablation",
        choices=["full", "ltm_only", "em_only", "no_memory"],
        default="full",
        help=(
            "Memory ablation: full (all components), ltm_only (no episodic retrieval), "
            "em_only (no LTM prior), no_memory (no LTM, no EM). Default: full."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    files: List[str] = []
    if args.data_file:
        files.append(args.data_file)
    if args.data_dir:
        files.extend(sorted(glob.glob(os.path.join(args.data_dir, "*.json"))))
    if not files:
        print("No input files provided. Use --data-file or --data-dir.")
        return 1

    # Create timestamped report dir and tee stdout to report.txt for this run.
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports") / f"llm_eval_{run_ts}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_txt_path = report_dir / "report.txt"
    report_txt_file = open(report_txt_path, "w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, report_txt_file)

    try:
        if args.num_rollouts < 1:
            print("--num-rollouts must be >= 1", file=sys.stderr)
            return 1
        if args.max_corrections < 0:
            print("--max-corrections must be >= 0", file=sys.stderr)
            return 1

        client = OpenAI(api_key=_resolve_api_key(args.api_key))

        cache_dir = Path(args.cache_dir)
        emb_cache = EmbeddingCache(cache_dir / "embeddings.json")
        ltm_cache_path = cache_dir / "ltm_summaries.json"
        if ltm_cache_path.exists():
            try:
                with open(ltm_cache_path, "r", encoding="utf-8") as f:
                    ltm_cache: Dict[str, str] = json.load(f)
            except Exception:
                ltm_cache = {}
        else:
            ltm_cache = {}

        def ltm_cache_key(user: str, boundary: int, previous_ltm_summary: str) -> str:
            prev_hash = hashlib.sha256(previous_ltm_summary.encode()).hexdigest()[:8] if previous_ltm_summary.strip() else "cold"
            return f"{user}|b{boundary}|delta{args.delta}|prev_{prev_hash}"

        user_reports: List[Dict[str, Any]] = []

        for path in files:
            print(f"Evaluating {path} ...", flush=True)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            user = str(data.get("user", "unknown"))
            physical_profile = str(data.get("physical_capability_profile", "")).strip()
            # We intentionally do NOT use dataset-provided long_term_preferences as an LTM seed,
            # since that would leak ground truth U^pref into evaluation.

            days: List[Dict[str, Any]] = list(data.get("days", []))
            days.sort(key=lambda r: int(r.get("day", 0)))

            total_meals = 0  # counts meals across rollouts
            total_corrections = 0  # sum of m across rollouts
            acc_after_m_sum: Dict[int, float] = {}
            acc_after_m_n: Dict[int, int] = {}

            # Per-day metrics (for Plots A, B, C)
            acc_m0_sum_by_day: Dict[int, float] = defaultdict(float)
            acc_m0_n_by_day: Dict[int, int] = defaultdict(int)
            acc_m1_sum_by_day: Dict[int, float] = defaultdict(float)
            acc_m1_n_by_day: Dict[int, int] = defaultdict(int)
            m_star_sum_by_day: Dict[int, float] = defaultdict(float)
            m_star_n_by_day: Dict[int, int] = defaultdict(int)
            affective_state_by_day: Dict[int, str] = {}

            # By affective state (Plot D)
            acc_m0_sum_by_state: Dict[str, float] = defaultdict(float)
            acc_m0_n_by_state: Dict[str, int] = defaultdict(int)
            m_star_sum_by_state: Dict[str, float] = defaultdict(float)
            m_star_n_by_state: Dict[str, int] = defaultdict(int)

            # Per-dimension m0 accuracy (Plot E)
            per_dim_correct: Dict[str, int] = defaultdict(int)
            per_dim_total: Dict[str, int] = defaultdict(int)

            # For summary_statistics: zero-correction meal counts
            zero_correction_meals_total = 0
            zero_correction_meals_final_week = 0

            for rollout_idx in range(args.num_rollouts):
                rng = random.Random(args.seed + rollout_idx)

                # History is always the accepted final bundle (ground truth), so it does not depend
                # on the correction trajectory. We still rebuild it per rollout to keep the logic
                # simple and avoid subtle state bugs; embeddings are cached so this is cheap.
                history_texts: List[str] = []
                past_for_ltm: List[str] = []
                meals_this_rollout = 0
                previous_ltm_summary = ""

                for day_rec in days:
                    day = int(day_rec.get("day", 0))
                    ctx = day_rec.get("context", {}) or {}
                    truth = _extract_truth_bundle(day_rec)

                    boundary = max(0, (day - 1) // args.delta) * args.delta
                    # Episodes since last LTM update: current window only (not last 10).
                    new_episodes = past_for_ltm[-args.delta:] if boundary > 0 else []
                    key = ltm_cache_key(user, boundary, previous_ltm_summary)
                    if key in ltm_cache:
                        ltm_summary = ltm_cache[key]
                    else:
                        print(
                            f"  [LTM] Calling OpenAI for summary (day {day}, boundary={boundary}) ...",
                            flush=True,
                        )
                        if args.progress != "none":
                            print(
                                f"  [user={user}] rollout {rollout_idx+1}/{args.num_rollouts} "
                                f"day {day}: updating LTM summary (Δ={args.delta}, boundary={boundary}) ...",
                                flush=True,
                            )
                        ltm_summary = _summarize_ltm(
                            client=client,
                            model=args.openai_model,
                            physical_profile=physical_profile,
                            previous_ltm_summary=previous_ltm_summary,
                            past_episode_snippets=new_episodes,
                        )
                        ltm_cache[key] = ltm_summary
                    previous_ltm_summary = ltm_summary

                    corrected: Dict[str, str] = {}
                    m = 0
                    affective_state = str(ctx.get("transient_affective_state") or "unknown").strip() or "unknown"
                    affective_state_by_day[day] = affective_state

                    if args.progress == "meal":
                        print(
                            f"  [user={user}] rollout {rollout_idx+1}/{args.num_rollouts} "
                            f"day {day}: start meal; meal={ctx.get('meal')} setting={ctx.get('setting')} time={ctx.get('time_of_day')}",
                            flush=True,
                        )

                    while True:
                        query = _query_text(ctx, corrected)

                        # Decide whether to actually use episodic memory for this ablation.
                        use_em = args.ablation not in ("ltm_only", "no_memory")
                        retrieved: List[str] = []
                        if use_em and args.k_retrieve > 0 and history_texts:
                            print(
                                f"  [EM] Retrieving episodes (day {day}, m={m}, history={len(history_texts)}, k={args.k_retrieve}) ...",
                                flush=True,
                            )
                            retrieved = _retrieve_episodes(
                                client=client,
                                embed_model=args.embed_model,
                                cache=emb_cache,
                                history_texts=history_texts,
                                query=query,
                                k=args.k_retrieve,
                            )

                        # Ablation: optionally disable LTM and/or EM.
                        ltm_for_pred = ltm_summary if args.ablation not in ("em_only", "no_memory") else ""
                        retrieved_for_pred = retrieved if args.ablation not in ("ltm_only", "no_memory") else []

                        print(
                            f"  [Predict] Calling OpenAI for bundle (day {day}, m={m}) ...",
                            flush=True,
                        )
                        pred = _predict_bundle_llm(
                            client=client,
                            model=args.openai_model,
                            physical_profile=physical_profile,
                            ltm_summary=ltm_for_pred,
                            retrieved_episodes=retrieved_for_pred,
                            context=ctx,
                            corrected=corrected,
                        )

                        unrevealed = [f for f in PREF_FIELDS if f not in corrected]
                        if unrevealed:
                            correct = sum(1 for f in unrevealed if pred.get(f) == truth.get(f))
                            acc = correct / float(len(unrevealed))
                        else:
                            acc = 1.0

                        acc_after_m_sum[m] = acc_after_m_sum.get(m, 0.0) + acc
                        acc_after_m_n[m] = acc_after_m_n.get(m, 0) + 1

                        if m == 0:
                            acc_m0_sum_by_day[day] += acc
                            acc_m0_n_by_day[day] += 1
                            acc_m0_sum_by_state[affective_state] += acc
                            acc_m0_n_by_state[affective_state] += 1
                            for f in unrevealed:
                                per_dim_total[f] += 1
                                if pred.get(f) == truth.get(f):
                                    per_dim_correct[f] += 1
                        elif m == 1:
                            acc_m1_sum_by_day[day] += acc
                            acc_m1_n_by_day[day] += 1

                        mismatches = [
                            f for f in PREF_FIELDS
                            if f not in corrected and pred.get(f) != truth.get(f)
                        ]
                        if args.progress == "step":
                            print(
                                f"  [user={user}] rollout {rollout_idx+1}/{args.num_rollouts} "
                                f"day {day} m={m}: acc_unrevealed={acc:.3f} mismatches={len(mismatches)} "
                                f"(retrieved={len(retrieved)})",
                                flush=True,
                            )
                        if not mismatches or m >= args.max_corrections:
                            total_meals += 1
                            total_corrections += m
                            meals_this_rollout += 1
                            m_star_sum_by_day[day] += m
                            m_star_n_by_day[day] += 1
                            m_star_sum_by_state[affective_state] += m
                            m_star_n_by_state[affective_state] += 1
                            if m == 0 and not mismatches:
                                acc_m1_sum_by_day[day] += 1.0
                                acc_m1_n_by_day[day] += 1
                                zero_correction_meals_total += 1
                                if day >= 24:
                                    zero_correction_meals_final_week += 1
                            break

                        f_corr = rng.choice(mismatches)
                        corrected[f_corr] = truth.get(f_corr, "")
                        if args.progress == "step":
                            print(
                                f"    correcting {f_corr} -> {corrected[f_corr]}",
                                flush=True,
                            )
                        m += 1

                    ep_txt = _episode_text(day=day, context=ctx, prefs=truth)
                    history_texts.append(ep_txt)
                    past_for_ltm.append(ep_txt)

                    if args.max_meals and meals_this_rollout >= args.max_meals:
                        break

            mean_corr = (total_corrections / float(total_meals)) if total_meals else 0.0
            acc_after_m = {
                str(m): (acc_after_m_sum[m] / float(acc_after_m_n[m]))
                for m in sorted(acc_after_m_sum.keys())
            }

            # Build per-day metrics
            all_days = sorted(set(acc_m0_sum_by_day.keys()) | set(acc_m1_sum_by_day.keys()) | set(m_star_sum_by_day.keys()))
            per_day_metrics: List[Dict[str, Any]] = []
            for d in all_days:
                rec: Dict[str, Any] = {"day": d, "affective_state": affective_state_by_day.get(d, "unknown")}
                if acc_m0_n_by_day[d]:
                    rec["acc_m0"] = acc_m0_sum_by_day[d] / float(acc_m0_n_by_day[d])
                if acc_m1_n_by_day[d]:
                    rec["acc_m1"] = acc_m1_sum_by_day[d] / float(acc_m1_n_by_day[d])
                if m_star_n_by_day[d]:
                    rec["m_star"] = m_star_sum_by_day[d] / float(m_star_n_by_day[d])
                per_day_metrics.append(rec)

            # Build by affective state
            by_affective_state: Dict[str, Dict[str, float]] = {}
            for state in sorted(set(acc_m0_n_by_state.keys()) | set(m_star_n_by_state.keys())):
                by_affective_state[state] = {}
                if acc_m0_n_by_state[state]:
                    by_affective_state[state]["acc_m0"] = acc_m0_sum_by_state[state] / float(acc_m0_n_by_state[state])
                if m_star_n_by_state[state]:
                    by_affective_state[state]["mean_m_star"] = m_star_sum_by_state[state] / float(m_star_n_by_state[state])

            # Build per-dimension m0 accuracy
            per_dimension_m0_accuracy: Dict[str, float] = {
                f: (per_dim_correct[f] / float(per_dim_total[f]) if per_dim_total[f] else 0.0)
                for f in PREF_FIELDS
            }

            # Summary statistics (early vs late, zero-correction %, best/worst day)
            day_to_acc_m0 = {rec["day"]: rec["acc_m0"] for rec in per_day_metrics if "acc_m0" in rec}
            day_to_m_star = {rec["day"]: rec["m_star"] for rec in per_day_metrics if "m_star" in rec}
            early_days = [d for d in day_to_acc_m0 if 1 <= d <= 5]
            late_days = [d for d in day_to_acc_m0 if 26 <= d <= 30]
            mean_acc_m0_early = sum(day_to_acc_m0[d] for d in early_days) / len(early_days) if early_days else 0.0
            mean_acc_m0_late = sum(day_to_acc_m0[d] for d in late_days) / len(late_days) if late_days else 0.0
            delta_acc_m0 = mean_acc_m0_late - mean_acc_m0_early
            early_m_star = [day_to_m_star[d] for d in day_to_m_star if 1 <= d <= 5]
            late_m_star = [day_to_m_star[d] for d in day_to_m_star if 26 <= d <= 30]
            mean_m_star_early = sum(early_m_star) / len(early_m_star) if early_m_star else 0.0
            mean_m_star_late = sum(late_m_star) / len(late_m_star) if late_m_star else 0.0
            delta_m_star = mean_m_star_late - mean_m_star_early  # expect negative if learning
            pct_zero_overall = 100.0 * zero_correction_meals_total / total_meals if total_meals else 0.0
            meals_final_week = sum(m_star_n_by_day.get(d, 0) for d in range(24, 31))
            pct_zero_final_week = 100.0 * zero_correction_meals_final_week / meals_final_week if meals_final_week else 0.0
            acc_m0_values = list(day_to_acc_m0.values()) if day_to_acc_m0 else [0.0]
            best_acc_m0 = max(acc_m0_values)
            worst_acc_m0 = min(acc_m0_values)

            summary_statistics = {
                "mean_acc_m0_early_days_1_5": mean_acc_m0_early,
                "mean_acc_m0_late_days_26_30": mean_acc_m0_late,
                "delta_acc_m0_late_minus_early": delta_acc_m0,
                "mean_m_star_early_days_1_5": mean_m_star_early,
                "mean_m_star_late_days_26_30": mean_m_star_late,
                "delta_m_star_late_minus_early": delta_m_star,
                "pct_zero_correction_overall": pct_zero_overall,
                "pct_zero_correction_final_week_days_24_30": pct_zero_final_week,
                "best_single_day_acc_m0": best_acc_m0,
                "worst_single_day_acc_m0": worst_acc_m0,
            }

            print(
                f"  Mean corrections-to-stop: {mean_corr:.3f} over {total_meals} meals "
                f"({args.num_rollouts} rollouts)"
            )
            if "0" in acc_after_m:
                print(f"  Accuracy-after-m=0 (unrevealed): {acc_after_m['0']:.3f}")
            if "1" in acc_after_m:
                print(f"  Accuracy-after-m=1 (unrevealed): {acc_after_m['1']:.3f}")

            print("  Summary statistics:")
            print(f"    Acc(m=0): early (days 1-5) = {mean_acc_m0_early:.3f}, late (days 26-30) = {mean_acc_m0_late:.3f}, delta = {delta_acc_m0:+.3f}")
            print(f"    m*:       early (days 1-5) = {mean_m_star_early:.3f}, late (days 26-30) = {mean_m_star_late:.3f}, delta = {delta_m_star:+.3f}")
            print(f"    Zero-correction days: {pct_zero_overall:.1f}% overall, {pct_zero_final_week:.1f}% in final week (24-30)")
            print(f"    Best / worst single-day Acc(m=0): {best_acc_m0:.3f} / {worst_acc_m0:.3f}")

            user_reports.append(
                {
                    "file": path,
                    "user": user,
                    "meals": total_meals,
                    "num_rollouts": args.num_rollouts,
                    "mean_corrections_to_stop": mean_corr,
                    "accuracy_after_m": acc_after_m,
                    "per_day_metrics": per_day_metrics,
                    "by_affective_state": by_affective_state,
                    "per_dimension_m0_accuracy": per_dimension_m0_accuracy,
                    "summary_statistics": summary_statistics,
                }
            )

        emb_cache.flush()
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(ltm_cache_path, "w", encoding="utf-8") as f:
            json.dump(ltm_cache, f, ensure_ascii=False, indent=2)

        if len(user_reports) > 1:
            overall = sum(u["mean_corrections_to_stop"] for u in user_reports) / len(user_reports)
            print(f"\nOverall mean corrections-to-stop across {len(user_reports)} users: {overall:.3f}")

        report_path = report_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "users": user_reports,
                    "run_timestamp": run_ts,
                    "ablation": args.ablation,
                    "metrics_notes": (
                        "Acc(t, m=1): On days where m=0 was perfect (no corrections needed), "
                        "acc_m1 is recorded as 1.0 so Plot C uses the same set of days for both lines."
                    ),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nWrote report: {report_path}")
        print(f"Terminal output saved to: {report_txt_path}")

        if HAS_MATPLOTLIB:
            _generate_plots(user_reports, report_dir)
        else:
            print("Skipping plots (matplotlib not installed). Add matplotlib to requirements.txt to enable.")

        return 0
    finally:
        sys.stdout = real_stdout
        report_txt_file.close()


if __name__ == "__main__":
    raise SystemExit(main())

