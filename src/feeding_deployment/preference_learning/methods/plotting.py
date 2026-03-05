from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar
from collections import defaultdict
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def _generate_plots(user_reports: List[Dict[str, Any]], report_dir: Path) -> None:
    """Generate five plots and save them in report_dir."""
    
    if not HAS_MATPLOTLIB:
        print("matplotlib is required for plotting. Install it with: pip install matplotlib")
        return

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

