from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _compute_mean_sem_by_day(
    users: list[dict[str, Any]],
    value_fn,
) -> tuple[list[int], list[float], list[float]]:
    sum_by_day: dict[int, float] = {}
    sq_sum_by_day: dict[int, float] = {}
    n_by_day: dict[int, int] = {}

    for user_report in users:
        per_day_metrics = user_report.get("per_day_metrics", [])
        if not isinstance(per_day_metrics, list):
            continue

        for rec in per_day_metrics:
            day, value = value_fn(rec)
            if day is None or value is None:
                continue

            sum_by_day[day] = sum_by_day.get(day, 0.0) + value
            sq_sum_by_day[day] = sq_sum_by_day.get(day, 0.0) + value * value
            n_by_day[day] = n_by_day.get(day, 0) + 1

    xs = sorted(sum_by_day.keys())
    means: list[float] = []
    sems: list[float] = []

    for day in xs:
        n = n_by_day[day]
        mean = sum_by_day[day] / n
        var = (sq_sum_by_day[day] / n) - (mean * mean)
        std = math.sqrt(max(var, 0.0))
        sem = std / math.sqrt(n) if n > 0 else 0.0

        means.append(mean)
        sems.append(sem)

    return xs, means, sems


def _plot_with_band(
    method_to_users: dict[str, list[dict[str, Any]]],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    value_fn,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))

    for method_name, users in method_to_users.items():
        xs, means, sems = _compute_mean_sem_by_day(users, value_fn)
        if not xs:
            continue

        lower = [m - e for m, e in zip(means, sems)]
        upper = [m + e for m, e in zip(means, sems)]

        line = plt.plot(xs, means, marker="o", label=method_name)
        color = line[0].get_color()
        plt.fill_between(xs, lower, upper, alpha=0.25, color=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _sem(values: list[float]) -> float | None:
    if not values:
        return None
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(max(var, 0.0))
    return std / math.sqrt(n)


def _mean_and_sem(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    sem = _sem(values)
    return mean, sem


def _compute_method_summary(users: list[dict[str, Any]]) -> dict[str, float | None]:
    all_m_star: list[float] = []
    first5_m_star: list[float] = []
    last5_m_star: list[float] = []

    all_acc_m0: list[float] = []
    first5_acc_m0: list[float] = []
    last5_acc_m0: list[float] = []

    for user_report in users:
        per_day_metrics = user_report.get("per_day_metrics", [])
        if not isinstance(per_day_metrics, list):
            continue

        sorted_metrics = sorted(
            [rec for rec in per_day_metrics if "day" in rec],
            key=lambda rec: int(rec["day"]),
        )

        first5 = sorted_metrics[:5]
        last5 = sorted_metrics[-5:] if len(sorted_metrics) >= 5 else sorted_metrics

        for rec in sorted_metrics:
            if "m_star" in rec:
                all_m_star.append(float(rec["m_star"]))
            if "acc_m0" in rec:
                all_acc_m0.append(float(rec["acc_m0"]))

        for rec in first5:
            if "m_star" in rec:
                first5_m_star.append(float(rec["m_star"]))
            if "acc_m0" in rec:
                first5_acc_m0.append(float(rec["acc_m0"]))

        for rec in last5:
            if "m_star" in rec:
                last5_m_star.append(float(rec["m_star"]))
            if "acc_m0" in rec:
                last5_acc_m0.append(float(rec["acc_m0"]))

    mean_m_star_all_days, sem_m_star_all_days = _mean_and_sem(all_m_star)
    mean_m_star_first_5_days, sem_m_star_first_5_days = _mean_and_sem(first5_m_star)
    mean_m_star_last_5_days, sem_m_star_last_5_days = _mean_and_sem(last5_m_star)

    mean_acc_m0_all_days, sem_acc_m0_all_days = _mean_and_sem(all_acc_m0)
    mean_acc_m0_first_5_days, sem_acc_m0_first_5_days = _mean_and_sem(first5_acc_m0)
    mean_acc_m0_last_5_days, sem_acc_m0_last_5_days = _mean_and_sem(last5_acc_m0)

    return {
        "mean_m_star_all_days": mean_m_star_all_days,
        "sem_m_star_all_days": sem_m_star_all_days,
        "mean_m_star_first_5_days": mean_m_star_first_5_days,
        "sem_m_star_first_5_days": sem_m_star_first_5_days,
        "mean_m_star_last_5_days": mean_m_star_last_5_days,
        "sem_m_star_last_5_days": sem_m_star_last_5_days,
        "mean_acc_m0_all_days": mean_acc_m0_all_days,
        "sem_acc_m0_all_days": sem_acc_m0_all_days,
        "mean_acc_m0_first_5_days": mean_acc_m0_first_5_days,
        "sem_acc_m0_first_5_days": sem_acc_m0_first_5_days,
        "mean_acc_m0_last_5_days": mean_acc_m0_last_5_days,
        "sem_acc_m0_last_5_days": sem_acc_m0_last_5_days,
    }


def generate_comparison_metrics(
    method_report_paths: dict[str, str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    method_to_users: dict[str, list[dict[str, Any]]] = {}

    for method_name, report_path_str in method_report_paths.items():
        report_path = Path(report_path_str)
        if not report_path.exists():
            raise FileNotFoundError(f"report.json not found for {method_name}: {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        users = report.get("users")
        if not isinstance(users, list):
            raise ValueError(f"Invalid report.json for {method_name}: missing 'users'")
        method_to_users[method_name] = users

    _plot_with_band(
        method_to_users=method_to_users,
        output_path=output_dir / "comparison_acc_m0_by_day.png",
        title="Initial Prediction Accuracy Over Time",
        xlabel="Day",
        ylabel="Initial Accuracy (m=0)",
        value_fn=lambda rec: (
            int(rec["day"]),
            float(rec["acc_m0"]),
        ) if "day" in rec and "acc_m0" in rec else (None, None),
    )

    _plot_with_band(
        method_to_users=method_to_users,
        output_path=output_dir / "comparison_m_star_by_day.png",
        title="Per-Day Mean Corrections to Stop by Method",
        xlabel="Day",
        ylabel="Mean Corrections to Stop (m*)",
        value_fn=lambda rec: (
            int(rec["day"]),
            float(rec["m_star"]),
        ) if "day" in rec and "m_star" in rec else (None, None),
    )

    _plot_with_band(
        method_to_users=method_to_users,
        output_path=output_dir / "comparison_interactive_benefit_by_day.png",
        title="Benefits of Interactive Prediction",
        xlabel="Day",
        ylabel="Corrections Avoided (mismatches_m0 - m*)",
        value_fn=lambda rec: (
            int(rec["day"]),
            float(rec["mismatches_m0"]) - float(rec["m_star"]),
        ) if "day" in rec and "mismatches_m0" in rec and "m_star" in rec else (None, None),
    )

    _plot_with_band(
        method_to_users=method_to_users,
        output_path=output_dir / "comparison_single_correction_gain_by_day.png",
        title="Information Gain with Single Correction",
        xlabel="Day",
        ylabel="|mismatches_m0 - mismatches_m1|",
        value_fn=lambda rec: (
            int(rec["day"]),
            abs(float(rec["mismatches_m0"]) - float(rec["mismatches_m1"])),
        ) if "day" in rec and "mismatches_m0" in rec and "mismatches_m1" in rec else (None, None),
    )

    summary_metrics = {
        method_name: _compute_method_summary(users)
        for method_name, users in method_to_users.items()
    }

    summary_path = output_dir / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"Wrote comparison plots to: {output_dir}")
    print(f"Wrote summary metrics to: {summary_path}")


def main() -> int:
    method_report_paths: dict[str, str] = {
        "Ours": "reports/run_2026_03_06__12_53_55/report.json",
        "Ablation: Long-Term Memory Only": "reports/run_2026_03_06__12_54_17/report.json",
        "Ablation: Episodic Memory Only": "reports/run_2026_03_06__12_54_36/report.json",
        "Ablation: No Memory": "reports/run_2026_03_06__12_55_16/report.json",
    }

    comparison_dir = Path(__file__).parent / "comparison_metrics"
    generate_comparison_metrics(method_report_paths, comparison_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())