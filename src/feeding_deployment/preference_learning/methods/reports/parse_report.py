from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from statistics import mean


DAY_RE = re.compile(r"^\[Day\s+(?P<day>\d+)\]")
USER_RE = re.compile(r"\[user=(?P<user>[^\]]+)\]")
FINISH_RE = re.compile(r"Meal finished after\s+(?P<corr>\d+)\s+corrections")


def parse_report(path: str) -> dict[str, list[tuple[int, int]]]:
    """
    Returns:
        {
            user_id: [(day, corrections), ...]
        }
    """
    results: dict[str, list[tuple[int, int]]] = defaultdict(list)

    current_day: int | None = None
    current_user: str | None = None

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            m_day = DAY_RE.search(line)
            if m_day:
                current_day = int(m_day.group("day"))

            m_user = USER_RE.search(line)
            if m_user:
                current_user = m_user.group("user")

            m_finish = FINISH_RE.search(line)
            if m_finish:
                if current_day is None:
                    raise ValueError(
                        f"Line {lineno}: found completion before any [Day ...] header"
                    )
                if current_user is None:
                    raise ValueError(
                        f"Line {lineno}: found completion before any [user=...] line"
                    )

                corrections = int(m_finish.group("corr"))
                results[current_user].append((current_day, corrections))

    return results


def avg(values: list[int]) -> float:
    if not values:
        return float("nan")
    return mean(values)


def fmt(x: float) -> str:
    if math.isnan(x):
        return "N/A"
    return f"{x:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse report.txt and compute mean corrections for a given user "
            "(all days, days 1-7, and days 24-30)."
        )
    )
    parser.add_argument("--report_path", help="Path to report.txt")
    parser.add_argument("--user", help="User id/name, e.g. 1")
    args = parser.parse_args()

    data = parse_report(args.report_path)

    if args.user not in data:
        print(f"User {args.user!r} not found.", file=sys.stderr)
        if data:
            print("Available users:", ", ".join(sorted(data.keys())), file=sys.stderr)
        return 1

    records = data[args.user]
    all_corrs = [corr for _, corr in records]
    days_1_7 = [corr for day, corr in records if 1 <= day <= 7]
    days_24_30 = [corr for day, corr in records if 24 <= day <= 30]

    print(f"user: {args.user}")
    print(f"completed_meals: {len(records)}")
    print(f"mean_corrections_all_days: {fmt(avg(all_corrs))}")
    print(f"mean_corrections_days_1_7: {fmt(avg(days_1_7))}")
    print(f"mean_corrections_days_24_30: {fmt(avg(days_24_30))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())