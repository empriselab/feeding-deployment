#!/usr/bin/env python3
"""
Export CSV dataset (from generate_dataset.py) to text prompt or JSON.

Reads a CSV file and exports the 15-day selection period to either:
  - txt: human-readable prompt (same format as before)
  - json: structured data for programmatic use

Formulation mapping (see FORMULATION.md):
  - long_term_preferences_days_1_15 = slow latent preference U^pref (not U^phys).
  - Context.transient_affective_state = Y_t (transient affective state).
  - CSV column: transient_affective_state (backward-compat: also read affective_state).
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List


# Preference field names and their human-readable labels
PREFERENCE_LABELS: Dict[str, str] = {
    "microwave_time": "Microwave time",
    "occlusion_relevance": "Occlusion relevance",
    "robot_speed": "Robot speed",
    "skewering_axis": "Skewering axis selection",
    "transfer_mode": "Outside mouth vs inside mouth transfer",
    "outside_mouth_distance": "For outside-mouth transfer: distance from the mouth",
    "robot_ready_cue": "[Robot→Human] Convey ready for initiating transfer",
    "bite_initiation_feeding": "[Human→Robot] Bite initiation for FEEDING",
    "bite_initiation_drinking": "[Human→Robot] Bite initiation for DRINKING",
    "bite_initiation_wiping": "[Human→Robot] Bite initiation for MOUTH WIPING",
    "robot_bite_available_cue": "[Robot→Human] Convey user can take a bite",
    "bite_completion_feeding": "[Human→Robot] Bite completion for FEEDING",
    "bite_completion_drinking": "[Human→Robot] Bite completion for DRINKING",
    "bite_completion_wiping": "[Human→Robot] Bite completion for MOUTH WIPING",
    "web_interface_confirmation": "Web interface confirmation",
    "retract_between_bites": "Retract between bites",
    "bite_ordering_preference": "Bite ordering preference (X/Y)",
    "bite_dipping_preference": "Bite dipping preference (X/Y in Z)",
    "amount_to_dip": "Amount to dip",
    "wait_before_autocontinue_seconds": "Time to wait before autocontinue",
}

ExportFormat = str  # "txt" | "json"


def get_preference_fields() -> List[str]:
    """Return list of preference field names."""
    return list(PREFERENCE_LABELS.keys())


def _long_term_prefs_dict(row: Dict[str, str], phase: int) -> Dict[str, str]:
    """Extract long-term preferences as a dict (field -> value)."""
    pref_fields = get_preference_fields()
    out = {}
    for field in pref_fields:
        lt_key = f"ltpref_{phase}_{field}"
        val = row.get(lt_key, "").strip()
        if val:
            out[field] = val
    return out


def _day_preferences_dict(row: Dict[str, str]) -> Dict[str, str]:
    """Extract daily preferences as a dict."""
    pref_fields = get_preference_fields()
    return {f: row.get(f, "").strip() for f in pref_fields if row.get(f, "").strip()}


def load_15day_data(csv_path: str) -> Dict[str, Any]:
    """
    Load CSV and return structured data for days 1-15.
    Used by both txt and json export.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("CSV file is empty or has no data rows")

    first_row = rows[0]
    user = first_row.get("user", "Unknown")
    deployment_id = first_row.get("deployment_id", "Unknown")

    days_1_15 = []
    for row in rows:
        try:
            day = int(row.get("day", "0"))
            if 1 <= day <= 15:
                days_1_15.append(row)
        except ValueError:
            continue

    if not days_1_15:
        raise ValueError("No data found for days 1-15")

    days_1_15.sort(key=lambda r: int(r.get("day", "0")))

    long_term_1 = _long_term_prefs_dict(days_1_15[0], phase=1)

    # Y_t: accept new or legacy column name (see FORMULATION.md)
    def _y_t(row: Dict[str, str]) -> str:
        return (
            row.get("transient_affective_state") or row.get("affective_state", "Unknown")
        ).strip() or "Unknown"

    daily_entries: List[Dict[str, Any]] = []
    for row in days_1_15:
        daily_entries.append({
            "day": int(row.get("day", 0)),
            "context": {
                "meal": row.get("meal", "Unknown"),
                "setting": row.get("setting", "Unknown"),
                "transient_affective_state": _y_t(row),
            },
            "preferences": _day_preferences_dict(row),
        })

    return {
        "user": user,
        "deployment_id": deployment_id,
        "long_term_preferences_days_1_15": long_term_1,
        "daily_entries": daily_entries,
    }


def format_long_term_preferences_text(lt_prefs: Dict[str, str]) -> str:
    """Format long-term preferences for text output."""
    if not lt_prefs:
        return "  (No long-term preferences recorded)"
    lines = [f"  - {PREFERENCE_LABELS[field]}: {value}" for field, value in lt_prefs.items()]
    return "\n".join(lines)


def format_day_preferences_text(prefs: Dict[str, str]) -> str:
    """Format daily preferences for text output."""
    if not prefs:
        return "  (No preferences recorded)"
    lines = [f"  - {PREFERENCE_LABELS[field]}: {value}" for field, value in prefs.items()]
    return "\n".join(lines)


def export_to_txt(data: Dict[str, Any]) -> str:
    """Convert structured data to the same text prompt format as before."""
    lines = []
    user = data["user"]
    dep = data["deployment_id"]
    lt = data["long_term_preferences_days_1_15"]
    entries = data["daily_entries"]

    lines.append("=" * 80)
    lines.append(f"15-Day Selection Period: {user} - {dep}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("SLOW LATENT PREFERENCE U^pref (Days 1-15)")
    lines.append("-" * 80)
    lines.append(format_long_term_preferences_text(lt))
    lines.append("")
    lines.append("")
    lines.append("DAILY ENTRIES (Days 1-15)")
    lines.append("-" * 80)
    lines.append("")

    for e in entries:
        day = e["day"]
        ctx = e["context"]
        prefs = e["preferences"]
        lines.append(f"Day {day}")
        lines.append("  Context:")
        lines.append(f"    - Meal: {ctx['meal']}")
        lines.append(f"    - Setting: {ctx['setting']}")
        lines.append(f"    - Transient affective state (Y_t): {ctx['transient_affective_state']}")
        lines.append("  Preferences:")
        lines.append(format_day_preferences_text(prefs))
        lines.append("")

    return "\n".join(lines)


def export_to_json(data: Dict[str, Any]) -> str:
    """Serialize structured data to JSON string."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def export_csv(
    csv_path: str,
    export_format: ExportFormat,
    output_path: str | None = None,
) -> str:
    """
    Export CSV dataset to txt or json.

    Args:
        csv_path: Path to input CSV file
        export_format: "txt" or "json"
        output_path: Path to output file (if None, auto-generated from CSV name and format)

    Returns:
        Path to the generated file
    """
    export_format = export_format.lower()
    if export_format not in ("txt", "json"):
        raise ValueError("export_format must be 'txt' or 'json'")

    data = load_15day_data(csv_path)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.dirname(csv_path) or "."
    if output_path is None:
        ext = ".txt" if export_format == "txt" else ".json"
        output_path = os.path.join(output_dir, f"{base_name}__15d{ext}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if export_format == "txt":
        content = export_to_txt(data)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        content = export_to_json(data)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    return output_path


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CSV dataset to text prompt or JSON (15-day selection period)."
    )
    parser.add_argument(
        "csv_file",
        help="Path to input CSV file (generated from generate_dataset.py)",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["txt", "json"],
        default="txt",
        help="Export format: txt (prompt) or json (default: txt)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: auto-generated from CSV name and format)",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    try:
        out_path = export_csv(args.csv_file, args.format, args.output)
        print(f"Exported to: {out_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
