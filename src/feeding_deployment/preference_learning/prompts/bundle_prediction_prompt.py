from pathlib import Path
from typing import List

from feeding_deployment.preference_learning import config as root_config  # type: ignore


_PROMPTS_DIR = Path(__file__).parent
_TEMPLATE_PATH = _PROMPTS_DIR / "bundle_prediction.txt"


_PREF_FIELDS: List[str] = [name for (name, _, _) in root_config.PREFERENCE_BUNDLE]


def render_bundle_prediction_prompt(
    physical_profile: str,
    ltm_summary: str,
    retrieved_block: str,
    context: dict,
    corrected_block: str,
    options_block: str,
) -> str:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(
        physical_profile=physical_profile,
        ltm_summary=ltm_summary,
        retrieved_block=retrieved_block,
        meal=context.get("meal"),
        setting=context.get("setting"),
        time_of_day=context.get("time_of_day"),
        corrected_block=corrected_block,
        options_block=options_block,
        pref_fields_csv=", ".join(_PREF_FIELDS),
    )

