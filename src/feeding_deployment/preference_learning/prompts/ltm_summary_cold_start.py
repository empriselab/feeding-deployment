from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent
_TEMPLATE_PATH = _PROMPTS_DIR / "ltm_summary_cold_start.txt"


def render_ltm_cold_start_prompt(
    physical_profile: str,
    options_block: str,
) -> str:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(
        physical_profile=physical_profile,
        options_block=options_block,
    )

