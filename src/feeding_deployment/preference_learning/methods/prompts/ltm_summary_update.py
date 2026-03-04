from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent
_TEMPLATE_PATH = _PROMPTS_DIR / "ltm_summary_update.txt"


def get_ltm_update_prompt(
    physical_profile: str,
    previous_ltm_summary: str,
    new_episodes_block: str,
    options_block: str,
) -> str:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(
        physical_profile=physical_profile,
        previous_ltm_summary=previous_ltm_summary,
        new_episodes_block=new_episodes_block,
        options_block=options_block,
    )

