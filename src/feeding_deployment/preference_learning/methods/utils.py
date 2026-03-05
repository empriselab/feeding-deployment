from typing import Any, Dict, List, Optional
from openai import RateLimitError as OpenAIRateLimitError
import time
import os
import feeding_deployment.preference_learning.config as root_config  # type: ignore
PREF_FIELDS: List[str] = [name for (name, _, _) in root_config.PREFERENCE_BUNDLE]

def _retry_on_rate_limit(fn, max_retries: int = 5, base_wait: float = 60.0):
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


def _resolve_api_key(cli_key: Optional[str]) -> str:
    if cli_key and cli_key.strip():
        return cli_key.strip()
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    try:
        import generate_dataset_llm  # type: ignore
        file_key = getattr(generate_dataset_llm, "OPENAI_API_KEY", None)
        if isinstance(file_key, str) and file_key.strip():
            return file_key.strip()
    except Exception:
        pass
    raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY env var or pass --api-key.")


def _episode_text(day: int, context: Dict[str, Any], prefs: Dict[str, str]) -> str:
    ctx = (
        f"day={day}; meal={context.get('meal')}; setting={context.get('setting')}; "
        f"time_of_day={context.get('time_of_day')};"
    )
    pref_str = "; ".join(f"{k}={prefs.get(k,'')}" for k in PREF_FIELDS)
    return f"{ctx}\npreferences: {pref_str}"


def _extract_truth_bundle(day_rec: Dict[str, Any]) -> Dict[str, str]:
    prefs = day_rec.get("preferences", {}) or {}
    out: Dict[str, str] = {}
    for field in PREF_FIELDS:
        val = prefs.get(field, {})
        out[field] = str(val.get("choice", "")).strip() if isinstance(val, dict) else ""
    return out
