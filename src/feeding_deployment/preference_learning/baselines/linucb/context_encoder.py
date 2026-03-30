from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "LinUCB baseline requires numpy. Install it with: pip install numpy"
    ) from e


@dataclass
class ContextEncoder:
    """Deterministic one-hot encoder for (meal, setting, time_of_day, ...)."""

    meal_vocab: List[str]
    setting_vocab: List[str]
    time_vocab: List[str]
    affect_vocab: List[str]
    profile_vocab: List[str]
    include_affective_state: bool = True
    include_physical_profile_label: bool = True
    add_bias: bool = True

    def __post_init__(self) -> None:
        self._meal_index = {v: i for i, v in enumerate(self.meal_vocab)}
        self._setting_index = {v: i for i, v in enumerate(self.setting_vocab)}
        self._time_index = {v: i for i, v in enumerate(self.time_vocab)}
        self._affect_index = {v: i for i, v in enumerate(self.affect_vocab)}
        self._profile_index = {v: i for i, v in enumerate(self.profile_vocab)}

        self._meal_dim = len(self.meal_vocab)
        self._setting_dim = len(self.setting_vocab)
        self._time_dim = len(self.time_vocab)
        self._affect_dim = len(self.affect_vocab) if self.include_affective_state else 0
        self._profile_dim = len(self.profile_vocab) if self.include_physical_profile_label else 0
        self._bias_dim = 1 if self.add_bias else 0

        self.dim = (
            self._meal_dim
            + self._setting_dim
            + self._time_dim
            + self._affect_dim
            + self._profile_dim
            + self._bias_dim
        )

    @staticmethod
    def _collect_unique(values: Iterable[Optional[str]]) -> List[str]:
        seen = set()
        out: List[str] = []
        for v in values:
            if not v:
                continue
            s = str(v)
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @classmethod
    def from_records(
        cls,
        contexts: Sequence[Dict[str, object]],
        physical_profile_labels: Sequence[str],
        *,
        include_affective_state: bool,
        include_physical_profile_label: bool,
        add_bias: bool,
        default_meals: Optional[List[str]] = None,
        default_settings: Optional[List[str]] = None,
        default_times: Optional[List[str]] = None,
        default_affects: Optional[List[str]] = None,
    ) -> ContextEncoder:
        meals = default_meals or []
        settings = default_settings or []
        times = default_times or []
        affects = default_affects or []

        meals = meals + cls._collect_unique(c.get("meal") for c in contexts)
        settings = settings + cls._collect_unique(c.get("setting") for c in contexts)
        times = times + cls._collect_unique(c.get("time_of_day") for c in contexts)
        affects = affects + cls._collect_unique(c.get("transient_affective_state") for c in contexts)
        profiles = cls._collect_unique(physical_profile_labels)

        def _dedup(xs: List[str]) -> List[str]:
            seen = set()
            out = []
            for x in xs:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        return cls(
            meal_vocab=_dedup(meals),
            setting_vocab=_dedup(settings),
            time_vocab=_dedup(times),
            affect_vocab=_dedup(affects),
            profile_vocab=_dedup(profiles),
            include_affective_state=include_affective_state,
            include_physical_profile_label=include_physical_profile_label,
            add_bias=add_bias,
        )

    def encode(
        self,
        context: Dict[str, object],
        *,
        physical_profile_label: str = "",
    ) -> np.ndarray:
        x = np.zeros(self.dim, dtype=np.float64)
        off = 0

        meal = str(context.get("meal") or "")
        if meal in self._meal_index:
            x[off + self._meal_index[meal]] = 1.0
        off += self._meal_dim

        setting = str(context.get("setting") or "")
        if setting in self._setting_index:
            x[off + self._setting_index[setting]] = 1.0
        off += self._setting_dim

        tod = str(context.get("time_of_day") or "")
        if tod in self._time_index:
            x[off + self._time_index[tod]] = 1.0
        off += self._time_dim

        if self.include_affective_state:
            aff = str(context.get("transient_affective_state") or "")
            if aff in self._affect_index:
                x[off + self._affect_index[aff]] = 1.0
            off += self._affect_dim

        if self.include_physical_profile_label:
            if physical_profile_label and physical_profile_label in self._profile_index:
                x[off + self._profile_index[physical_profile_label]] = 1.0
            off += self._profile_dim

        if self.add_bias:
            x[off] = 1.0

        return x

