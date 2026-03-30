from dataclasses import dataclass


@dataclass(frozen=True)
class LinUCBConfig:
    alpha: float = 0.5
    include_affective_state: bool = True
    include_physical_profile_label: bool = True
    add_bias: bool = True


DEFAULT_CONFIG = LinUCBConfig()

