from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PhysicalCapability:
    label: str
    description: str


PHYSICAL_CAPABILITY_PROFILES: List[PhysicalCapability] = [
    PhysicalCapability(
        label="limited_arms_no_trunk_good_head", # Benjamin
        description="This user has severe upper-limb paralysis with very limited voluntary control of their arms (cannot press physical buttons). They cannot lean forward due to lack of trunk control. However, they have good neck and head control and are able to open their mouth wide and perform head gestures. They interact with the web interface through a reflective dot on their nose, which is tracked by their personal device.",
    ),
    PhysicalCapability(
        label="moderate_arms_good_trunk_good_head",
        description="This user has moderate voluntary control of their arms and is able to press physical buttons. They can lean forward to reach food during outside-mouth transfers. They have good neck and head control and can open their mouth wide and perform head gestures. They interact with the web interface on their personal device using their arms.",
    ),
    PhysicalCapability(
        label="moderate_arms_limited_trunk_limited_head",
        description="This user has moderate voluntary control of their arms and is able to press physical buttons. However, they cannot lean forward due to limited trunk control. They also have limited control of their neck muscles and cannot reliably perform head gestures. In addition, they have limited mouth opening, making open-mouth or mouth-gesture-based interaction challenging. They interact with the web interface directly using their arms.",
    ),
    PhysicalCapability(
        label="limited_arms_no_trunk_limited_head",
        description="This user has severe upper-limb paralysis and cannot reliably press physical buttons. They also have limited control of their neck muscles and cannot reliably perform head gestures. They cannot lean forward due to limited trunk control. Mouth opening is possible but may be slow. They interact with the web interface using an assistive input device (e.g., eye gaze or switch scanning).",
    ),
    PhysicalCapability(
        label="good_arms_good_trunk_good_head_limited_mouth",
        description="This user has good voluntary control of their arms and can reliably press physical buttons and interact with the web interface directly using their arms. They can lean forward and have good neck and head control, including performing head gestures. However, they have limited mouth opening, which can make open-mouth-based readiness detection challenging and may require smaller bites during feeding.",
    ),
]