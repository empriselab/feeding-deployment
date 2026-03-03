from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Meal structure: per-meal structure and serving assumptions
# - storage_condition: how the meal is stored before serving (here: usually refrigerated)
# - intended_serving_temp: how it is meant to be served (hot / warm / cold)
MEAL_STRUCTURE: Dict[str, Dict[str, object]] = {
    "buffalo chicken bites, potato wedges, and ranch dressing": {
        "dippable_items": ["buffalo chicken bites", "potato wedges"],
        "sauces": ["ranch dressing"],
        "storage_condition": "refrigerated_leftover",
        "intended_serving_temp": "hot",
    },
    "strawberries with whipped cream": {
        "dippable_items": ["strawberries"],
        "sauces": ["whipped cream"],
        "storage_condition": "refrigerated",
        "intended_serving_temp": "cold",
    },
    "chicken nuggets, broccoli, and ketchup": {
        "dippable_items": ["chicken nuggets", "broccoli"],
        "sauces": ["ketchup"],
        "storage_condition": "refrigerated_leftover",
        "intended_serving_temp": "hot",
    },
    "general tso's chicken and broccoli": {
        "dippable_items": ["general tso's chicken", "broccoli"],
        "sauces": [],
        "storage_condition": "refrigerated_leftover",
        "intended_serving_temp": "hot",
    },
    "chicken breast strips and hash brown": {
        "dippable_items": ["chicken breast strips", "hash brown"],
        "sauces": [],
        "storage_condition": "refrigerated_leftover",
        "intended_serving_temp": "hot",
    },
    "cantaloupes, bananas, watermelon": {
        "dippable_items": ["cantaloupes", "bananas", "watermelon"],
        "sauces": [],
        "storage_condition": "refrigerated",
        "intended_serving_temp": "cold",
    },
    "bananas, brownies, and chocolate sauce": {
        "dippable_items": ["bananas", "brownies"],
        "sauces": ["chocolate sauce"],
        "storage_condition": "refrigerated_leftover",
        "intended_serving_temp": "warm",
    },
    "bite-sized sandwiches": {
        "dippable_items": ["bite-sized sandwiches"],
        "sauces": [],
        "storage_condition": "refrigerated_leftover",
        "intended_serving_temp": "warm",
    },
    "breaded fish bites, roasted potatoes, tartar sauce": {
        "dippable_items": ["breaded fish bites", "roasted potatoes"],
        "sauces": ["tartar sauce"],
        "storage_condition": "refrigerated_leftover",
        "intended_serving_temp": "hot",
    },
    "bite-sized pizza and broccoli": {
        "dippable_items": ["bite-sized pizza", "broccoli"],
        "sauces": [],
        "storage_condition": "refrigerated_leftover",
        "intended_serving_temp": "hot",
    },
}

MEALS = list(MEAL_STRUCTURE.keys())

SETTINGS = [
    "Personal",
    "Social with Person on Left",
    "Social with Person on Right",
    "Watching TV",
    "Social + Watching TV",
    "Working on laptop",
]

TIMES_OF_DAY = ["morning", "noon", "evening"]

AFFECTIVE_STATES = ["Neutral", "Hurried", "Slower", "Fatigued", "Energetic"]

# Preference bundle schema (19 dimensions, excluding confirm_bundle)
PREFERENCE_BUNDLE: List[Tuple[str, str, List[str]]] = [
    ("microwave_time", "Microwave time", ["no microwave", "1 min", "2 min", "3 min"]),
    ("occlusion_relevance", "Occlusion relevance", ["minimize left occlusion", "minimize front occlusion", "do not consider occlusion"]),
    ("robot_speed", "Robot speed", ["slow", "medium", "fast"]),
    ("skewering_axis", "Skewering axis selection", ["parallel to major axis", "perpendicular to major axis"]),
    ("transfer_mode", "Outside mouth vs inside mouth transfer", ["outside mouth transfer", "inside mouth transfer"]),
    ("outside_mouth_distance", "For outside-mouth transfer: distance from the mouth", ["near", "medium", "far"]),
    ("robot_ready_cue", "[Robot→Human] Convey ready for initiating transfer", ["speech", "LED", "speech + LED", "no cue"]),
    ("bite_initiation_feeding", "[Human→Robot] Bite initiation for FEEDING", ["open mouth", "button", "autocontinue"]),
    ("bite_initiation_drinking", "[Human→Robot] Bite initiation for DRINKING", ["open mouth", "button", "autocontinue"]),
    ("bite_initiation_wiping", "[Human→Robot] Bite initiation for MOUTH WIPING", ["open mouth", "button", "autocontinue"]),
    ("robot_bite_available_cue", "[Robot→Human] Convey user can take a bite", ["speech", "LED", "speech + LED", "no cue"]),
    ("bite_completion_feeding", "[Human→Robot] Bite completion for FEEDING", ["perception", "button", "autocontinue"]),
    ("bite_completion_drinking", "[Human→Robot] Bite completion for DRINKING", ["perception", "button", "autocontinue"]),
    ("bite_completion_wiping", "[Human→Robot] Bite completion for MOUTH WIPING", ["perception", "button", "autocontinue"]),
    ("web_interface_confirmation", "Web interface confirmation", ["yes", "no"]),
    ("retract_between_bites", "Retract between bites", ["yes", "no"]),
    ("bite_dipping_preference", "Bite dipping preference (X/Y in Z)", ["dip X in Z", "dip Y in Z", "dip both X and Y in Z", "do not dip"]),
    ("amount_to_dip", "Amount to dip", ["less", "more"]),
    ("wait_before_autocontinue_seconds", "Time to wait before autocontinue", ["10", "100", "1000"]),
]

# Note: bite_ordering_preference is not in the user's list, but keeping it for compatibility
PREFERENCE_BUNDLE_WITH_ORDERING = PREFERENCE_BUNDLE + [
    ("bite_ordering_preference", "Bite ordering preference (X/Y)", ["alternate X and Y", "start with X then Y", "start with Y then X"]),
]

PHYSICAL_CAPABILITY_PROFILES = {
    "severe_paralysis_clear_speech": """Severe upper-limb paralysis with clear speech

This user has very limited voluntary control of their arms and hands and cannot reliably press buttons or perform gestures.

However, they have clear and consistent speech, allowing them to communicate intentions verbally.

They can open and close their mouth reliably and tolerate moderate feeding speeds, but require the robot to handle nearly all physical aspects of the task.

Fatigue is present but not extreme, so they can sustain interaction through an entire meal with consistent pacing.""",

    "moderate_motor_unreliable_speech": """Moderate motor control with unreliable speech

This user retains some arm and hand movement, enough to press a large accessible button and make small adjustments in posture.

Their speech is slurred or inconsistent, making voice commands unreliable for precise interaction.

They show stable mouth control and can safely receive food, but fatigue appears quickly, requiring slower pacing and occasional pauses.

Because speech is unreliable, they depend heavily on simple physical interfaces for control and confirmation.""",

    "high_fatigue_swallowing_risk": """High fatigue and elevated swallowing risk

This user demonstrates limited endurance and becomes fatigued quickly during meals.

Their mouth opening is delayed and sometimes inconsistent, requiring careful timing for safe transfer.

They are sensitive to fast movements and large bite sizes, and there is heightened concern for choking or aspiration, meaning feeding must proceed slowly and cautiously.

Although they may have some speech or motor ability, safety considerations dominate all interaction design.""",

    "inconsistent_mouth_control": """Inconsistent mouth control with cognitive or timing variability

This user has some physical movement but limited precision and coordination.

Speech is present but slow, and gestures are minimal or absent.

The most significant challenge is inconsistent mouth timing—they may open too early, too late, or unpredictably—making perception-based bite detection unreliable.

They benefit from explicit confirmations and predictable pacing, even when fatigue is not severe.""",
}