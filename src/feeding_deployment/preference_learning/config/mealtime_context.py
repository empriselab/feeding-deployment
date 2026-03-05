from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

SETTINGS = [
    "Personal",
    "Social with person on Left",
    "Social with person in Front",
    "Social with person on Right",
    "Watching TV with TV on Left",
    "Watching TV with TV in Front",
    "Watching TV with TV on Right",
    "Working on laptop with laptop on Left",
    "Working on laptop with laptop in Front",
    "Working on laptop with laptop on Right",
]

TIMES_OF_DAY = [
    "morning", 
    "noon", 
    "evening"
]

@dataclass(frozen=True)
class MealContents:
    label: str
    dippable_items: List[str]
    sauces: List[str]
    storage_condition: str
    intended_serving_temp: str


MEAL_CONTENTS: List[MealContents] = [
    MealContents(
        label="buffalo chicken bites, potato wedges, and ranch dressing",
        dippable_items=["buffalo chicken bites", "potato wedges"],
        sauces=["ranch dressing"],
        storage_condition="refrigerated_leftover",
        intended_serving_temp="hot",
    ),
    MealContents(
        label="strawberries with whipped cream",
        dippable_items=["strawberries"],
        sauces=["whipped cream"],
        storage_condition="refrigerated",
        intended_serving_temp="cold",
    ),
    MealContents(
        label="chicken nuggets, broccoli, and ketchup",
        dippable_items=["chicken nuggets", "broccoli"],
        sauces=["ketchup"],
        storage_condition="refrigerated_leftover",
        intended_serving_temp="hot",
    ),
    MealContents(
        label="general tso's chicken and broccoli",
        dippable_items=["general tso's chicken", "broccoli"],
        sauces=[],
        storage_condition="refrigerated_leftover",
        intended_serving_temp="hot",
    ),
    MealContents(
        label="chicken breast strips and hash brown",
        dippable_items=["chicken breast strips", "hash brown"],
        sauces=[],
        storage_condition="refrigerated_leftover",
        intended_serving_temp="hot",
    ),
    MealContents(
        label="cantaloupes, bananas, watermelon",
        dippable_items=["cantaloupes", "bananas", "watermelon"],
        sauces=[],
        storage_condition="refrigerated",
        intended_serving_temp="cold",
    ),
    MealContents(
        label="bananas, brownies, and chocolate sauce",
        dippable_items=["bananas", "brownies"],
        sauces=["chocolate sauce"],
        storage_condition="refrigerated_leftover",
        intended_serving_temp="warm",
    ),
    MealContents(
        label="bite-sized sandwiches",
        dippable_items=["bite-sized sandwiches"],
        sauces=[],
        storage_condition="refrigerated_leftover",
        intended_serving_temp="warm",
    ),
    MealContents(
        label="breaded fish bites, roasted potatoes, tartar sauce",
        dippable_items=["breaded fish bites", "roasted potatoes"],
        sauces=["tartar sauce"],
        storage_condition="refrigerated_leftover",
        intended_serving_temp="hot",
    ),
    MealContents(
        label="bite-sized pizza and broccoli",
        dippable_items=["bite-sized pizza", "broccoli"],
        sauces=[],
        storage_condition="refrigerated_leftover",
        intended_serving_temp="hot",
    ),
]


MEAL_CONTENTS_BY_LABEL: Dict[str, MealContents] = {
    m.label: m for m in MEAL_CONTENTS
}

MEALS: List[str] = [m.label for m in MEAL_CONTENTS]