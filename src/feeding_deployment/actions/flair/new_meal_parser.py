import re
import ast
from pathlib import Path
from tomsutils.llm import OpenAILLM

class NewMealParser:
    def __init__(self, log_dir):

        self.llm = OpenAILLM(
            model_name="gpt-4o",
            cache_dir=log_dir / "llm_cache",
        )

        with open(Path(__file__).parent / "prompts" / "user_message_prompt.txt", 'r') as f:
            self.prompt_skeleton = f.read()

    def parse_user_message(self, food_items, bite_ordering_preference):
        """
        Parses the user message to extract the food items and the bite ordering preference.
        Returns None for fields that cannot be parsed.
        """
        prompt = self.prompt_skeleton % (food_items, bite_ordering_preference)
        response = self.llm.sample_completions(prompt, imgs=None, temperature=0.0, seed=0)[0]

        # Initialize results
        parsed_food_items = None
        parsed_bite_ordering_preference = None

        # Extract potential matches
        food_items_match = re.search(r"Parsed Food Items:\s*(\[.*?\]|.*)", response)
        bite_preference_match = re.search(r"Parsed Bite Preference Ordering:\s*(.*)", response)

        # Parse food items
        if food_items_match:
            food_items_str = food_items_match.group(1).strip()
            try:
                # Attempt to evaluate as a Python list
                parsed_food_items = ast.literal_eval(food_items_str)
                if not isinstance(parsed_food_items, list):
                    raise ValueError
            except Exception as e:
                print("Failed to parse food items as a list:", e)
                print("Attempting to parse as a comma-separated string...")
                # Attempt to split a comma-separated string
                parsed_food_items = [item.strip() for item in food_items_str.split(",") if item.strip()]
                if not parsed_food_items:  # If parsing fails, reset to None
                    print("Also failed to parse as a comma-separated string... setting to None.")
                    parsed_food_items = None

        # Parse bite ordering preference
        if bite_preference_match:
            parsed_bite_ordering_preference = bite_preference_match.group(1).strip()
            if not parsed_bite_ordering_preference:
                parsed_bite_ordering_preference = None

        return parsed_food_items, parsed_bite_ordering_preference
    
if __name__ == '__main__':
    new_meal_parser = NewMealParser()

    parsed_food_items, parsed_bite_ordering_preference = new_meal_parser.parse_user_message("apple and banana", "first apples")
    print("Parsed Food Items: ", parsed_food_items)
    print("Parsed Bite Ordering Preference: ", parsed_bite_ordering_preference)

    parsed_food_items, parsed_bite_ordering_preference = new_meal_parser.parse_user_message("sesame chicken and broccoli", "first sesame chicken")
    print("Parsed Food Items: ", parsed_food_items)
    print("Parsed Bite Ordering Preference: ", parsed_bite_ordering_preference)

    parsed_food_items, parsed_bite_ordering_preference = new_meal_parser.parse_user_message("chicken and rice", "I do not like rice")
    print("Parsed Food Items: ", parsed_food_items)
    print("Parsed Bite Ordering Preference: ", parsed_bite_ordering_preference)