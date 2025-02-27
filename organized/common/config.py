"""Configuration constants for name processing."""

# File paths
DATASET_PATH = "./lp_members.csv"
RESULT_DIR_PATH = "./batch_results/"
TASKS_DIR = "./tasks/"
BATCH_TRACKING_FILE = "batch_tracking.txt"

# API prompt
PROMPT = (
    "Your job is to normalize and split names into a JSON object with given and family names. "
    "User prompts are names to be normalized. Do the following:\n"
    "- Set is_person to true if the name is a person's name (not a business, institution, etc.), "
    "or contains an extractable person's name mixed with other text.\n"
    "- Extract given_name and family_name.\n"
    "- Drop titles (ד״ר, הרב, פרופ׳, עו״ד, etc.) from the output.\n"
    "- In case both names could be both given and family names, assume 'given family' order.\n"
    "- Keep the spelling as is.\n"
)

# API response format
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "name_schema",
        "schema": {
            "type": "object",
            "required": ["is_person", "given_name", "family_name"],
            "properties": {
                "is_person": {
                    "type": "boolean",
                },
                "family_name": {
                    "type": "string",
                },
                "given_name": {
                    "type": "string",
                },
            },
            "additionalProperties": False,
        },
        "strict": True,
    },
} 