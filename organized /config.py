from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    DATASET_PATH: str = "./lp_members.csv"
    RESULT_DIR_PATH: str = "./results/"
    TASKS_PATH: str = "./tasks.jsonl"
    
    RESPONSE_FORMAT: Dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "name_schema",
            "schema": {
                "type": "object",
                "required": ["is_person", "first_name", "last_name"],
                "properties": {
                    "is_person": {
                        "type": "boolean",
                        "description": "Whether it is a person's name.",
                    },
                    "last_name": {
                        "type": "string",
                        "description": "The last name",
                    },
                    "first_name": {
                        "type": "string",
                        "description": "The first name",
                    },
                },
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    SYSTEM_PROMPT: str = (
        "For Hebrew text: "
        "1. Is this a person (not business)? If no, return is_person=false "
        "2. If yes: extract first_name, last_name, ignoring titles "
        "3. Use only existing words, Hebrew chars only "
        "4. If both names could be first names or both could be family names, "
        "   keep the original order (first word as first_name, second as last_name) "
        "Return clean name without titles, exact substrings from input"
    ) 