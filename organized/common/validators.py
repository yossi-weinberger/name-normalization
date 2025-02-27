"""Validation utilities for name processing."""
import re
from typing import Tuple


def is_name_valid(name: str) -> bool:
    """
    Validates if a name contains only Hebrew characters, spaces, and apostrophes.

    Args:
        name (str): The name to validate

    Returns:
        bool: True if the name contains only valid characters, False otherwise
    """
    return not re.search(r"[^א-תפףךםןץ\s']", name)


def validate_name_parts(full_name: str, given_name: str, family_name: str) -> Tuple[bool, str]:
    """
    Validates that the output names only contain words from the input name.

    Args:
        full_name (str): The original full name
        given_name (str): The extracted first name
        family_name (str): The extracted last name

    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    input_words = set(full_name.split())
    output_words = set()

    if given_name:
        output_words.update(given_name.split())
    if family_name:
        output_words.update(family_name.split())

    invalid_words = output_words - input_words
    if invalid_words:
        return False, f"validation_error: output contains invalid words: {', '.join(invalid_words)}"

    return True, "" 