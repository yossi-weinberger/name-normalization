from typing import Tuple

def validate_name_parts(full_name: str, first_name: str, last_name: str) -> Tuple[bool, str]:
    """
    Validates that the output names only contain words from the input name.
    """
    input_words = set(full_name.split())
    output_words = set()
    
    if first_name:
        output_words.update(first_name.split())
    if last_name:
        output_words.update(last_name.split())
    
    invalid_words = output_words - input_words
    if invalid_words:
        return False, f"validation_error: output contains invalid words: {', '.join(invalid_words)}"
    
    return True, ""