"""Schema creation utilities for name processing."""

def create_name_schema(name: str) -> dict:
    """Creates a schema that allows consecutive sequences from either start or end, excluding full-length sequences."""
    words = name.split()
    
    # Create sequences from both start and end, but not full length
    from_start = []
    from_end = []
    
    # Add sequences from start (excluding full length)
    for i in range(1, len(words)):
        from_start.append("_".join(words[:i]))
    
    # Add sequences from end (excluding full length)
    for i in range(1, len(words)):
        from_end.append("_".join(words[-i:]))
    
    # Both given_name and family_name can be from either direction
    name_options = from_start + from_end
    
    return {
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
                        "enum": name_options,
                    },
                    "given_name": {
                        "type": "string",
                        "enum": name_options,
                    },
                },
                "additionalProperties": False,
            },
            "strict": True,
        },
    } 