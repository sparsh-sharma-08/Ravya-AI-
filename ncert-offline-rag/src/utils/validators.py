def validate_jsonl_format(data):
    required_keys = {"text", "class", "subject", "chapter", "language", "textbook", "tokens"}
    if not isinstance(data, dict):
        return False
    if not required_keys.issubset(data.keys()):
        return False
    if not isinstance(data["class"], int) or data["class"] <= 0:
        return False
    if not isinstance(data["subject"], str) or not data["subject"].islower():
        return False
    if not isinstance(data["chapter"], int) or data["chapter"] < 0:
        return False
    if data["language"] not in {"en", "hi", "bn"}:  # Add more languages as needed
        return False
    if data["textbook"] not in {"ncert", "cbse", "state"}:
        return False
    if not isinstance(data["tokens"], int) or data["tokens"] <= 0:
        return False
    return True

def validate_chunk(chunk):
    if not validate_jsonl_format(chunk):
        raise ValueError("Invalid chunk format")
    # Additional validation rules can be added here

def validate_bundle(bundle):
    # Implement validation logic for the entire bundle if necessary
    pass