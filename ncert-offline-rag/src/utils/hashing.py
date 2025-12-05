import hashlib

def md5(text: str) -> str:
    """Generate MD5 hash for the given text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()