import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

from src.utils.hashing import generate_md5


def _coerce_int_field(val, field_name: str):
    # coerce numeric strings to int for class only; allow chapter to be string
    if isinstance(val, int):
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        if s.lstrip("-+").isdigit():
            return int(s)
        if field_name == "chapter":
            return s  # keep chapter string as-is
    raise ValueError(f"Field '{field_name}' has wrong type: expected int")


def ingest_jsonl(path: str,
                 *,
                 default_class: Optional[int] = None,
                 default_subject: Optional[str] = None,
                 default_language: Optional[str] = None) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    errors: List[str] = []
    out: List[Dict[str, Any]] = []

    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw_strip = raw.strip()
            if not raw_strip:
                continue
            try:
                obj = json.loads(raw_strip)
                if not isinstance(obj, dict):
                    raise ValueError("JSONL line is not an object")

                if "text" not in obj and "title" not in obj:
                    raise ValueError("Missing required keys in JSONL format.")

                if "class" not in obj:
                    raise ValueError("Missing class in JSONL line.")
                # class must be int (coerce strings); chapter may be int or string
                obj["class"] = _coerce_int_field(obj["class"], "class")
                if "chapter" in obj:
                    obj["chapter"] = _coerce_int_field(obj["chapter"], "chapter")
                else:
                    raise ValueError("Missing chapter in JSONL line.")

                # Ensure id and hash
                if "id" not in obj or not obj.get("id"):
                    text_seed = (obj.get("text") or obj.get("title") or raw_strip).strip()
                    # full md5 for tests, short 8-char for id
                    full_hash = generate_md5(text_seed)
                    short_hash = full_hash[:8]
                    subj = str(obj.get("subject", "")).strip()
                    chap_s = str(obj.get("chapter")).strip().replace(" ", "_")
                    obj["hash"] = full_hash
                    obj["id"] = f"{obj['class']}_{subj}_{chap_s}_{short_hash}"

                if "subject" not in obj and default_subject is not None:
                    obj["subject"] = default_subject
                if "language" not in obj and default_language is not None:
                    obj["language"] = default_language
                if "class" not in obj and default_class is not None:
                    obj["class"] = default_class

                out.append(obj)
            except Exception as e:
                errors.append(f"Line {lineno}: {raw_strip}. {e}")

    if errors:
        for msg in errors:
            print(f"Error processing line: {msg}")
        raise ValueError("Invalid JSONL file: see errors")

    return out

if __name__ == "__main__":
    input_file = "path/to/your/input.jsonl"  # Update this path as needed
    chunks = ingest_jsonl(input_file)
    print(f"Successfully ingested {len(chunks)} chunks.")