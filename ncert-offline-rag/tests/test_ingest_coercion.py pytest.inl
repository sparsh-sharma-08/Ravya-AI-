import json
import subprocess
import sys
from pathlib import Path
import tempfile

def test_ingest_accepts_string_chapter(tmp_path):
    p = tmp_path / "sample.jsonl"
    p.write_text(json.dumps({"id":"c1","chapter":"3","title":"T","text":"x"}) + "\n", encoding="utf-8")
    cmd = [sys.executable, str(Path.cwd() / "ingest_pipeline.py"),
           "--input", str(p), "--class", "8", "--subject", "science", "--language", "en"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0, f"ingest failed: stdout={proc.stdout} stderr={proc.stderr}"