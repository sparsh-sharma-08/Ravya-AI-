from __future__ import annotations
"""
src/rag/gemma_call.py
Call local Ollama Gemma model via subprocess. Captures stdout and raises on error.
"""
import subprocess


def call_gemma(prompt: str, model_variant: str = "2b", timeout: int = 60) -> str:
    """
    Call local Ollama Gemma model. Sends prompt via stdin to 'ollama run <model>'.
    Returns stdout string or raises RuntimeError on failure.
    """
    cmd = ["ollama", "run", f"gemma:{model_variant}"]
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, check=True, timeout=timeout)
        return proc.stdout
    except FileNotFoundError:
        raise RuntimeError("Ollama executable not found. Start the Ollama app or install ollama.")
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        raise RuntimeError(f"Ollama failed: {stderr}")