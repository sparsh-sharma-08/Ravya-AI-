from __future__ import annotations
import sys
import subprocess
from typing import Optional

def call_gema(prompt: str, model_variant: str = "2b", timeout: int = 60) -> str:
    """
    Call local Ollama Gemma model with a prompt.
    Returns stdout string or raises RuntimeError with a clear message.
    """
    model_name = f"gemma:{model_variant}"
    print(f"[DEBUG] Using model: {model_name}", file=sys.stderr)

    # Ensure model exists
    try:
        result = subprocess.run(["ollama", "show", model_name], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Model {model_name} not installed.\nRun: ollama pull {model_name}"
            )
        print(f"[DEBUG] Model {model_name} found", file=sys.stderr)
    except FileNotFoundError:
        raise RuntimeError("Ollama CLI not found on PATH. Start Ollama app or install Ollama.")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error checking model availability: {str(e)}")

    cmd = ["ollama", "run", model_name]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        return proc.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Ollama model timeout after {timeout}s")
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        msg = f"Ollama error: {stderr or '<no stderr>'}"
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

# compatibility alias (some modules expect call_gemma)