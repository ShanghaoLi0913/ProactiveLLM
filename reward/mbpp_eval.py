"""
MBPP test execution for computing task success score.

Executes assistant code with MBPP tests in a subprocess and returns pass/fail.
"""
import subprocess
import tempfile
from typing import Optional


def run_python_with_tests(code: str, tests: str, timeout_sec: int = 5) -> bool:
    """Execute code and tests in a subprocess; return True if exit code==0.
    
    Tests should contain assertions; a non-zero exit implies failure.
    """
    snippet = code + "\n\n" + tests + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(snippet)
        path = f.name
    try:
        proc = subprocess.run(["python", path], capture_output=True, text=True, timeout=timeout_sec)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        import os
        try:
            os.unlink(path)
        except:
            pass


def score_mbpp_passfail(assistant_text: str, tests: Optional[str]) -> float:
    """Score assistant output based on MBPP test execution.
    
    Args:
        assistant_text: Generated assistant message (should contain code)
        tests: MBPP test code (string)
    
    Returns:
        1.0 if tests pass, 0.0 if fail, 0.6 if no tests provided
    """
    if not tests:
        return 0.6  # fallback neutral if no tests provided
    
    # Naive extraction: assume assistant_text contains code block or python code
    # TODO: Add more robust code extraction (e.g., extract from markdown code blocks)
    code = assistant_text
    ok = run_python_with_tests(code, tests)
    return 1.0 if ok else 0.0

