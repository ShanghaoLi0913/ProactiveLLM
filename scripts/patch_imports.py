"""Post-hoc import patching: detect missing imports in generated code and re-test.

Applied uniformly across all method evals (Direct/CF/Base/PO/TactfulLLM) for
fair comparison. Only patches code where:
  1. Original code FAILED (passed=False)
  2. Missing imports detected (heuristic match on common module usage)
  3. Patched code re-tested via score_code_passfail

Usage:
    python scripts/patch_imports.py \
        --input outputs/eval_v33_v3_qwen_dpo_v2_100.json \
        --output outputs/eval_v33_v3_qwen_dpo_v2_100_patched.json \
        --test_states data/seeds/test_states_v29_eval_200.jsonl
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path("/root/autodl-tmp/ProactiveLLM")
sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate_dpo_model import extract_code_from_text, score_code_passfail  # noqa


# Common module → import statement mapping
# Conservative: only patterns we're confident about
IMPORT_PATTERNS = [
    # (regex_pattern, import_statement, dedup_key)
    (r'\bnp\.', 'import numpy as np', 'numpy'),
    (r'\bpd\.', 'import pandas as pd', 'pandas'),
    (r'\bplt\.', 'import matplotlib.pyplot as plt', 'matplotlib'),
    (r'\bsns\.', 'import seaborn as sns', 'seaborn'),
    (r'\brequests\.\w', 'import requests', 'requests'),
    (r'\bjson\.\w', 'import json', 'json'),
    (r'\bre\.\w', 'import re', 're'),
    (r'\bos\.\w', 'import os', 'os'),
    (r'\bsys\.\w', 'import sys', 'sys'),
    (r'\brandom\.\w', 'import random', 'random'),
    (r'\bcsv\.\w', 'import csv', 'csv'),
    (r'\bio\.\w|BytesIO\(|StringIO\(', 'import io', 'io'),
    (r'\bmath\.\w', 'import math', 'math'),
    (r'\bdatetime\(|datetime\.', 'from datetime import datetime', 'datetime'),
    (r'\btimedelta\(', 'from datetime import timedelta', 'timedelta'),
    (r'\bImage\.open|Image\.fromarray', 'from PIL import Image', 'PIL'),
    (r'\bCounter\(', 'from collections import Counter', 'Counter'),
    (r'\bdefaultdict\(', 'from collections import defaultdict', 'defaultdict'),
    (r'\bBeautifulSoup\(', 'from bs4 import BeautifulSoup', 'BeautifulSoup'),
    (r'\bunescape\(', 'from html import unescape', 'unescape'),
    (r'\burlparse\(', 'from urllib.parse import urlparse', 'urlparse'),
    (r'\bquote\(|\bunquote\(', 'from urllib.parse import quote, unquote', 'urlquote'),
    (r'\bb64decode\(|\bb64encode\(', 'import base64', 'base64'),
    (r'\bzip_longest\(|\bpermutations\(|\bcombinations\(|\bproduct\(|\bchain\(',
     'import itertools', 'itertools'),
    (r'\bsqrt\(|\bcos\(|\bsin\(|\blog\(', 'import math', 'math_alt'),  # alt for math import
    (r'\bzipfile\.', 'import zipfile', 'zipfile'),
    (r'\bglob\.\w', 'import glob', 'glob'),
    (r'\bshutil\.', 'import shutil', 'shutil'),
    (r'\bsubprocess\.', 'import subprocess', 'subprocess'),
    (r'\bhashlib\.', 'import hashlib', 'hashlib'),
    (r'\bLinearRegression\(', 'from sklearn.linear_model import LinearRegression', 'lr'),
    (r'\bStandardScaler\(|\bMinMaxScaler\(',
     'from sklearn.preprocessing import StandardScaler, MinMaxScaler', 'sklearn_pre'),
    (r'\bscipy\.', 'import scipy', 'scipy'),
    (r'\bstats\.\w', 'from scipy import stats', 'scipy_stats'),
    (r'\bcalendar\.', 'import calendar', 'calendar'),
    (r'\brelativedelta\(', 'from dateutil.relativedelta import relativedelta', 'reldelta'),
]


def detect_missing_imports(code: str) -> List[str]:
    """Detect modules used but not imported in code. Returns list of import statements."""
    # Get existing imports (rough)
    existing_imports = re.findall(
        r'^[ \t]*(?:from\s+\S+\s+)?import\s+[^\n]+', code, re.MULTILINE,
    )
    existing_text = '\n'.join(existing_imports)

    needed = []
    seen_keys = set()
    for pattern, import_stmt, dedup_key in IMPORT_PATTERNS:
        if dedup_key in seen_keys:
            continue
        if not re.search(pattern, code):
            continue
        # Check if already imported (heuristic: key appears in existing imports)
        # E.g., 'numpy' for np module
        primary_check = {
            'numpy': r'numpy', 'pandas': r'pandas', 'matplotlib': r'matplotlib',
            'seaborn': r'seaborn', 'requests': r'requests', 'json': r'\bjson\b',
            're': r'\bimport\s+re\b|\bfrom\s+re\b', 'os': r'\bimport\s+os\b',
            'sys': r'\bimport\s+sys\b', 'random': r'random', 'csv': r'\bcsv\b',
            'io': r'\bio\b', 'math': r'\bmath\b', 'datetime': r'datetime',
            'timedelta': r'timedelta', 'PIL': r'PIL', 'Counter': r'Counter',
            'defaultdict': r'defaultdict', 'BeautifulSoup': r'bs4|BeautifulSoup',
            'unescape': r'unescape', 'urlparse': r'urlparse', 'urlquote': r'quote',
            'base64': r'base64', 'itertools': r'itertools',
            'math_alt': r'\bmath\b', 'zipfile': r'zipfile', 'glob': r'\bglob\b',
            'shutil': r'shutil', 'subprocess': r'subprocess', 'hashlib': r'hashlib',
            'lr': r'LinearRegression|sklearn',
            'sklearn_pre': r'StandardScaler|MinMaxScaler|sklearn',
            'scipy': r'scipy', 'scipy_stats': r'\bstats\b|scipy',
            'calendar': r'calendar', 'reldelta': r'relativedelta|dateutil',
        }
        check_pattern = primary_check.get(dedup_key, dedup_key)
        if re.search(check_pattern, existing_text):
            continue
        needed.append(import_stmt)
        seen_keys.add(dedup_key)
    return needed


def patch_code(code: str) -> str:
    """Add missing imports to code."""
    missing = detect_missing_imports(code)
    if not missing:
        return code

    # Find where to insert: before first def/class, after existing imports
    lines = code.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if re.match(r'\s*(?:def|class)\s+', line):
            insert_idx = i
            break
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insert_idx = i + 1

    new_lines = lines[:insert_idx] + missing + [''] + lines[insert_idx:]
    return '\n'.join(new_lines)


def load_state_tests(test_states_path: Path) -> dict:
    """Load mapping state_id → tests."""
    state_tests = {}
    with open(test_states_path) as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                sid = s.get('id') or s.get('state_id')
                tests = s.get('convcodeworld_tests') or s.get('test')
                if sid and tests:
                    state_tests[sid] = tests
    return state_tests


def repatch_eval_file(input_path: Path, output_path: Path,
                       state_tests: dict, dry_run: bool = False) -> dict:
    print(f"Loading {input_path}", flush=True)
    d = json.load(open(input_path))
    detailed = d.get('detailed_results', [])

    n_total_candidates = 0
    n_pass_before = 0
    n_patched_attempted = 0
    n_pass_after_patch = 0
    n_state_pass_changes = 0  # entries where pass@1 changed False→True

    for r in detailed:
        sid = r['state_id']
        tests = state_tests.get(sid)
        last_turn = r['conversation'][-1]
        if last_turn.get('action') != 'Execute':
            continue

        candidates = last_turn.get('candidate_results', [])
        n_total_candidates += len(candidates)

        # Track changes for this entry
        original_p1 = last_turn.get('pass_at_k', {}).get('pass@1', False)

        for cand in candidates:
            if cand.get('passed'):
                n_pass_before += 1
                continue

            code_raw = cand.get('code', '')
            extracted = extract_code_from_text(code_raw)
            if not extracted:
                continue

            missing = detect_missing_imports(extracted)
            if not missing:
                continue

            patched = patch_code(extracted)
            n_patched_attempted += 1

            if dry_run:
                continue

            if not tests:
                continue
            try:
                score = score_code_passfail(patched, tests, timeout=30)
            except Exception as e:
                continue
            if score == 1.0:
                cand['passed'] = True
                cand['patched_imports'] = missing
                n_pass_after_patch += 1

        if not dry_run:
            # Recompute pass@k
            for k in [1, 3, 5]:
                n_pass_in_first_k = sum(1 for c in candidates[:k] if c.get('passed'))
                last_turn.setdefault('pass_at_k', {})[f'pass@{k}'] = n_pass_in_first_k > 0
            new_p1 = last_turn['pass_at_k']['pass@1']
            if new_p1 and not original_p1:
                n_state_pass_changes += 1

    print(f'  Total candidates: {n_total_candidates}')
    print(f'  Already passing: {n_pass_before}')
    print(f'  Patched (had missing imports): {n_patched_attempted}')
    print(f'  Newly passing after patch: {n_pass_after_patch}')
    print(f'  Conversations with pass@1 flipped F→T: {n_state_pass_changes}')

    # Recompute aggregate stats per persona
    by_p = {}
    for r in detailed:
        by_p.setdefault(r['persona'], {'n': 0, 'p1': 0, 'p3': 0, 'p5': 0})
        by_p[r['persona']]['n'] += 1
        last = r['conversation'][-1]
        pk = last.get('pass_at_k', {})
        if pk.get('pass@1'): by_p[r['persona']]['p1'] += 1
        if pk.get('pass@3'): by_p[r['persona']]['p3'] += 1
        if pk.get('pass@5'): by_p[r['persona']]['p5'] += 1

    print(f'\n  Per-persona AFTER patch:')
    for pn, s in by_p.items():
        print(f'    {pn:22s}: pass@1={s["p1"]}/{s["n"]}={100*s["p1"]/s["n"]:.1f}% pass@5={100*s["p5"]/s["n"]:.1f}%')

    if not dry_run:
        with open(output_path, 'w') as f:
            json.dump(d, f, indent=2)
        print(f'\n  Saved → {output_path}')

    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--test_states", default="data/seeds/test_states_v29_eval_200.jsonl")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    state_tests = load_state_tests(Path(args.test_states))
    print(f"Loaded test cases for {len(state_tests)} states")

    repatch_eval_file(Path(args.input), Path(args.output), state_tests, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
