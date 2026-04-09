"""
快速测试：Base Llama 在 unmasked (full_query) prompt 下，v28 20个task的 pass rate。
对比：masked prompt = 0%，目的是确认是 masking 的锅还是模型能力不足。

运行：
    cd /root/autodl-tmp/ProactiveLLM
    python scripts/test_base_llama_unmasked.py
"""

import json
import sys
import subprocess
import tempfile
import os
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EXECUTE_PROMPT = (PROJECT_ROOT / "prompts/coding_execute.txt").read_text().strip()

V28_IDS = [
    'BigCodeBench/1007','BigCodeBench/1066','BigCodeBench/1081','BigCodeBench/1133',
    'BigCodeBench/138','BigCodeBench/409','BigCodeBench/414','BigCodeBench/415',
    'BigCodeBench/463','BigCodeBench/471','BigCodeBench/478','BigCodeBench/484',
    'BigCodeBench/516','BigCodeBench/593','BigCodeBench/604','BigCodeBench/615',
    'BigCodeBench/630','BigCodeBench/678','BigCodeBench/873','BigCodeBench/963',
]


def extract_code(text: str) -> str:
    # 提取 ```python ... ``` 或 ``` ... ```
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_test(code: str, task: dict) -> bool:
    full = task["complete_prompt"] + "\n" + code + "\n" + task["test"]
    full += "\nimport unittest; r=unittest.main(exit=False,verbosity=0); exit(0 if r.result.wasSuccessful() else 1)"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        tmp = f.name
    try:
        result = subprocess.run(
            ["python3", tmp], capture_output=True, timeout=15, text=True
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(tmp)


def generate(model, tokenizer, task_prompt: str) -> str:
    messages = [
        {"role": "system", "content": EXECUTE_PROMPT},
        {"role": "user", "content": f"[Task]\n{task_prompt}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def main():
    # 加载 full_query states
    with open(PROJECT_ROOT / "data/dpo/test_states_full_query.jsonl") as f:
        full_states = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}

    # 加载 masked states (for comparison)
    with open(PROJECT_ROOT / "data/dpo/test_states_clean_for_eval.jsonl") as f:
        masked_states = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}

    # 加载 BigCodeBench test cases
    with open(PROJECT_ROOT / "data/external/BigCodeBench/v0.1.4.jsonl") as f:
        bcb = {json.loads(l)["task_id"]: json.loads(l) for l in f if l.strip()}

    # 加载 base model
    print(f"Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model.eval()
    print("Model loaded.\n")

    results = []
    for tid in V28_IDS:
        if tid not in full_states or tid not in bcb:
            print(f"SKIP {tid} (not found)")
            continue

        task = bcb[tid]
        full_query = full_states[tid]["query"]
        masked_query = masked_states.get(tid, {}).get("query", "")

        # Generate with full (unmasked) query
        code_text = generate(model, tokenizer, full_query)
        code = extract_code(code_text)
        passed = run_test(code, task)

        results.append({"task_id": tid, "passed": passed})
        print(f"  {'✓' if passed else '✗'} {tid}")

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n=== Base Llama, Unmasked prompt ===")
    print(f"Pass@1: {passed_count}/{total} = {passed_count/total*100:.1f}%")
    print(f"\n参考: masked prompt 下 Base Llama = 0/20 = 0%")
    print(f"      masked prompt 下 v28 DPO Busy = 1/20 = 5%")

    out = PROJECT_ROOT / "outputs/base_llama_unmasked_20tasks.json"
    with open(out, "w") as f:
        json.dump({"pass_rate": passed_count / total, "passed": passed_count, "total": total, "results": results}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
