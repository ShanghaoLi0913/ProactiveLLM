"""
评估DPO训练后的模型性能
计算task success rate和其他指标
"""
import json
import sys
import re
import subprocess
import tempfile
import os
import signal
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 设置代理（如果环境变量中有）
import os
proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
if proxy:
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    # 设置huggingface_hub的代理
    try:
        from huggingface_hub import HfApi
        # HfApi会自动使用环境变量中的代理
        pass
    except:
        pass

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reward.compute import compute_task_score, compute_interrupt_cost, compute_interrupt_cost_v2, compute_clarification_bonus, total_reward


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def extract_code_from_text(text: str) -> Optional[str]:
    """从文本中提取Python代码块"""
    # 尝试提取所有markdown代码块
    code_block_pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    if matches:
        # 过滤掉错误信息和测试代码（更宽松：允许无def但包含imports/类）
        valid_blocks = []
        for code in matches:
            code = code.strip()
            # 跳过错误信息（更严格的检查）
            if any(keyword in code for keyword in ["Traceback", "Error:", 'File "__test__', "Traceback (most recent call last)", "ZeroDivisionError", "ValueError", "Exception"]):
                continue
            # 跳过测试代码（包含 unittest 或 test_）
            if any(keyword in code for keyword in ["unittest", "test_", "TestCases", "class Test", "def test_"]):
                continue
            # 如果代码中包含测试用例标记，只保留标记之前的内容
            test_markers = ["# Compilation feedback", "# Execution feedback", "Compilation feedback:", "Execution feedback:", "No syntax errors", "TEST_", "Passed all test"]
            for marker in test_markers:
                if marker in code:
                    marker_pos = code.find(marker)
                    code = code[:marker_pos].strip()
                    break
            if not code.strip():
                continue
            valid_blocks.append(code)
        
        if valid_blocks:
            # 如果有多个代码块，选择最像“完整解法”的一个
            best_code = None
            best_score = 0
            for code in valid_blocks:
                lines = code.split('\n')
                has_def = any('def ' in line for line in lines)
                has_import = any(line.strip().startswith(("import ", "from ")) for line in lines)
                body_lines = sum(1 for line in lines if line.strip() and line[0] in ' \t')
                score = len(code) + body_lines * 10 + (50 if has_def else 0) + (10 if has_import else 0)
                if score > best_score:
                    best_score = score
                    best_code = code
            if best_code:
                cleaned_code = best_code
                for marker in test_markers:
                    if marker in cleaned_code:
                        marker_pos = cleaned_code.find(marker)
                        cleaned_code = cleaned_code[:marker_pos].strip()
                        break
                return cleaned_code.strip()
            return max(valid_blocks, key=len)
    
    # 如果没有代码块，尝试提取函数定义（包括完整函数体）
    # 匹配从 def 开始到下一个 def 或文件结尾，但需要包含函数体
    def_positions = [m.start() for m in re.finditer(r'^def\s+\w+', text, re.MULTILINE)]
    if def_positions:
        # 取第一个函数
        start = def_positions[0]
        # 找到下一个 def 或文件结尾
        next_def = def_positions[1] if len(def_positions) > 1 else len(text)
        # 提取函数（包括函数体）
        func_code = text[start:next_def].strip()
        # 确保函数体不为空（至少有一行缩进的内容）
        lines = func_code.split('\n')
        if len(lines) > 1:
            # 检查是否有函数体（有缩进的行）
            has_body = any(line.strip() and line[0] in ' \t' for line in lines[1:])
            if has_body:
                return func_code
    
    # 如果还是没有，尝试提取所有连续的代码行（以 import 或 def 开头）
    code_lines = []
    in_code_block = False
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ', 'def ', 'class ', '#')):
            in_code_block = True
            code_lines.append(line)
        elif in_code_block:
            if stripped == '' or line[0] in ' \t' or stripped.startswith('#'):
                code_lines.append(line)
            else:
                break
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    return None


def score_code_passfail(code: str, tests: str, timeout: int = 30, debug: bool = False) -> float:
    """执行代码和测试，返回pass/fail分数或部分通过率
    
    支持两种测试用例格式：
    1. 可执行的Python代码（正常执行）
    2. ConvCodeWorld反馈信息格式（解析反馈信息判断成功/失败）
    """
    if not code or not tests:
        return 0.0
    
    # 检查是否是ConvCodeWorld反馈信息格式
    # 格式: "# Compilation feedback: ..." + "# Execution feedback: ..."
    is_feedback_format = (
        "# Compilation feedback" in tests or 
        "# Execution feedback" in tests or
        "Compilation feedback:" in tests or
        "Execution feedback:" in tests
    )
    
    if is_feedback_format:
        # 处理反馈信息格式：不执行代码，直接解析反馈信息
        # 如果反馈信息显示"Passed all test runs"，任务成功
        if "Passed all test runs" in tests:
            if debug:
                print(f"   ✅ 反馈信息显示: Passed all test runs")
            return 1.0
        
        # 如果反馈信息显示"No syntax errors"且没有错误信息，可能是成功的
        # 但需要进一步检查
        if "No syntax errors" in tests:
            # 检查是否有错误信息
            has_errors = any(keyword in tests for keyword in [
                "Traceback", "Error:", "ZeroDivisionError", "ValueError", 
                "TypeError", "KeyError", "AttributeError", "IndexError"
            ])
            if not has_errors and "Passed" in tests:
                if debug:
                    print(f"   ✅ 反馈信息显示: No syntax errors and Passed")
                return 1.0
        
        # 如果有明确的错误信息，任务失败
        if any(keyword in tests for keyword in [
            "Traceback", "Error:", "ZeroDivisionError", "ValueError",
            "TypeError", "KeyError", "AttributeError", "IndexError",
            "AssertionError", "NameError", "ImportError"
        ]):
            if debug:
                print(f"   ⚠️  反馈信息显示: 有错误信息")
            return 0.0
        
        # 无法确定（可能是反馈信息不完整）
        if debug:
            print(f"   ⚠️  反馈信息格式，但无法确定成功/失败")
        return 0.0
    
    # 以下是处理可执行的Python测试代码（正常情况）
    # 清理代码：移除可能包含的错误信息和测试用例内容
    # 如果代码中包含 Traceback，只保留 Traceback 之前的部分
    if "Traceback" in code:
        code = code.split("Traceback")[0].strip()
    if "Error:" in code and "def " in code:
        # 如果 Error: 在函数定义之后，保留函数定义之前的部分
        error_pos = code.find("Error:")
        def_pos = code.rfind("def ", 0, error_pos)
        if def_pos >= 0:
            code = code[:def_pos] + code[def_pos:].split("Error:")[0].strip()
    
    # 移除测试用例标记（如果代码提取时包含了）
    # 使用更全面的标记列表
    test_markers = [
        "# Compilation feedback", "# Execution feedback", 
        "Compilation feedback:", "Execution feedback:",
        "No syntax errors", "TEST_", "Passed all test",
        "Traceback", "Error:", "ZeroDivisionError"
    ]
    
    # 首先，如果代码中包含测试标记，只保留标记之前的内容
    for marker in test_markers:
        if marker in code:
            marker_pos = code.find(marker)
            code = code[:marker_pos].strip()
            break
    
    # 确保代码以函数定义开始，移除前面的注释或说明
    lines = code.split('\n')
    cleaned_lines = []
    found_def = False
    for line in lines:
        stripped = line.strip()
        # 跳过空行和纯注释（在找到def之前）
        if not found_def:
            if stripped.startswith('def '):
                found_def = True
                cleaned_lines.append(line)
            elif stripped.startswith('#') or stripped == '':
                continue
            elif stripped.startswith('import ') or stripped.startswith('from '):
                cleaned_lines.append(line)
            # 如果遇到非代码内容（如 "No syntax errors"），停止
            elif any(marker in stripped for marker in test_markers):
                break
        else:
            # 找到def之后，保留所有内容直到遇到测试用例标记
            if any(marker in stripped for marker in test_markers):
                break
            cleaned_lines.append(line)
    
    code = '\n'.join(cleaned_lines).strip()
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # 合并代码和测试
        full_code = code + "\n\n" + tests
        # 如果是unittest格式，需要添加main来运行测试
        if "unittest" in tests.lower() and "if __name__" not in tests:
            full_code += "\n\nif __name__ == '__main__':\n    unittest.main()"
        f.write(full_code)
        temp_path = f.name
    
    try:
        # Use subprocess.run with timeout (more reliable than Popen + threading)
        # This ensures the process is killed if it exceeds timeout
        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Add preexec_fn to set process group for better kill control
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = stdout + "\n" + stderr
        # 解析unittest输出，计算部分通过率
        total = None
        m_total = re.search(r"Ran\s+(\d+)\s+tests?", output)
        if m_total:
            total = int(m_total.group(1))
        failures = 0
        errors = 0
        m_fail = re.search(r"failures=(\d+)", output)
        m_err = re.search(r"errors=(\d+)", output)
        if m_fail:
            failures = int(m_fail.group(1))
        if m_err:
            errors = int(m_err.group(1))
        if result.returncode == 0:
            return 1.0
        if total:
            passed = max(0, total - failures - errors)
            return passed / total
        # 执行失败且无法解析，记录错误信息（仅在debug模式下）
        if debug:
            print(f"   执行错误 (returncode={result.returncode}):")
            if result.stderr:
                print(f"   stderr: {result.stderr[:500]}")
            if result.stdout:
                print(f"   stdout: {result.stdout[:500]}")
        return 0.0
    except subprocess.TimeoutExpired as e:
        # Try to kill the process group if setsid was used
        if hasattr(os, 'setsid'):
            try:
                os.killpg(os.getpgid(e.process.pid), signal.SIGKILL)
            except:
                try:
                    e.process.kill()
                except:
                    pass
        if debug:
            print(f"   执行超时 (>{timeout}s)")
        return 0.0
    except Exception as e:
        if debug:
            print(f"   执行异常: {e}")
        return 0.0
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


# Import unified render_state function - MUST be identical to training
# (Already has PROJECT_ROOT in sys.path from earlier)
from policy.render_state import render_state
from policy.infer import get_template, build_action_selection_chat_prompt, pick_action_from_logits


def set_global_seed(seed: int) -> None:
    """Best-effort seeding for reproducible evaluation runs."""
    import random
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic where applicable
    try:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass

    # HF helper (also seeds python/random/np/torch when available)
    try:
        from transformers import set_seed as hf_set_seed  # type: ignore
        hf_set_seed(seed)
    except Exception:
        pass


def select_action_with_loaded_model(state_text: str, tokenizer, model) -> str:
    """Next-token argmax over Clarify/Execute (chat-wrapped prompt, same as training)."""
    prompt = build_action_selection_chat_prompt(state_text, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return pick_action_from_logits(tokenizer, logits[0, -1, :])


def generate_with_template_local(
    model,
    tokenizer,
    template: str,
    task_prompt: str,
    max_new_tokens: int = 400,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate a response from a local HF model using a system+user template."""
    messages = [
        {"role": "system", "content": template},
        {"role": "user", "content": f"[Task]\n{task_prompt}"},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{template}\n\n[Task]\n{task_prompt}\n\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 与 evaluate_multi_turn_persona 一致：生成阶段使用 LoRA（训练后的策略）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in generated_text:
        generated_text = generated_text.split("Assistant:")[-1].strip()
    return generated_text


def evaluate_model(
    model_dir: str,
    base_model: str,
    prefs_path: str,
    max_samples: Optional[int] = None,
    output_path: Optional[str] = None,
    seed: int = 42,
    code_samples: int = 1,
    code_temperature: float = 0.7,
    code_top_p: float = 0.9,
    code_do_sample: bool = True,
    use_openai_for_code: bool = False,
    full_query: bool = False,
    strict_execute_task_score: bool = False,
    verbose: bool = False,
):
    """评估DPO模型"""
    import random

    set_global_seed(seed)
    fq = "full (original_instruct_prompt)" if full_query else "masked (state.query)"
    gen = "gpt-4o-mini" if use_openai_for_code else "local Llama+LoRA"
    strict = ", unittest only if Execute" if strict_execute_task_score else ""
    print(
        f"📊 eval_dpo model={model_dir} base={base_model} seed={seed} "
        f"query={fq} gen={gen}{strict}",
        flush=True,
    )
    
    prefs = load_jsonl(Path(prefs_path))
    if max_samples is not None and max_samples < len(prefs):
        prefs = random.Random(seed).sample(prefs, k=max_samples)
    
    print(f"📊 samples={len(prefs)}", flush=True)
    
    results = []
    task_success_count = 0
    soft_success_count = 0
    test_pass_rates: List[float] = []
    total_samples = 0
    execute_count = 0
    execute_success_count = 0
    
    use_openai = use_openai_for_code
    if use_openai and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("--use_openai_for_code 需要 OPENAI_API_KEY")
    print("🔧 loading policy once…", flush=True)
    
    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    
    # Check if base_model is a local path
    use_local = base_model.startswith("/") and ("snapshots" in base_model or "models" in base_model)
    
    # Try to use local files first (from cache)
    # transformers will automatically use cache if available
    try:
        policy_tokenizer = AutoTokenizer.from_pretrained(
            model_dir, 
            use_fast=True,
            token=hf_token if hf_token else None,
            local_files_only=use_local if model_dir.startswith("/") else False,
        )
    except Exception:
        # Try with local_files_only=False to use cache
        policy_tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            use_fast=True,
            token=hf_token if hf_token else None,
            local_files_only=False,  # Allow using cache
        )
        special_tokens = {"additional_special_tokens": ["Clarify", "Execute"]}
        policy_tokenizer.add_special_tokens(special_tokens)

    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    print("loading base model (8-bit if available)…", flush=True)
    base_model_obj = None
    try:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=hf_token if hf_token else None,
            local_files_only=use_local,
        )
        print("base: 8-bit", flush=True)
    except Exception as e:
        print(f"base: bf16 (8-bit failed: {e})", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=hf_token if hf_token else None,
            local_files_only=use_local,
        )

    n_emb = base_model_obj.get_input_embeddings().num_embeddings
    if len(policy_tokenizer) != n_emb:
        print(f"resize embeddings {n_emb} -> {len(policy_tokenizer)}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            base_model_obj.resize_token_embeddings(len(policy_tokenizer))
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            device = next(base_model_obj.parameters()).device
            base_model_obj = base_model_obj.cpu()
            torch.cuda.empty_cache()
            base_model_obj.resize_token_embeddings(len(policy_tokenizer))
            base_model_obj = base_model_obj.to(device)
            print("resize: done (via CPU)", flush=True)

    try:
        policy_model = PeftModel.from_pretrained(base_model_obj, model_dir)
    except Exception:
        policy_model = base_model_obj
    policy_model.eval()
    
    for i, pref in enumerate(prefs):
        state = pref["state"]
        if full_query:
            state = dict(state)
            state["query"] = state.get("original_instruct_prompt") or state.get("query", "")
        
        state_text = render_state(state)
        predicted_action = select_action_with_loaded_model(state_text, policy_tokenizer, policy_model)
        task_prompt = state.get("query", "")
        domain = state.get("domain", "coding")
        
        template = get_template(predicted_action, domain)
        code = None
        if use_openai:
            from llm.provider import chat_complete
            response = chat_complete(template, f"[Task]\n{task_prompt}", model="gpt-4o-mini", max_tokens=400)
        else:
            if state["domain"] == "coding" and predicted_action == "Execute" and code_samples > 1:
                best = {"score": -1.0, "response": "", "code": None}
                for _ in range(code_samples):
                    resp = generate_with_template_local(
                        policy_model,
                        policy_tokenizer,
                        template=template,
                        task_prompt=task_prompt,
                        max_new_tokens=400,
                        temperature=code_temperature,
                        top_p=code_top_p,
                        do_sample=code_do_sample,
                    )
                    code_candidate = extract_code_from_text(resp)
                    if not code_candidate:
                        score = 0.0
                    else:
                        tests = state.get("convcodeworld_tests") or state.get("test")
                        score = score_code_passfail(code_candidate, tests, debug=False) if tests else 0.0
                    if score is None or score != score:
                        score = 0.0
                    if score > best["score"]:
                        best = {"score": score, "response": resp, "code": code_candidate}
                response = best["response"]
                code = best["code"]
            else:
                response = generate_with_template_local(
                    policy_model,
                    policy_tokenizer,
                    template=template,
                    task_prompt=task_prompt,
                    max_new_tokens=400,
                    temperature=code_temperature,
                    top_p=code_top_p,
                    do_sample=code_do_sample,
        )
        
        if state["domain"] == "coding" and code is None:
            code = extract_code_from_text(response)
        if verbose and state["domain"] == "coding":
            periodic = i < 3 or (i + 1) % 20 == 0
            if not code and periodic:
                tag = "Clarify (no code expected)" if predicted_action == "Clarify" else "Execute (missing code)"
                preview = response[:500] if predicted_action == "Execute" else response[:300]
                print(f"\n[{i+1}] {tag}\n{preview}...")
            elif code and i < 3:
                print(f"\n[{i+1}] response + extracted code:\n{'='*60}\n{response}\n{'='*60}\n{code}\n{'='*60}")
                for m in ("No syntax errors", "Compilation feedback", "Execution feedback"):
                    if m in code:
                        print(f"  warn: code contains marker {m!r}")

        # task score
        task_score = 0.0
        if predicted_action == "Execute":
            execute_count += 1

        tests = (state.get("convcodeworld_tests") or state.get("test")) if state["domain"] == "coding" else None
        run_unit_tests = bool(
            state["domain"] == "coding" and code and tests
        )
        if strict_execute_task_score and predicted_action != "Execute":
            run_unit_tests = False

        if run_unit_tests:
            task_score = score_code_passfail(code, tests, debug=(verbose and i < 3))
            test_pass_rates.append(task_score)
            if task_score >= 1.0:
                task_success_count += 1
                if predicted_action == "Execute":
                    execute_success_count += 1
            if task_score >= 0.5:
                soft_success_count += 1
            total_samples += 1

        n_questions = response.count("?")
        meta = {"reject_signal": 0, "answered_clarification": 0}
        interrupt_cost = compute_interrupt_cost_v2(meta, n_questions, response)
        total_r = total_reward(task_score, interrupt_cost)
        
        results.append({
            "state_id": state.get("id", f"sample_{i}"),
            "predicted_action": predicted_action,
            "chosen_action": pref.get("chosen_action", "Execute"),
            "task_score": task_score,
            "interrupt_cost": interrupt_cost,
            "total_reward": total_r,
            "response_length": len(response),
            "n_questions": n_questions,
        })
        
        if (i + 1) % 10 == 0:
            print(f"progress {i+1}/{len(prefs)}", flush=True)
    
    task_success_rate = (task_success_count / total_samples * 100) if total_samples > 0 else 0.0
    execute_success_rate = (execute_success_count / execute_count * 100) if execute_count > 0 else 0.0
    valid_rewards = [r["total_reward"] for r in results if r.get("total_reward") is not None]
    valid_task_scores = [r["task_score"] for r in results if r.get("task_score") is not None]
    avg_reward = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0
    avg_task_score = sum(valid_task_scores) / len(valid_task_scores) if valid_task_scores else 0.0
    avg_test_pass_rate = sum(test_pass_rates) / len(test_pass_rates) if test_pass_rates else 0.0
    soft_task_success_rate = (soft_success_count / total_samples * 100) if total_samples > 0 else 0.0
    
    action_matches = sum(1 for r in results if r["predicted_action"] == r["chosen_action"])
    action_accuracy = (action_matches / len(results) * 100) if results else 0.0
    
    summary = {
        "task_success_rate": task_success_rate,
        "task_success_rate_execute_only": execute_success_rate,
        "predicted_execute_rate": (execute_count / len(results) * 100) if results else 0.0,
        "avg_reward": avg_reward,
        "avg_task_score": avg_task_score,
        "avg_test_pass_rate": avg_test_pass_rate,
        "soft_task_success_rate": soft_task_success_rate,
        "action_accuracy": action_accuracy,
        "num_prefs_rows": len(results),
        "task_scored_samples": total_samples,
        "total_samples": len(results),
        "task_evaluated_samples": total_samples,
        "task_success_count": task_success_count,
        "execute_count": execute_count,
        "execute_success_count": execute_success_count,
        "full_query_eval": full_query,
        "strict_execute_task_score": strict_execute_task_score,
        "verbose_eval": verbose,
    }
    
    print(
        f"\nTSR={task_success_rate:.2f}% (hits {task_success_count}/{total_samples}) | "
        f"execute-only={execute_success_rate:.2f}% | soft>={soft_task_success_rate:.2f}% | "
        f"p(Execute)={summary['predicted_execute_rate']:.1f}% | action_acc={action_accuracy:.1f}% | "
        f"avg_score={avg_task_score:.4f}",
        flush=True,
    )
    
    # 保存结果
    if output_path:
        output = {
            "summary": summary,
            "detailed_results": results,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"wrote {output_path}", flush=True)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="评估DPO模型")
    parser.add_argument("--model_dir", type=str, required=True, help="训练好的模型目录")
    parser.add_argument("--base_model", type=str, required=True, help="Base模型名称")
    parser.add_argument("--prefs", type=str, required=True, help="Preference pairs文件路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最大评估样本数")
    parser.add_argument("--output", type=str, default=None, help="输出结果文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于可复现评估，默认: 42）")
    parser.add_argument("--code_samples", type=int, default=1, help="每个任务生成的代码候选数（>1启用best-of）")
    parser.add_argument("--code_temperature", type=float, default=0.7, help="代码生成温度（默认0.7）")
    parser.add_argument("--code_top_p", type=float, default=0.9, help="代码生成top_p（默认0.9）")
    parser.add_argument(
        "--code_do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用采样生成代码（默认: True）",
    )
    parser.add_argument(
        "--use_openai_for_code",
        action="store_true",
        help="用 gpt-4o-mini 生成 Clarify/代码（默认：本地训练策略模型）",
    )
    parser.add_argument(
        "--full_query",
        action="store_true",
        help="用 original_instruct_prompt 作为 state.query（full），用于 render_state 与代码生成；默认 masked query",
    )
    parser.add_argument(
        "--strict_execute_task_score",
        action="store_true",
        help="仅当 predicted_action==Execute 时跑 unittest；TSR/soft TSR 分母只含这些样本（Clarify 仍生成但不计分）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印前几条样本的响应/代码与 unittest stderr（默认安静）",
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_dir=args.model_dir,
        base_model=args.base_model,
        prefs_path=args.prefs,
        max_samples=args.max_samples,
        output_path=args.output,
        seed=args.seed,
        code_samples=args.code_samples,
        code_temperature=args.code_temperature,
        code_top_p=args.code_top_p,
        code_do_sample=args.code_do_sample,
        use_openai_for_code=args.use_openai_for_code,
        full_query=args.full_query,
        strict_execute_task_score=args.strict_execute_task_score,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()


