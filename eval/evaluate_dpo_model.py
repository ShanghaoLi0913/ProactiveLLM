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
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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
        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
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
    except subprocess.TimeoutExpired:
        if debug:
            print(f"   执行超时 (>{timeout}s)")
        return 0.0
    except Exception as e:
        if debug:
            print(f"   执行异常: {e}")
        return 0.0
    finally:
        import os
        try:
            os.unlink(temp_path)
        except:
            pass


# Import unified render_state function - MUST be identical to training
# (Already has PROJECT_ROOT in sys.path from earlier)
from policy.render_state import render_state
from policy.infer import get_template


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
    """Select action using logits on the next token, without re-loading models per sample."""
    inputs = tokenizer(state_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        next_token_logits = logits[0, -1, :]

        action_tokens = ["Clarify", "Execute"]
        action_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in action_tokens]

        valid_actions = []
        valid_ids = []
        for token, token_id in zip(action_tokens, action_token_ids):
            if token_id is not None:
                valid_actions.append(token)
                valid_ids.append(token_id)

        if not valid_ids:
            return "Execute"

        action_logits = next_token_logits[valid_ids]
        best_action_idx = torch.argmax(action_logits).item()
        return valid_actions[best_action_idx]


def generate_with_template_local(
    model,
    tokenizer,
    template: str,
    task_prompt: str,
    max_new_tokens: int = 400,
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

    # For PeftModel, disable adapter during code generation to keep "separated" behavior.
    adapter_ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()

    with torch.no_grad(), adapter_ctx:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in generated_text:
        generated_text = generated_text.split("Assistant:")[-1].strip()
    return generated_text


def generate_response(model, tokenizer, prompt: str, max_length: int = 2048) -> str:
    """使用模型生成响应"""
    # 对于Instruct模型，使用chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 移除prompt部分
    if prompt in generated_text:
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()
    return response


def extract_action_from_response(response: str, state: Dict) -> str:
    """从响应中提取action (Clarify/Execute)"""
    response_lower = response.lower()
    
    # 检查是否包含明确的action标记
    if "action:" in response_lower or "proactivity:" in response_lower:
        for action in ["Clarify", "Execute"]:
            if action in response:
                return action
    
    # 基于内容推断
    question_count = response.count("?")
    if question_count >= 1:
        return "Clarify"
    elif "code" in response_lower or "solution" in response_lower or "```" in response:
        return "Execute"
    else:
        return "Execute"  # 默认执行


def evaluate_model(
    model_dir: str,
    base_model: str,
    prefs_path: str,
    max_samples: Optional[int] = None,
    output_path: Optional[str] = None,
    seed: int = 42,
):
    """评估DPO模型"""
    print(f"📊 加载模型: {model_dir}")
    print(f"📊 Base模型: {base_model}")
    print(f"🎲 Seed: {seed}")
    set_global_seed(seed)
    
    # Scheme A: Separated Architecture
    # Policy model only predicts action, code generation is separate
    print("📋 使用分离架构 (Scheme A)")
    print("   - Policy模型: 预测action (Clarify/Execute)")
    print("   - Code生成: 使用独立模型（不受DPO影响）")
    
    # 加载测试数据
    prefs = load_jsonl(Path(prefs_path))
    # IMPORTANT:
    # Do NOT take the first N examples — prefs JSONL ordering is arbitrary and can heavily
    # bias results. Instead, sample deterministically using the evaluation seed.
    if max_samples:
        if max_samples >= len(prefs):
            pass
        else:
            import random
            rng = random.Random(seed)
            prefs = rng.sample(prefs, k=max_samples)
    
    print(f"📊 评估 {len(prefs)} 个样本", flush=True)
    
    results = []
    task_success_count = 0
    soft_success_count = 0
    test_pass_rates: List[float] = []
    total_samples = 0
    execute_count = 0
    execute_success_count = 0
    
    # Code generation strategy:
    # 1. If OpenAI API is available, use it (higher quality)
    # 2. Otherwise, use base Llama model (no API needed)
    use_openai = os.environ.get("OPENAI_API_KEY") is not None
    if use_openai:
        print("✅ 使用OpenAI API进行代码生成")
        code_model_name = None
    else:
        print("✅ 使用Base Llama模型进行代码生成（不需要API）")
        code_model_name = base_model  # Use base Llama model for code generation

    # Load policy model ONCE (previously re-loaded per sample, making eval extremely slow and noisy)
    print("🔧 预加载Policy模型（一次加载，循环复用）", flush=True)
    try:
        policy_tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception:
        policy_tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        special_tokens = {"additional_special_tokens": ["Clarify", "Execute"]}
        policy_tokenizer.add_special_tokens(special_tokens)

    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if len(policy_tokenizer) != base_model_obj.get_input_embeddings().num_embeddings:
        base_model_obj.resize_token_embeddings(len(policy_tokenizer))

    try:
        policy_model = PeftModel.from_pretrained(base_model_obj, model_dir)
    except Exception:
        policy_model = base_model_obj
    policy_model.eval()
    
    for i, pref in enumerate(prefs):
        state = pref["state"]
        
        # Scheme A: Separated Architecture
        # Step 1: Predict action using policy model
        state_text = render_state(state)
        predicted_action = select_action_with_loaded_model(state_text, policy_tokenizer, policy_model)
        
        # Step 2: Generate code using separate code generation
        task_prompt = state.get("query", "")
        domain = state.get("domain", "coding")

        template = get_template(predicted_action, domain)
        if use_openai:
            from llm.provider import chat_complete
            response = chat_complete(template, f"[Task]\n{task_prompt}", model="gpt-4o-mini", max_tokens=400)
        else:
            response = generate_with_template_local(
                policy_model,
                policy_tokenizer,
                template=template,
                task_prompt=task_prompt,
                max_new_tokens=400,
            )
        
        # 提取代码（如果是coding任务）
        code = None
        if state["domain"] == "coding":
            code = extract_code_from_text(response)
            # 调试信息：记录代码提取情况
            if not code and (i < 3 or (i + 1) % 20 == 0):
                # 根据predicted_action判断：Clarify action是问问题，这是正常的
                if predicted_action == "Clarify":
                    print(f"\n📋 样本 {i+1}: 预测Clarify action（问问题）")
                    print(f"   响应类型: 澄清问题（正常行为）")
                    print(f"   响应预览: {response[:300]}...")
                else:
                    # Execute action应该生成代码，如果没有代码才是问题
                    print(f"\n⚠️  样本 {i+1}: Execute action但未提取到代码")
                    print(f"   响应长度: {len(response)}")
                    print(f"   响应预览: {response[:500]}...")
            elif code and i < 3:
                # 对前3个样本，显示完整响应以便调试
                print(f"\n📝 样本 {i+1} 完整响应:")
                print("="*80)
                print(response)
                print("="*80)
                print(f"\n📦 提取的代码（清理后）:")
                print("="*80)
                print(code)
                print("="*80)
                # 检查代码中是否还有测试标记
                test_markers = ["No syntax errors", "Compilation feedback", "Execution feedback"]
                has_markers = any(marker in code for marker in test_markers)
                if has_markers:
                    print(f"⚠️  警告：提取的代码中仍然包含测试标记！")
                    for marker in test_markers:
                        if marker in code:
                            print(f"   包含标记: {marker}")
                else:
                    print(f"✅ 代码已成功清理，不包含测试标记")
                print("="*80)
        
        # 计算task score
        task_score = 0.0
        if predicted_action == "Execute":
            execute_count += 1
        if state["domain"] == "coding" and code:
            tests = state.get("convcodeworld_tests")
            if tests:
                task_score = score_code_passfail(code, tests, debug=(i < 3))
                test_pass_rates.append(task_score)
                if task_score > 0:
                    task_success_count += 1
                    if predicted_action == "Execute":
                        execute_success_count += 1
                if task_score >= 0.5:
                    soft_success_count += 1
                elif i < 3 or (i + 1) % 20 == 0:
                    print(f"\n⚠️  样本 {i+1}: 代码执行失败 (score=0)")
                    print(f"   提取的代码长度: {len(code)}")
                    print(f"   代码预览: {code[:300]}...")
                    # 显示完整代码（前3个样本）
                    if i < 3:
                        print(f"   完整代码:\n{code}")
                        print(f"   测试用例长度: {len(tests)}")
                total_samples += 1
        elif state["domain"] == "coding" and not code:
            # 记录没有提取到代码的情况
            if i < 3 or (i + 1) % 20 == 0:
                # 根据predicted_action判断：Clarify action是问问题，这是正常的
                if predicted_action == "Clarify":
                    print(f"\n📋 样本 {i+1}: 预测Clarify action（问问题）")
                    print(f"   响应类型: 澄清问题（正常行为，task_score=0）")
                    print(f"   响应预览: {response[:300]}...")
                else:
                    # Execute action应该生成代码
                    print(f"\n⚠️  样本 {i+1}: Execute action但未提取到代码")
                    print(f"   响应长度: {len(response)}")
                    print(f"   响应预览: {response[:300]}...")
        
        # 计算interrupt cost（使用新公式）
        # C_Interrupt = Σ_{t=1}^{T} (δb_t r_t + λb_t - γb_t a_t)
        n_questions = response.count("?")
        # 评估时没有user_reaction，假设既没有answered也没有rejected
        meta = {"reject_signal": 0, "answered_clarification": 0}
        interrupt_cost = compute_interrupt_cost_v2(meta, n_questions, response)
        
        # 总reward（新公式：R = R_task - C_interrupt）
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
            print(f"  处理进度: {i+1}/{len(prefs)}", flush=True)
    
    # 计算统计信息
    task_success_rate = (task_success_count / total_samples * 100) if total_samples > 0 else 0.0
    execute_success_rate = (execute_success_count / execute_count * 100) if execute_count > 0 else 0.0
    avg_reward = sum(r["total_reward"] for r in results) / len(results) if results else 0.0
    avg_task_score = sum(r["task_score"] for r in results) / len(results) if results else 0.0
    avg_test_pass_rate = sum(test_pass_rates) / len(test_pass_rates) if test_pass_rates else 0.0
    soft_task_success_rate = (soft_success_count / total_samples * 100) if total_samples > 0 else 0.0
    
    # Action准确率
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
        "total_samples": len(results),
        "task_evaluated_samples": total_samples,
        "task_success_count": task_success_count,
        "execute_count": execute_count,
        "execute_success_count": execute_success_count,
    }
    
    print("\n" + "="*50)
    print("📊 评估结果:")
    print(f"  Task Success Rate: {task_success_rate:.2f}%")
    print(f"  Task Success (Execute Only): {execute_success_rate:.2f}%")
    print(f"  Soft Task Success (>=50% tests): {soft_task_success_rate:.2f}%")
    print(f"  Predicted Execute Rate: {summary['predicted_execute_rate']:.2f}%")
    print(f"  Average Reward: {avg_reward:.4f}")
    print(f"  Average Task Score: {avg_task_score:.4f}")
    print(f"  Avg Test Pass Rate: {avg_test_pass_rate:.4f}")
    print(f"  Action Accuracy: {action_accuracy:.2f}%")
    print("="*50)
    
    # 保存结果
    if output_path:
        output = {
            "summary": summary,
            "detailed_results": results,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已保存到: {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="评估DPO模型")
    parser.add_argument("--model_dir", type=str, required=True, help="训练好的模型目录")
    parser.add_argument("--base_model", type=str, required=True, help="Base模型名称")
    parser.add_argument("--prefs", type=str, required=True, help="Preference pairs文件路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最大评估样本数")
    parser.add_argument("--output", type=str, default=None, help="输出结果文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于可复现评估，默认: 42）")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_dir=args.model_dir,
        base_model=args.base_model,
        prefs_path=args.prefs,
        max_samples=args.max_samples,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


