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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reward.compute import compute_task_score, compute_interrupt_cost, total_reward


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
        # 过滤掉错误信息和测试代码
        valid_blocks = []
        for code in matches:
            code = code.strip()
            # 跳过错误信息（更严格的检查）
            if any(keyword in code for keyword in ["Traceback", "Error:", 'File "__test__', "Traceback (most recent call last)", "ZeroDivisionError", "ValueError", "Exception"]):
                continue
            # 跳过测试代码（包含 unittest 或 test_）
            if any(keyword in code for keyword in ["unittest", "test_", "TestCases", "class Test", "def test_"]):
                continue
            # 确保代码包含函数定义
            if "def " not in code:
                continue
            valid_blocks.append(code)
        
        if valid_blocks:
            # 如果有多个代码块，选择最长的（通常是最完整的）
            # 或者选择包含函数体的那个
            best_code = None
            best_score = 0
            
            for code in valid_blocks:
                # 计算分数：函数体行数
                lines = code.split('\n')
                # 检查是否有函数定义
                has_def = any('def ' in line for line in lines)
                if not has_def:
                    continue
                
                # 计算函数体行数（缩进的行）
                body_lines = sum(1 for line in lines if line.strip() and line[0] in ' \t')
                score = len(code) + body_lines * 10  # 长度 + 函数体行数权重
                
                if score > best_score:
                    best_score = score
                    best_code = code
            
            if best_code:
                return best_code
            # 如果没有找到最好的，返回最长的
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
    """执行代码和测试，返回pass/fail分数"""
    if not code or not tests:
        return 0.0
    
    # 清理代码：移除可能包含的错误信息
    # 如果代码中包含 Traceback，只保留 Traceback 之前的部分
    if "Traceback" in code:
        code = code.split("Traceback")[0].strip()
    if "Error:" in code and "def " in code:
        # 如果 Error: 在函数定义之后，保留函数定义之前的部分
        error_pos = code.find("Error:")
        def_pos = code.rfind("def ", 0, error_pos)
        if def_pos >= 0:
            code = code[:def_pos] + code[def_pos:].split("Error:")[0].strip()
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # 合并代码和测试
        full_code = code + "\n\n" + tests
        f.write(full_code)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            return 1.0
        else:
            # 执行失败，记录错误信息（仅在debug模式下）
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
from policy.infer import select_action, execute_action


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
    """从响应中提取action (LOW/MID/HIGH)"""
    response_lower = response.lower()
    
    # 检查是否包含明确的action标记
    if "action:" in response_lower or "proactivity:" in response_lower:
        for action in ["LOW", "MID", "HIGH"]:
            if action in response.upper():
                return action
    
    # 基于内容推断
    question_count = response.count("?")
    if question_count >= 2:
        return "HIGH"
    elif question_count == 1:
        return "MID"
    elif "code" in response_lower or "solution" in response_lower or "```" in response:
        return "LOW"
    else:
        return "MID"  # 默认


def evaluate_model(
    model_dir: str,
    base_model: str,
    prefs_path: str,
    max_samples: Optional[int] = None,
    output_path: Optional[str] = None
):
    """评估DPO模型"""
    print(f"📊 加载模型: {model_dir}")
    print(f"📊 Base模型: {base_model}")
    
    # Scheme A: Separated Architecture
    # Policy model only predicts action, code generation is separate
    print("📋 使用分离架构 (Scheme A)")
    print("   - Policy模型: 预测action (LOW/MID/HIGH)")
    print("   - Code生成: 使用独立模型（不受DPO影响）")
    
    # 加载测试数据
    prefs = load_jsonl(Path(prefs_path))
    if max_samples:
        prefs = prefs[:max_samples]
    
    print(f"📊 评估 {len(prefs)} 个样本", flush=True)
    
    results = []
    task_success_count = 0
    total_samples = 0
    
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
    
    for i, pref in enumerate(prefs):
        state = pref["state"]
        
        # Scheme A: Separated Architecture
        # Step 1: Predict action using policy model
        state_text = render_state(state)
        predicted_action = select_action(state_text, model_dir, base_model)
        
        # Step 2: Generate code using separate code generation
        task_prompt = state.get("query", "")
        domain = state.get("domain", "coding")
        
        response = execute_action(
            predicted_action,
            task_prompt,
            domain,
            code_model_name=code_model_name,  # Use base Llama model if no API
            use_openai=use_openai
        )
        
        # 提取代码（如果是coding任务）
        code = None
        if state["domain"] == "coding":
            code = extract_code_from_text(response)
            # 调试信息：记录代码提取情况
            if not code and (i < 3 or (i + 1) % 20 == 0):
                # 根据predicted_action判断：HIGH action是问问题，这是正常的
                if predicted_action == "HIGH":
                    print(f"\n📋 样本 {i+1}: 预测HIGH action（问问题）")
                    print(f"   响应类型: 澄清问题（正常行为）")
                    print(f"   响应预览: {response[:300]}...")
                elif predicted_action == "MID":
                    print(f"\n📋 样本 {i+1}: 预测MID action（问一个问题）")
                    print(f"   响应类型: 澄清问题（正常行为）")
                    print(f"   响应预览: {response[:300]}...")
                else:
                    # LOW action应该生成代码，如果没有代码才是问题
                    print(f"\n⚠️  样本 {i+1}: LOW action但未提取到代码")
                    print(f"   响应长度: {len(response)}")
                    print(f"   响应预览: {response[:500]}...")
            elif code and i < 3:
                # 对前3个样本，显示完整响应以便调试
                print(f"\n📝 样本 {i+1} 完整响应:")
                print("="*80)
                print(response)
                print("="*80)
                print(f"\n📦 提取的代码:")
                print("="*80)
                print(code)
                print("="*80)
        
        # 计算task score
        task_score = 0.0
        if state["domain"] == "coding" and code:
            tests = state.get("convcodeworld_tests")
            if tests:
                task_score = score_code_passfail(code, tests, debug=(i < 3))
                if task_score > 0:
                    task_success_count += 1
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
                # 根据predicted_action判断：HIGH/MID action是问问题，这是正常的
                if predicted_action in ["HIGH", "MID"]:
                    print(f"\n📋 样本 {i+1}: 预测{predicted_action} action（问问题）")
                    print(f"   响应类型: 澄清问题（正常行为，task_score=0）")
                    print(f"   响应预览: {response[:300]}...")
                else:
                    # LOW action应该生成代码
                    print(f"\n⚠️  样本 {i+1}: LOW action但未提取到代码")
                    print(f"   响应长度: {len(response)}")
                    print(f"   响应预览: {response[:300]}...")
        
        # 计算interrupt cost（简化版）
        n_questions = response.count("?")
        length_tokens = len(response.split())
        meta = {"reject_signal": 0, "off_topic": 0}
        interrupt_cost = compute_interrupt_cost(meta, n_questions, length_tokens, 0)
        
        # 总reward
        total_r = total_reward(task_score, interrupt_cost)
        
        results.append({
            "state_id": state.get("id", f"sample_{i}"),
            "predicted_action": predicted_action,
            "chosen_action": pref.get("chosen_action", "MID"),
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
    avg_reward = sum(r["total_reward"] for r in results) / len(results) if results else 0.0
    avg_task_score = sum(r["task_score"] for r in results) / len(results) if results else 0.0
    
    # Action准确率
    action_matches = sum(1 for r in results if r["predicted_action"] == r["chosen_action"])
    action_accuracy = (action_matches / len(results) * 100) if results else 0.0
    
    summary = {
        "task_success_rate": task_success_rate,
        "avg_reward": avg_reward,
        "avg_task_score": avg_task_score,
        "action_accuracy": action_accuracy,
        "total_samples": len(results),
        "task_evaluated_samples": total_samples,
        "task_success_count": task_success_count,
    }
    
    print("\n" + "="*50)
    print("📊 评估结果:")
    print(f"  Task Success Rate: {task_success_rate:.2f}%")
    print(f"  Average Reward: {avg_reward:.4f}")
    print(f"  Average Task Score: {avg_task_score:.4f}")
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
    
    args = parser.parse_args()
    
    evaluate_model(
        model_dir=args.model_dir,
        base_model=args.base_model,
        prefs_path=args.prefs,
        max_samples=args.max_samples,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()


