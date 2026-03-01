"""
单轮评估：直接用base模型生成代码（不经过Clarify/Execute决策）
用于测试base模型在masked query下的代码生成能力
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# 确保输出不被缓冲
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate_dpo_model import extract_code_from_text, score_code_passfail


def load_jsonl(path: Path) -> List[Dict]:
    """加载JSONL文件"""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def generate_code_directly(
    model,
    tokenizer,
    masked_query: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """直接用base模型生成代码（不经过Clarify/Execute决策）"""
    
    # 构建prompt
    prompt = f"[Task]\n{masked_query}\n\nAssistant:"
    
    # 使用chat template（如果可用）
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": f"[Task]\n{masked_query}"}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 移除prompt部分
    if prompt in generated_text:
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()
    
    return response


def evaluate_base_model_single_turn(
    base_model: str,
    test_states_path: str,
    max_samples: Optional[int] = None,
    output_path: Optional[str] = None,
    seed: int = 42,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
    use_original_query: bool = False,
):
    """评估base模型单轮代码生成能力"""
    
    print("=" * 80)
    print("🔍 Base模型单轮代码生成评估")
    print("=" * 80)
    print(f"Base模型: {base_model}")
    print(f"测试数据: {test_states_path}")
    print(f"最大样本数: {max_samples if max_samples else '全部'}")
    print(f"使用Original Query: {'是' if use_original_query else '否 (使用Masked Query)'}")
    print(f"Seed: {seed}")
    print("")
    
    # 加载测试数据
    print(f"📂 加载测试数据...")
    test_states = load_jsonl(Path(test_states_path))
    print(f"✅ 加载了 {len(test_states)} 个测试样本")
    
    # 采样
    if max_samples and max_samples < len(test_states):
        import random
        rng = random.Random(seed)
        test_states = rng.sample(test_states, max_samples)
        print(f"📊 采样了 {len(test_states)} 个样本")
    
    # 加载模型
    print(f"\n🔄 加载base模型...")
    hf_token = os.getenv("HF_TOKEN")
    
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=hf_token if hf_token else None,
        )
    except Exception as e:
        print(f"⚠️  8-bit量化失败 ({e}), 使用bfloat16")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=hf_token if hf_token else None,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
        token=hf_token if hf_token else None,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print("✅ 模型加载完成")
    
    # 评估
    print(f"\n🔍 开始评估...")
    results = []
    task_success_count = 0
    task_evaluated_count = 0
    total_task_score = 0.0
    
    for i, state in enumerate(test_states):
        state_id = state.get("id", f"unknown_{i}")
        
        # 根据参数选择使用masked query还是original query
        if use_original_query:
            query = state.get("original_instruct_prompt", state.get("query", ""))
            query_type = "original_instruct_prompt"
        else:
            query = state.get("query", "")
            query_type = "masked_query"
        
        if not query:
            print(f"⚠️  样本 {i+1}: 没有{query_type}，跳过")
            continue
        
        print(f"\n样本 {i+1}/{len(test_states)}: {state_id}", flush=True)
        print(f"  使用: {query_type}", flush=True)
        print(f"  Query长度: {len(query)}", flush=True)
        
        # 生成代码
        try:
            response = generate_code_directly(
                model,
                tokenizer,
                query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            # 提取代码
            code = extract_code_from_text(response)
            
            # 评估代码
            task_score = 0.0
            task_completed = False
            
            # 安全地检查代码是否存在
            has_code = code is not None and len(code) > 0 if code else False
            
            if has_code:
                tests = state.get("convcodeworld_tests") or state.get("test")
                if tests:
                    task_score = score_code_passfail(code, tests, timeout=30, debug=(i < 2))
                    # 确保task_score不是None
                    if task_score is None:
                        task_score = 0.0
                    task_completed = (task_score >= 1.0)
                    task_evaluated_count += 1
                    total_task_score += task_score
                    
                    if task_completed:
                        task_success_count += 1
                    
                    # 安全地计算代码长度
                    try:
                        if code is not None:
                            code_len = len(code)
                        else:
                            code_len = 0
                        print(f"  ✅ 提取到代码 (长度: {code_len})")
                    except Exception as e:
                        print(f"  ✅ 提取到代码 (长度: 未知, 错误: {e})")
                        code_len = 0
                    
                    # 安全地打印task_score
                    try:
                        print(f"  Task Score: {task_score:.3f} {'✅' if task_completed else '❌'}")
                    except Exception as e:
                        print(f"  Task Score: 错误 ({e})")
                else:
                    print(f"  ⚠️  没有测试用例")
            else:
                print(f"  ❌ 未提取到代码")
            
            # 记录结果
            result = {
                "state_id": state_id,
                "query_type": query_type,
                "query": query[:200] + "..." if len(query) > 200 else query,
                "response": response[:500] + "..." if len(response) > 500 else response,
                "has_code": code is not None,
                "code_length": len(code) if code else 0,
                "task_score": task_score,
                "task_completed": task_completed,
            }
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "state_id": state_id,
                "error": str(e),
            })
    
    # 计算统计
    print("\n" + "=" * 80)
    print("📊 评估结果")
    print("=" * 80)
    
    print(f"总样本数: {len(test_states)}")
    print(f"评估样本数: {task_evaluated_count}")
    print(f"有代码的样本数: {sum(1 for r in results if r.get('has_code', False))}")
    
    if task_evaluated_count > 0:
        task_success_rate = (task_success_count / task_evaluated_count) * 100
        avg_task_score = total_task_score / task_evaluated_count
        
        print(f"\nTask Success Rate: {task_success_rate:.1f}% ({task_success_count}/{task_evaluated_count})")
        print(f"平均Task Score: {avg_task_score:.3f}")
    else:
        print("\n⚠️  没有可评估的样本")
        task_success_rate = 0.0
        avg_task_score = 0.0
    
    # 保存结果
    if output_path:
        output = {
            "summary": {
                "total_samples": len(test_states),
                "evaluated_samples": task_evaluated_count,
                "task_success_count": task_success_count,
                "task_success_rate": task_success_rate,
                "avg_task_score": avg_task_score,
            },
            "detailed_results": results,
        }
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 结果已保存到: {output_path}")
    
    return results, {
        "task_success_rate": task_success_rate,
        "avg_task_score": avg_task_score,
        "task_success_count": task_success_count,
        "task_evaluated_count": task_evaluated_count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估base模型单轮代码生成能力")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base模型路径")
    parser.add_argument("--test_states", type=str, required=True,
                       help="测试数据路径")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大样本数")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p采样")
    parser.add_argument("--use_original_query", action="store_true",
                       help="使用original_instruct_prompt而不是masked query")
    
    args = parser.parse_args()
    
    evaluate_base_model_single_turn(
        base_model=args.base_model,
        test_states_path=args.test_states,
        max_samples=args.max_samples,
        output_path=args.output,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_original_query=args.use_original_query,
    )
