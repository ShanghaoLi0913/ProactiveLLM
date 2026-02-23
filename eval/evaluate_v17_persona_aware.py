"""
Evaluate V17 model's persona-aware behavior on test set.
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from collections import defaultdict, Counter
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.render_state import render_state


def load_model(base_model_path, adapter_path):
    """Load base model and adapter."""
    print(f"🔄 Loading base model: {base_model_path}")
    
    # Load tokenizer from adapter (has special tokens)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    
    # Get HF token from environment
    import os
    hf_token = os.getenv("HF_TOKEN")
    
    # Load base model with 8-bit quantization to save memory
    # If path looks like a local snapshot, use local_files_only
    use_local = base_model_path.startswith("/") and "snapshots" in base_model_path
    
    # Try 8-bit quantization first to save memory
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        print("🔄 Loading base model with 8-bit quantization (to save memory)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=hf_token if hf_token else None,
            local_files_only=use_local,
        )
    except Exception as e:
        print(f"⚠️  8-bit quantization failed ({e}), falling back to bfloat16")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=hf_token if hf_token else None,
            local_files_only=use_local,
        )
    
    # Check if we need to resize embeddings
    base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    
    if base_vocab_size != tokenizer_vocab_size:
        print(f"⚠️  Vocab size mismatch: base={base_vocab_size}, tokenizer={tokenizer_vocab_size}", flush=True)
        print(f"🔄 Resizing embeddings to match tokenizer...", flush=True)
        # Clear GPU cache before resize
        torch.cuda.empty_cache()
        # Resize embeddings (this requires extra memory temporarily)
        base_model.resize_token_embeddings(tokenizer_vocab_size)
        torch.cuda.empty_cache()
        print(f"✅ Embeddings resized to {tokenizer_vocab_size}", flush=True)
    else:
        print(f"✅ Vocab sizes match: {base_vocab_size}", flush=True)
    
    print(f"🔄 Loading adapter: {adapter_path}", flush=True)
    sys.stdout.flush()
    # Load adapter weights to CPU first to save GPU memory
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path, device_map="auto")
    except Exception as e:
        print(f"⚠️  Failed to load adapter with device_map: {e}")
        print(f"🔄 Trying to load adapter directly...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print(f"✅ Model loaded")
    return model, tokenizer


def extract_action_from_response(response):
    """Extract action from model response."""
    response_lower = response.lower()
    
    # Check for explicit action markers
    if response.startswith("Clarify\n") or response.startswith("clarify\n"):
        return "Clarify"
    if response.startswith("Execute\n") or response.startswith("execute\n"):
        return "Execute"
    
    # Check for question marks (indicates Clarify)
    if "?" in response[:200]:
        return "Clarify"
    
    # Check for code (indicates Execute)
    if "```" in response or "def " in response or "class " in response:
        return "Execute"
    
    # Default: if contains clarifying words
    clarify_keywords = ["could you", "can you", "please clarify", "what", "which", "how"]
    if any(keyword in response_lower[:100] for keyword in clarify_keywords):
        return "Clarify"
    
    return "Execute"  # Default


def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate model response."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_length = inputs['input_ids'].shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Extract only the generated part (skip the input tokens)
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def evaluate_on_test_set(model, tokenizer, test_states_file):
    """Evaluate model on test states with different personas."""
    
    # Load test states
    test_states = []
    with open(test_states_file) as f:
        for line in f:
            test_states.append(json.loads(line))
    
    print(f"📊 Loaded {len(test_states)} test states")
    
    # Define personas
    personas = [
        {"name": "Busy-Developer", "patience": "low", "expertise": "mid"},
        {"name": "Experienced-Engineer", "patience": "mid", "expertise": "high"},
        {"name": "Novice-Learner", "patience": "high", "expertise": "low"}
    ]
    
    results = []
    stats_by_persona = defaultdict(lambda: {"actions": [], "uncertainties": []})
    
    for state in test_states:
        for persona in personas:
            # Render state with persona
            state_text = render_state(state, persona=persona)
            
            # Create prompt using chat template
            messages = [{"role": "user", "content": state_text}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate response
            response = generate_response(model, tokenizer, prompt)
            
            # Extract action
            action = extract_action_from_response(response)
            
            # Store result
            result = {
                "state_id": state["id"],
                "persona": persona["name"],
                "uncertainty": state.get("task_uncertainty", 0.5),
                "predicted_action": action,
                "response": response[:200],  # First 200 chars
            }
            results.append(result)
            
            stats_by_persona[persona["name"]]["actions"].append(action)
            stats_by_persona[persona["name"]]["uncertainties"].append(state.get("task_uncertainty", 0.5))
    
    return results, stats_by_persona


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to adapter checkpoint")
    parser.add_argument("--base_model", required=True, help="Path to base model")
    parser.add_argument("--test_states", required=True, help="Path to test states file")
    parser.add_argument("--output", default="eval_results/v17_persona_aware.json", help="Output file")
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.base_model, args.adapter)
    
    # Evaluate
    print("\n🔍 Evaluating on test set...")
    results, stats = evaluate_on_test_set(model, tokenizer, args.test_states)
    
    # Print statistics
    print("\n" + "="*80)
    print("📊 Evaluation Results")
    print("="*80)
    
    for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
        actions = stats[persona]["actions"]
        action_counts = Counter(actions)
        
        print(f"\n👤 {persona}:")
        print(f"   Total predictions: {len(actions)}")
        for action, count in action_counts.most_common():
            pct = count / len(actions) * 100
            print(f"   {action}: {count} ({pct:.1f}%)")
    
    # Check persona differentiation
    print("\n" + "="*80)
    print("🎯 Persona Differentiation Check")
    print("="*80)
    
    clarify_rates = {}
    for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
        actions = stats[persona]["actions"]
        clarify_rate = actions.count("Clarify") / len(actions) * 100
        clarify_rates[persona] = clarify_rate
        print(f"{persona}: {clarify_rate:.1f}% Clarify")
    
    # Check if there's differentiation
    if clarify_rates["Novice-Learner"] > clarify_rates["Busy-Developer"]:
        print("\n✅ Persona differentiation detected!")
        print(f"   Novice asks more questions than Busy: "
              f"{clarify_rates['Novice-Learner']:.1f}% vs {clarify_rates['Busy-Developer']:.1f}%")
    else:
        print("\n⚠️  No clear persona differentiation")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "results": results,
            "statistics": {
                persona: {
                    "total": len(stats[persona]["actions"]),
                    "clarify_count": stats[persona]["actions"].count("Clarify"),
                    "execute_count": stats[persona]["actions"].count("Execute"),
                    "clarify_rate": stats[persona]["actions"].count("Clarify") / len(stats[persona]["actions"]),
                }
                for persona in stats
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
