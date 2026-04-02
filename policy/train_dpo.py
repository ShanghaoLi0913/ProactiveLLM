import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
import datasets as ds
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None  # type: ignore
    _HAS_BNB = False

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    _HAS_PEFT = True
except Exception:
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    prepare_model_for_kbit_training = None  # type: ignore
    _HAS_PEFT = False


def load_prefs(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Import unified render_state function
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from policy.render_state import render_state


def to_dpo_format(records, tokenizer):
    """Convert preference JSONL to TRL DPO format.
    
    Sequential Decision Process:
    - Policy model learns to predict action tokens (Clarify/Execute) at each turn
    - Code generation is handled separately (not affected by DPO)
    - This prevents DPO from polluting code generation capabilities
    
    IMPORTANT: Uses render_state() which contains ONLY pure state information,
    NO action_prompts or templates. This allows the model to freely decide
    which action to take based on state (task_uncertainty, dialogue_turn, prev_reject).
    
    PERSONA-AWARE: Each state includes persona information, allowing the model
    to learn different proactivity levels for different user types.
    
    V15 FIX: Uses chat template to provide clear prompt/response boundary.
    This ensures the model knows where to start generating the response.
    """

    dataset = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    for ex in records:
        # ⭐ Pass persona as separate parameter for explicit persona-awareness
        state = ex["state"]
        persona = ex.get("persona", None)
        
        # Render state to text with persona
        state_text = render_state(state, persona=persona)
        
        # ✅ V15 Fix: Use chat template for clear prompt/response boundary
        messages = [{"role": "user", "content": state_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Adds <|start_header_id|>assistant<|end_header_id|>
        )
        
        dataset["prompt"].append(prompt)
        # Full response (code/question)
        dataset["chosen"].append(ex["chosen_assistant_msg"])
        dataset["rejected"].append(ex["rejected_assistant_msg"])
    return dataset


def train(
    model_name: str,
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    lr: float = 5e-5,
    beta: float = 0.1,
):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading preference pairs from: {data_path}")
    records = load_prefs(Path(data_path))
    print(f"📊 Loaded {len(records)} preference pairs")

    print(f"🔡 Loading tokenizer: {model_name}")
    hf_token = os.environ.get("HF_TOKEN")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        token=hf_token,
        trust_remote_code=False,
        local_files_only=True,  # 强制使用本地缓存
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add action tokens as special tokens (Sequential Decision: Clarify/Execute)
    special_tokens = {"additional_special_tokens": ["Clarify", "Execute"]}
    tokenizer.add_special_tokens(special_tokens)
    print("✅ Added special tokens: Clarify, Execute")
    
    # ✅ V15 Fix: Convert to DPO format using chat template
    dpo_data = to_dpo_format(records, tokenizer)
    print(f"📊 Using {len(dpo_data['prompt'])} examples for training (sequential decision: Clarify/Execute)")

    # Model loading with optional 4-bit quantization (QLoRA)
    use_qlora = False
    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16
    device_map = "auto" if has_cuda else "cpu"
    print(f"🔄 Loading model... (cuda={has_cuda}, bf16={use_bf16}, fp16={use_fp16})")
    if _HAS_BNB and has_cuda:
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("✅ Using 4-bit quantization (QLoRA)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                low_cpu_mem_usage=True,
                token=hf_token,
                local_files_only=True,  # 强制使用本地缓存
            )
            use_qlora = True
        except Exception as e:  # pragma: no cover - runtime safeguard
            print(f"⚠️  QLoRA load failed ({e}), falling back to FP16")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
                device_map=device_map,
                low_cpu_mem_usage=True,
                token=hf_token,
                local_files_only=True,  # 强制使用本地缓存
            )
    else:
        print("⚠️  bitsandbytes not available, using FP16/FP32")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
            token=hf_token,
            local_files_only=True,  # 强制使用本地缓存
        )

    # Resize embeddings after adding special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA if available
    if _HAS_PEFT:
        try:
            if use_qlora and prepare_model_for_kbit_training is not None:
                model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            print("✅ Applied LoRA (r=64, alpha=16)")
            model.print_trainable_parameters()
        except Exception as e:  # pragma: no cover - runtime safeguard
            print(f"⚠️  Failed to apply LoRA: {e}")
            print("⚠️  Continuing without LoRA (may use more GPU memory)")
    else:
        print("⚠️  peft not available, training full model (may use more GPU memory)")

    # Build Dataset
    dataset = ds.Dataset.from_dict(dpo_data)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    # Training configuration (4090-friendly)
    training_args = DPOConfig(
        output_dir=str(output_dir_path),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        remove_unused_columns=False,
        save_strategy="no",  # save final model only via save_model
        save_total_limit=1,
        beta=beta,
        max_length=2048,
        gradient_checkpointing=True,
        bf16=use_bf16,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        reference_free=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
    )

    print("🚀 Starting DPO training...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )

    trainer.train()
    print("✅ Training finished, saving model...")
    trainer.save_model(str(output_dir_path))
    tokenizer.save_pretrained(str(output_dir_path))
    print("✅ Model and tokenizer saved.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPO policy model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to preference pairs JSONL (e.g., data/dpo/prefs_150.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model name (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save trained adapter/tokenizer",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
    )
