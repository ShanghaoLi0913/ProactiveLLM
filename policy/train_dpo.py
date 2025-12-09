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


def to_dpo_format(records):
    """Convert preference JSONL to TRL DPO format.
    
    Sequential Decision Process:
    - Policy model learns to predict action tokens (Clarify/Execute) at each turn
    - Code generation is handled separately (not affected by DPO)
    - This prevents DPO from polluting code generation capabilities
    
    IMPORTANT: Uses render_state() which contains ONLY pure state information,
    NO action_prompts or templates. This allows the model to freely decide
    which action to take based on state (task_uncertainty, dialogue_turn, prev_reject).
    """

    dataset = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    for ex in records:
        dataset["prompt"].append(render_state(ex["state"]))
        # Use action tokens, not full messages
        dataset["chosen"].append(ex["chosen_action"])  # action token: Clarify/Execute
        dataset["rejected"].append(ex["rejected_action"])  # action token
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
    dpo_data = to_dpo_format(records)
    print(f"📊 Using {len(dpo_data['prompt'])} examples for training (sequential decision: Clarify/Execute)")

    print(f"🔡 Loading tokenizer: {model_name}")
    hf_token = os.environ.get("HF_TOKEN")
    cache_dir = os.environ.get("HF_HOME")
    # Try to find local snapshot directory first
    snapshot_dir = None
    if cache_dir:
        import glob
        pattern = f"{cache_dir}/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*"
        snapshots = glob.glob(pattern)
        if snapshots:
            snapshot_dir = snapshots[0]
            print(f"📁 Found local snapshot: {snapshot_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_dir if snapshot_dir else model_name,
        use_fast=True,
        token=hf_token if not snapshot_dir else None,
        cache_dir=cache_dir if not snapshot_dir else None,
        local_files_only=snapshot_dir is not None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add action tokens as special tokens (Sequential Decision: Clarify/Execute)
    special_tokens = {"additional_special_tokens": ["Clarify", "Execute"]}
    tokenizer.add_special_tokens(special_tokens)
    print("✅ Added special tokens: Clarify, Execute")

    # Model loading with optional 4-bit quantization (QLoRA)
    use_qlora = False
    print("🔄 Loading model...")
    if _HAS_BNB:
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("✅ Using 4-bit quantization (QLoRA)")
            model = AutoModelForCausalLM.from_pretrained(
                snapshot_dir if snapshot_dir else model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                token=hf_token if not snapshot_dir else None,
                cache_dir=cache_dir if not snapshot_dir else None,
                local_files_only=snapshot_dir is not None,
            )
            use_qlora = True
        except Exception as e:  # pragma: no cover - runtime safeguard
            print(f"⚠️  QLoRA load failed ({e}), falling back to FP16")
            model = AutoModelForCausalLM.from_pretrained(
                snapshot_dir if snapshot_dir else model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                token=hf_token if not snapshot_dir else None,
                cache_dir=cache_dir if not snapshot_dir else None,
                local_files_only=snapshot_dir is not None,
            )
    else:
        print("⚠️  bitsandbytes not available, using FP16")
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_dir if snapshot_dir else model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=hf_token if not snapshot_dir else None,
            cache_dir=cache_dir if not snapshot_dir else None,
            local_files_only=snapshot_dir is not None,
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
        bf16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        reference_free=True,
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
