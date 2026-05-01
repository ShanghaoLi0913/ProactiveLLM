"""SFT warmup with proper prompt masking (v33).

Fix vs existing train_sft.py: only compute next-token loss on RESPONSE tokens,
not prompt tokens. This concentrates the gradient signal on the chosen
response (including "Clarify\\n" / "Execute\\n" prefix), making 1 epoch on 500
pairs effective.

Usage:
    KEEP_PREFIX=1 python policy/train_sft_v33.py \
        --data data/dpo/prefs_v29_100states.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output models/v33_sft_v2 \
        --epochs 1 --lr 1e-5
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import datasets as ds
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
)

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from policy.render_state import render_state  # noqa: E402


def load_prefs(path: Path) -> List[Dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def collate_fn_factory(tokenizer, max_length: int = 2048):
    """Collate fn with prompt masking: labels = -100 for prompt tokens, real for response."""
    def collate_fn(batch):
        input_ids_list = []
        labels_list = []
        attention_masks = []

        for ex in batch:
            prompt = ex["prompt"]
            response = ex["response"]

            # Tokenize prompt alone (with bos/special if needed)
            prompt_enc = tokenizer(prompt, add_special_tokens=False)
            response_enc = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)

            input_ids = prompt_enc["input_ids"] + response_enc["input_ids"]
            labels = [-100] * len(prompt_enc["input_ids"]) + list(response_enc["input_ids"])

            # Truncate
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        # Pad
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = tokenizer.pad_token_id

        padded_ids = []
        padded_labels = []
        attn = []
        for ids, lbl in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(lbl + [-100] * pad_len)
            attn.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    return collate_fn


def build_dataset(records: List[Dict], tokenizer) -> ds.Dataset:
    """Build dataset of {prompt, response} pairs.

    KEEP_PREFIX=1 keeps "Clarify\\n" / "Execute\\n" in response targets.
    """
    keep_prefix = os.environ.get("KEEP_PREFIX") == "1"

    rows = []
    for ex in records:
        state = ex.get("state", {})
        persona = ex.get("persona")
        state_text = render_state(state, persona=persona)

        # User-message wrapped via chat template, with assistant header for generation
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": state_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = state_text

        response = ex["chosen_assistant_msg"]
        if not keep_prefix:
            for pfx in ("Clarify\n", "Execute\n"):
                if response.startswith(pfx):
                    response = response[len(pfx):]
                    break

        rows.append({"prompt": prompt, "response": response})
    return ds.Dataset.from_list(rows)


def train(
    model_name: str,
    data_path: str,
    output_dir: str,
    epochs: int = 1,
    lr: float = 1e-5,
):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading prefs from: {data_path}")
    records = load_prefs(Path(data_path))
    print(f"📊 Loaded {len(records)} records")

    print(f"🔡 Loading tokenizer: {model_name}")
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, token=hf_token,
        trust_remote_code=False, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(records, tokenizer)
    print(f"📊 Built {len(dataset)} examples (keep_prefix={os.environ.get('KEEP_PREFIX', '0')})")

    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    device_map = "auto" if has_cuda else "cpu"

    print(f"🔄 Loading model... (cuda={has_cuda}, bf16={use_bf16})")
    quantization_config = None
    if _HAS_BNB and has_cuda:
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
        token=hf_token,
        local_files_only=True,
    )

    if _HAS_PEFT:
        if quantization_config is not None and prepare_model_for_kbit_training is not None:
            model = prepare_model_for_kbit_training(model)

        _alpha = int(os.environ.get("LORA_ALPHA", "16"))
        _r = int(os.environ.get("LORA_R", "64"))
        lora_config = LoraConfig(
            r=_r, lora_alpha=_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print(f"✅ Applied LoRA (r={_r}, alpha={_alpha})")
        model.print_trainable_parameters()

    split = dataset.train_test_split(test_size=0.1, seed=42)
    collate_fn = collate_fn_factory(tokenizer)

    args = TrainingArguments(
        output_dir=str(output_dir_path),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="epoch",
        bf16=use_bf16,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=collate_fn,
    )

    print("🚀 Starting v33 SFT training (with prompt masking)...")
    trainer.train()
    print("✅ Training finished, saving adapter...")
    trainer.save_model(str(output_dir_path))
    tokenizer.save_pretrained(str(output_dir_path))
    print(f"✅ SFT adapter + tokenizer saved to {output_dir_path}")


def parse_args():
    p = argparse.ArgumentParser(description="v33 SFT with prompt masking")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
    )
