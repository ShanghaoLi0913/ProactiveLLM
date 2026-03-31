"""
Supervised Fine-Tuning (SFT) training script for ProactiveLLM.

This script trains a model using supervised fine-tuning on trajectory data,
where the model learns to generate assistant responses given states.

Input:
  - Trajectories JSONL from data/logs/, each line:
      {
        "state": {...},
        "action": "Clarify" | "Execute",
        "assistant_msg": "...",
        "persona": {...}
      }

Output:
  - Trained model saved to output_dir
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import datasets as ds
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
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
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    _HAS_PEFT = False

# Import unified render_state function
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from policy.render_state import render_state


def load_trajectories(path: Path) -> List[Dict]:
    """Load trajectories JSONL."""
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def to_sft_format(trajectories: List[Dict], tokenizer) -> Dict[str, List[str]]:
    """Convert trajectories to SFT format (prompt, response pairs).
    
    For SFT, we use the chosen action's assistant message (from preference pairs)
    or all trajectories if no preference pairs are available.
    """
    dataset = {
        "prompt": [],
        "response": [],
    }
    
    for traj in trajectories:
        state = traj.get("state", {})
        persona = traj.get("persona", None)
        assistant_msg = traj.get("assistant_msg", "")
        
        if not assistant_msg:
            continue
        
        # Render state to text with persona
        state_text = render_state(state, persona=persona)
        
        # Use chat template for clear prompt/response boundary
        messages = [
            {"role": "user", "content": state_text},
            {"role": "assistant", "content": assistant_msg}
        ]
        
        # Format as full conversation for training
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Split into prompt and response
        # The chat template includes both user and assistant messages
        # We'll use the full formatted text as input, and the model learns to generate the assistant part
        dataset["prompt"].append(state_text)
        dataset["response"].append(assistant_msg)
    
    return dataset


def train(
    model_name: str,
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    lr: float = 5e-5,
    use_preference_pairs: bool = False,
):
    """
    Train model using SFT.
    
    Args:
        model_name: Base model name
        data_path: Path to trajectories JSONL (if use_preference_pairs=False) or preference pairs JSONL (if True)
        output_dir: Output directory for trained model
        epochs: Number of training epochs
        lr: Learning rate
        use_preference_pairs: If True, use only chosen actions from preference pairs; if False, use all trajectories
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Loading data from: {data_path}")
    if use_preference_pairs:
        # Load preference pairs and use chosen actions
        prefs = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    prefs.append(json.loads(line))
        
        # Convert preference pairs to trajectory-like format
        trajectories = []
        for pref in prefs:
            trajectories.append({
                "state": pref["state"],
                "persona": pref.get("persona", {}),
                "assistant_msg": pref["chosen_assistant_msg"],
                "action": pref["chosen_action"],
            })
        print(f"📊 Loaded {len(trajectories)} examples from preference pairs (using chosen actions)")
    else:
        # Load trajectories directly
        trajectories = load_trajectories(Path(data_path))
        print(f"📊 Loaded {len(trajectories)} trajectories")
    
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
    
    # Add action tokens as special tokens
    special_tokens = {"additional_special_tokens": ["Clarify", "Execute"]}
    tokenizer.add_special_tokens(special_tokens)
    print("✅ Added special tokens: Clarify, Execute")
    
    # Convert to SFT format
    sft_data = to_sft_format(trajectories, tokenizer)
    print(f"📊 Using {len(sft_data['prompt'])} examples for SFT training")
    
    # Model loading
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
                trust_remote_code=False,
                local_files_only=True,  # 强制使用本地缓存
            )
        except Exception as e:
            print(f"⚠️  Failed to use quantization: {e}")
            print("⚠️  Loading full precision model")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                low_cpu_mem_usage=True,
                token=hf_token,
                trust_remote_code=False,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
            low_cpu_mem_usage=True,
            token=hf_token,
            trust_remote_code=False,
        )
    
    # Resize embeddings after adding special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA if available
    if _HAS_PEFT:
        try:
            if _HAS_BNB and has_cuda and prepare_model_for_kbit_training is not None:
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
        except Exception as e:
            print(f"⚠️  Failed to apply LoRA: {e}")
            print("⚠️  Continuing without LoRA (may use more GPU memory)")
    else:
        print("⚠️  peft not available, training full model (may use more GPU memory)")
    
    # Prepare dataset with proper formatting
    def format_prompt_response(examples):
        """Format prompt and response for training."""
        prompts = examples["prompt"]
        responses = examples["response"]
        
        formatted_texts = []
        for prompt, response in zip(prompts, responses):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            formatted_texts.append(formatted)
        
        return {"text": formatted_texts}
    
    # Build dataset
    dataset_dict = ds.Dataset.from_dict(sft_data)
    dataset = dataset_dict.map(format_prompt_response, batched=True, remove_columns=["prompt", "response"])
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding=False,
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=str(output_dir_path),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="no",
        save_total_limit=1,
        bf16=use_bf16,
        fp16=use_fp16 and not use_bf16,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        optim="adamw_torch",
    )
    
    print("🚀 Starting SFT training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
    )
    
    trainer.train()
    print("✅ Training finished, saving model...")
    trainer.save_model(str(output_dir_path))
    tokenizer.save_pretrained(str(output_dir_path))
    print("✅ Model and tokenizer saved.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SFT policy model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to trajectories JSONL or preference pairs JSONL",
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
        help="Output directory for trained model",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--use_preference_pairs",
        action="store_true",
        help="If set, use preference pairs JSONL and train on chosen actions only",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        use_preference_pairs=args.use_preference_pairs,
    )
