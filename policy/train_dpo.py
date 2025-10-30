import json
from pathlib import Path
from typing import Dict

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig


def load_prefs(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def to_dpo_format(records):
    # state_text as instruction; actions as responses
    def render_state(st: Dict) -> str:
        return (
            f"domain={st['domain']} | turn={st['dialogue_turn']} | "
            f"clarity={st['query_clarity']} | prev_reject={st['prev_reject']} |\n"
            f"user_query: {st['query']}"
        )

    dataset = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    for ex in records:
        dataset["prompt"].append(render_state(ex["state"]))
        dataset["chosen"].append(ex["chosen_action"])  # action token: LOW/MID/HIGH
        dataset["rejected"].append(ex["rejected_action"])  # action token
    return dataset


def train(model_name: str, data_path: str, output_dir: str):
    records = load_prefs(Path(data_path))
    dpo_data = to_dpo_format(records)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Simple token mapping: ensure LOW/MID/HIGH are in vocab (add as special tokens if needed)
    special_tokens = {"additional_special_tokens": ["LOW", "MID", "HIGH"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Wrap into datasets.Dataset
    import datasets as ds

    dataset = ds.Dataset.from_dict(dpo_data)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=1,
            learning_rate=5e-5,
            logging_steps=10,
            remove_unused_columns=False,
        ),
        beta=0.1,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        max_length=256,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    # Example usage
    train(
        model_name="gpt2",  # replace with Llama/Mistral locally if available
        data_path=str(Path(__file__).resolve().parent.parent / "data/prefs_mvp.jsonl"),
        output_dir=str(Path(__file__).resolve().parent.parent / "outputs/policy"),
    )


