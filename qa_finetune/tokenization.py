from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import json

# Load tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Function to load jsonl
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

# Tokenization function
def tokenize_fn(example):
    prompt = f"<|user|>\n{example['input']}\n<|assistant|>\n{example['output']}"
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Load and convert jsonl to HF dataset
train_data = Dataset.from_list(load_jsonl("processed/train_instruction.jsonl"))
val_data = Dataset.from_list(load_jsonl("processed/validation_instruction.jsonl"))

# Tokenize
train_tokenized = train_data.map(tokenize_fn)
val_tokenized = val_data.map(tokenize_fn)

# Save to disk for reuse
train_tokenized.save_to_disk("tokenized/train")
val_tokenized.save_to_disk("tokenized/validation")

print("✅ Tokenized and saved train + validation datasets.")
