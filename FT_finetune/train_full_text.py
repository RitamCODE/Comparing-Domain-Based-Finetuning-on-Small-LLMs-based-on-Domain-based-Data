from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import json

# removing unwanted warnings to clear the terminal for better debugging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# === Step 1: Load and prepare dataset ===
# Your JSONL file should have: title, abstract, full_text
INPUT_PATH = "processed/unsupervised_structured_train.jsonl"

# Flatten title + abstract + full_text into a single text string per sample
def format_for_lm(example):
    text = f"### Title:\n{example['title']}\n\n### Abstract:\n{example['abstract']}\n\n{example['full_text']}"
    return {"text": text}

# Load dataset from JSONL
dataset = load_dataset("json", data_files=INPUT_PATH, split="train")
dataset = dataset.map(format_for_lm)

# === Step 2: Load Qwen model & tokenizer ===
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Required for Trainer

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

# === Step 3: Tokenize the dataset ===
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# === Step 4: Setup Trainer ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="qwen_unsup_fulltext_output",
    # evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="logs",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Step 5: Train ===
trainer.train()
trainer.save_model("qwen_unsup_fulltext_model_final")  # Saves model + config
tokenizer.save_pretrained("qwen_unsup_fulltext_model_final")
