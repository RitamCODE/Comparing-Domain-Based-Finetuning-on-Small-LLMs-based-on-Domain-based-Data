from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_from_disk

# Load tokenizer and model
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    # load_in_4bit=True,
    device_map="auto"
)

# LoRA setup
# model = prepare_model_for_kbit_training(model)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


# Load tokenized dataset
tokenized_dataset = load_from_disk("tokenized/train")

# Training setup
training_args = TrainingArguments(
    output_dir="qwen_qasper_lora_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="epoch",
    report_to="none",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# Save model
trainer.save_model("qwen_qasper_finetuned")
tokenizer.save_pretrained("qwen_qasper_finetuned")
