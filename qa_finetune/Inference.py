import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm

# Load model + tokenizer
model_path = "qwen_qasper_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()
tokenizer.pad_token = tokenizer.eos_token

# Load test inputs
test_inputs = []
with open("processed/test_instruction.jsonl", "r") as f:
    for line in f:
        obj = json.loads(line)
        test_inputs.append(obj["input"])

# Generate predictions
results = []
model.eval()
with torch.no_grad():
    for prompt in tqdm(test_inputs, desc="Generating"):
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        input_ids = encoded["input_ids"].cuda()
        attention_mask = encoded["attention_mask"].cuda()
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Strip prompt to get only generated answer
        generated_answer = output_text.replace(prompt, "").strip()
        results.append({
            "input": prompt,
            "prediction": generated_answer
        })

# Save predictions
with open("qwen_test_predictions.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"✅ Saved {len(results)} predictions to qwen_test_predictions.jsonl")
