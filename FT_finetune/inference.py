import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# === Config ===
model_path = "qwen_unsup_fulltext_model_final"
input_file = "full_text_prompts_with_ids.jsonl"
output_file = "full_text_predictions.jsonl"
max_input_tokens = 12000
max_full_text_tokens = 11000
max_new_tokens = 128
num_samples = 1150

# === Load Tokenizer & Model ===
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"  # Truncate from beginning to keep question/answer

# === Inference ===
results = []
empty_outputs = 0
missing_questions = 0

with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f, total=num_samples, desc="Running inference")):
        if i >= num_samples:
            break

        entry = json.loads(line)
        paper_id = entry["id"]
        full_text = entry["prompt"].strip()
        question = entry["question"].strip()
        expected_output = entry["expected_output"].strip()

        # Step 1: Trim full_text to 3800 tokens
        trimmed = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_full_text_tokens
        )
        cropped_full_text = tokenizer.decode(trimmed["input_ids"][0], skip_special_tokens=True)

        # Step 2: Build prompt with answer anchor
        prompt = (
            "You are an AI assistant that answers questions about research papers. Do not answer additional questions.\n\n"
            "Do not refer to figures, tables in your answer, even if mentioned in the text.\n\n"
            f"{cropped_full_text}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # Step 3: Tokenize full prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
            padding=True
        ).to("cuda")

        # Step 4: Check if question was truncated
        decoded_input = tokenizer.decode(inputs["input_ids"][0])
        if "question:" not in decoded_input.lower():
            missing_questions += 1

        # Step 5: Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = decoded_output[len(prompt):].strip()

        if not generated_answer:
            empty_outputs += 1

        results.append({
            "id": paper_id,
            "prompt": cropped_full_text,
            "question": question,
            "expected_output": expected_output,
            "generated_answer": generated_answer
        })

# === Save Predictions ===
with open(output_file, "w", encoding="utf-8") as out:
    for r in results:
        out.write(json.dumps(r) + "\n")

# === Stats ===
print(f"\n✅ Done! Saved {len(results)} test predictions to {output_file}")
print(f"⚠️ Empty outputs: {empty_outputs}")
print(f"⚠️ Prompts where question was likely truncated: {missing_questions}")
