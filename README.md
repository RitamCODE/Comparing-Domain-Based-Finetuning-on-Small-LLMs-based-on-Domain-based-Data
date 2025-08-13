# Comparing Small LLMs for Scientific Question Answering  
**Fine-tuning Qwen2.5-0.5B on Question-Answer Pairs vs. Full-Text Context**  

## 📌 Overview  
This project investigates how two different fine-tuning strategies affect the question-answering capabilities of small language models (LLMs) in the context of scientific research papers. We use **Qwen2.5-0.5B**, a compact open-source LLM, and **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.  

We compare:  
1. **QA Pair Fine-Tuning** – Model trained on abstract + gold answer pairs.  
2. **Full-Text Fine-Tuning** – Model trained with the entire research paper as context.  

Evaluation is performed using **DeepSeek-R1**, a larger reasoning model, as an external, blind judge.  

---

## 🗂 Dataset  
We use the **[QASPER](https://huggingface.co/datasets/allenai/qasper)** dataset — a high-quality QA dataset focused on NLP research papers.  
- **QA Pairs:** Abstract + question → gold answer (+ evidence if available)  
- **Full-Text:** Entire paper content merged into a single text block  

**Data preparation highlights:**  
- Removed unanswerable questions and poor annotations.  
- Standardized formatting for instruction-style fine-tuning.  
- Token length limits set to 1024 (QA) and up to 10,000 (Full-Text).  

---

## ⚙️ Methodology  

### 1. Model & Tokenization  
- **Base model:** [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)  
- **Tokenization:** Hugging Face tokenizer  
- Sequence length:  
  - QA: 1024 tokens  
  - Full-text: up to 10,000 tokens with left truncation  

### 2. Fine-Tuning with LoRA  
- LoRA rank: `8`  
- LoRA alpha: `16`  
- Dropout: `0.1`  
- Only attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) are trainable.  

**Training setup:**  
- Batch size: 1 (grad. accumulation = 8 → effective batch = 8)  
- Epochs: 3  
- Mixed precision (fp16) enabled  
- PEFT & Hugging Face Trainer used  

### 3. Inference  
- **QA Model:** Greedy decoding, max new tokens = 128  
- **Full-Text Model:** Prompt engineering to prevent figure/table references  
- Left-side truncation to preserve latest context  

---

## 📊 Evaluation  
**Judge:** [DeepSeek-R1](https://huggingface.co/deepseek-ai)  
- Blind comparison between models  
- Metrics: **Relevance, Correctness, Completeness** (1–5 scale)  
- Outputs stored in JSONL with mapping for analysis  

**Results:**  
- **Full-text model:** 65.5% wins  
- **QA model:** 12.2% wins  
- **Ties:** 22.3%  

---

## 🔍 Key Findings  
- Full-text fine-tuning consistently outperforms QA-only fine-tuning.  
- Input coverage plays a major role in small LLM performance.  
- Even a small model like Qwen2.5-0.5B can deliver strong results with LoRA.  

---

## 📌 Limitations  
- Limited evaluation size (~150 examples) due to API constraints.  
- Gold answers sometimes incomplete.  
- LLM-based evaluation may have biases.  

---

## 🛠 Installation & Usage  

```bash
# Clone repo
git clone https://github.com/your-username/small-llm-qa-comparison.git
cd small-llm-qa-comparison

# Install dependencies
pip install -r requirements.txt

# Fine-tune QA model
python train_qa.py

# Fine-tune full-text model
python train_fulltext.py

# Run inference
python inference.py
```

## 📚 References  

1. [QASPER Dataset](https://huggingface.co/datasets/allenai/qasper)  
2. [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)  
3. [LoRA: Efficient Fine-tuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)  


