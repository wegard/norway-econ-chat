# Norway Econ Chat: Strategy & Implementation Plan

This document outlines the strategy for building a specialized chatbot expert on the Norwegian economy, leveraging the `nanochat` architecture.

## 1. Executive Summary

**Goal:** Create a Large Language Model (LLM) capable of expert-level dialogue regarding the Norwegian economy, monetary policy, fiscal budget, and market trends.

**Approach:**
1.  **Data First:** Curate a high-quality, domain-specific dataset (The "Norwegian Economic Corpus").
2.  **Specialized Tokenization:** Retrain the tokenizer to efficiently handle Norwegian vocabulary and grammar.
3.  **Pre-training:** Train a `nanochat` model from scratch (or continue pre-training) on the domain corpus.
4.  **Supervised Fine-Tuning (SFT):** Instruction-tune the model using synthetic Q&A pairs derived from authoritative sources.

---

## 2. Data Strategy (The "Gold Mine")

The quality of the model will be strictly determined by the quality of the data. We need two distinct datasets.

### 2.1. Pre-training Corpus (Raw Text)
*Target Size: 1B - 10B tokens (depending on compute budget).*

This data teaches the model *knowledge* and *language structure*.

| Source Category | Specific Targets | Action |
| :--- | :--- | :--- |
| **Government Reports** | [Regjeringen.no](https://www.regjeringen.no): Statsbudsjettet (National Budget), NOU (Official Norwegian Reports), White papers (Meld. St.). | Scrape PDFs/HTML. Convert to Markdown. |
| **Central Bank** | [Norges Bank](https://www.norges-bank.no): Pengepolitisk rapport (Monetary Policy Report), Finansiell stabilitet, Speeches by the Governor. | Download PDF archive. OCR/Text Extraction. |
| **Statistics** | [SSB (Statistics Norway)](https://www.ssb.no): Analysis reports, "Økonomiske analyser", "Konjunkturtendensene". | API or bulk download reports. |
| **Legal/Regulatory** | [Lovdata](https://lovdata.no): Tax laws (Skatteloven), Financial markets acts. | Scrape relevant sections. |
| **News & Analysis** | E24, Dagens Næringsliv (DN), Finansavisen (Archives). *Note: Copyright restrictions apply.* | Partner or use headlines/summaries if full text is restricted. |
| **Academic** | NHH / BI Open Repositories (Master's theses, papers on Norwegian econ). | Filter for Norwegian language PDF papers. |

**Processing Pipeline:**
1.  **Ingestion:** Python scripts to scrape/download.
2.  **Extraction:** Use `marker` or similar tools to convert PDF -> Markdown.
3.  **Cleaning:** Remove headers, footers, page numbers, and "cookie consent" text.
4.  **Format:** Save as `.parquet` files with a single column `text`.
    *   *Why Parquet?* The `nanochat` loader expects this format.

### 2.2. SFT Corpus (Instruction Tuning)
*Target Size: 10k - 50k conversation pairs.*

This data teaches the model *how to chat* and answer questions.

*   **Synthetic Q&A:** Use a strong model (GPT-4o, Claude 3.5 Sonnet) to "read" chunks of the Pre-training Corpus and generate training examples.
    *   *Prompt:* "Here is a section from the National Budget 2025. Create 3 complex user questions about this text and provide expert answers based ONLY on the text."
*   **General Chat:** Translate a subset of `SmolTalk` or `OpenHermes` into Norwegian to ensure the model can handle greetings and general conversation logic.

**Format:**
Standard `nanochat` JSON structure:
```json
{
  "messages": [
    {"role": "user", "content": "Hva er handlingsregelen?"},
    {"role": "assistant", "content": "Handlingsregelen er en retningslinje for bruken av oljepenger..."}
  ]
}
```

---

## 3. Implementation Steps

### Step 1: Custom Tokenizer (`nanochat/tokenizer.py`)
The standard GPT-4 tokenizer is English-centric. A Norwegian tokenizer will be ~30-50% more efficient (fewer tokens per word).

1.  **Concatenate** a 1GB sample of your Norwegian corpus.
2.  **Train** a new BPE model using `scripts/tok_train.py` (requires minor adaptation to point to your data).
3.  **Output:** A new `tokenizer.model` optimized for Norwegian.

### Step 2: Dataset Integration
Create a new task definition file `tasks/norway_econ.py`:

```python
from tasks.common import Task
# Implement loading logic for your SFT .jsonl files
class NorwayEcon(Task):
    ...
```

Modify `nanochat/dataset.py` to point to your local `base_data` directory containing your `.parquet` files instead of downloading FineWeb.

### Step 3: Pre-training (`scripts/base_train.py`)
Run the training script on your new corpus.

*   **Config:**
    *   `vocab_size`: Set to your new tokenizer's size (e.g., 32000).
    *   `max_seq_len`: 1024 or 2048.
    *   `depth`: Start small (`d12`) to verify data quality. Scale to `d24` or `d36` if results are promising.

### Step 4: Supervised Fine-Tuning (`scripts/chat_sft.py`)
Modify the `TaskMixture` in `chat_sft.py` to include your new task:

```python
train_ds = TaskMixture([
    NorwayEcon(split="train"),
    SmolTalkNorwegian(split="train"), # Your translated general chat data
])
```

---

## 4. Evaluation Strategy (NorEconBench)

We cannot rely on standard US-centric benchmarks (MMLU, GSM8K).

1.  **Create a Benchmark:**
    *   **Facts:** 100 Multiple Choice questions on Norwegian tax rates, historical GDP data, etc.
    *   **Reasoning:** 100 Open-ended questions like "How would a rate hike by Norges Bank likely affect the housing market in Oslo vs. rural areas?"
2.  **LLM-as-a-Judge:** Use GPT-4o to grade the `nanochat` model's open-ended answers against a "Gold Standard" answer.
3.  **Integration:** Add a new class in `tasks/` for evaluation and hook it into `scripts/chat_eval.py`.

## 5. Future Proofing

*   **Scale Up:** If this prototype works, the dataset is the asset. You can swap the `nanochat` model for a larger Llama 3 or Mistral base and continue training on the same data.
*   **RAG (Retrieval Augmented Generation):** For a smaller model to be accurate on *specific numbers* (e.g., "What was the inflation in March 2023?"), integrating a RAG system later is recommended. The model handles the *reasoning*, the RAG provides the *facts*.
