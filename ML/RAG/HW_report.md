# Experimental Report: Machine Learning QA System (RAG vs. No RAG)

## 1. Describe Your Dataset

### 1.1 Primary Corpus: Machine Learning Lecture Slides

The core knowledge base was constructed from course-specific lecture slides (`.pptx`) covering topics such as Perceptrons, GANs, VAEs, Transformers, and Reinforcement Learning.

- **Preprocessing:** Text was extracted using `python-pptx`, concatenated by slide, and stored as a JSONL file (`ml_lectures_corpus.jsonl`).
- **Indexing:** Document chunks were embedded using `all-MiniLM-L6-v2` and indexed via FAISS (`IndexFlatIP`) for cosine similarity search.

- **ML QA Dataset Example:**

```json
{"text": "Introduction of Transformer"}
{"text": "2\nEncoder-Decoder model\nSeq2Seq"}
{"text": "3\nK,\nV\nC\ntokens: 輸入的原始資料經過一個網路，產生embedding features, 為tokens\nC\nK\nV\nToken from encoder\nQ\nToken from\ndecoder\nQ\nTransformer"}
{"text": "4\nSelf-attention\nvectors\nscalar\ntokens\nInput embedding\nx1\nx2\nx3\nx4"}
{"text": "Self-attention\n5"}
```

### 1.2 Evaluation Dataset (QA Pairs)

- **ML QA Dataset:** 10 fundamental questions (e.g., "What is overfitting?", "Explain L2 regularization") paired with "textbook-style" reference answers derived from the lectures.
- **Bonus Dataset (Emoji/In-Joke):** A custom `emoji_corpus.jsonl` defining private, project-specific semantics for emojis (e.g., defining 🥹 as "begging for leniency after criticism"). This dataset was created to test the system's ability to handle non-public knowledge.

---

## 2. Result Without RAG (Baseline)

### Methodology

The baseline system used **`google/gemma-3-1b-it`** directly. The system prompt requested concise definitions without access to external documents.

### Quantitative Results (ML Dataset)

| Metric             | Score  |
| :----------------- | :----- |
| **BLEU**           | 0.0278 |
| **ROUGE-L**        | 0.1914 |
| **BERTScore (F1)** | 0.8483 |

### Observation

The model demonstrated decent general knowledge (High BERTScore) but failed to match the specific phrasing and definition style of the course material (Low BLEU), often providing generic internet-style explanations.

---

## 3. Result With RAG

### Methodology

The RAG pipeline retrieved the top $k=3$ relevant slide chunks based on the user query. These chunks were injected into the prompt context before generation.

### Quantitative Results (ML Dataset)

| Metric             | Score      | Improvement |
| :----------------- | :--------- | :---------- |
| **BLEU**           | **0.1104** | +297%       |
| **ROUGE-L**        | **0.3255** | +70%        |
| **BERTScore (F1)** | **0.8882** | +0.04       |

### Observation

RAG significantly improved the "style matching." The model successfully adopted the lecture's specific terminology and sentence structure, evidenced by the 4x increase in BLEU score.

---

## 4. Compare the Result (w/o RAG)

Comparing the baseline and RAG results reveals two distinct benefits:

1.  **Style Alignment (Lexical):** The most visible improvement is in BLEU/ROUGE. The Baseline model uses generic synonyms, whereas the RAG model uses the exact keywords found in the slides (e.g., specific variable names or formal definitions).
2.  **Definition Precision (Semantic):** The increase in BERTScore (0.84 to 0.88) indicates that RAG helps the model narrow down broad concepts (like "Bias-Variance Tradeoff") to the specific context taught in this class, removing ambiguity.

---

## 5. Bonus: Different Model Comparison

To stress-test the RAG architecture, we compared two different backends and models on the **Custom Emoji Semantic Task**. This task requires the model to identify that the emoji 🥹 represents _"pretending to be obedient to beg for leniency after criticism"_—a rule that does not exist in public pre-training data.

### Setup

- **Model A:** `google/gemma-3-1b-it` (Hugging Face pipeline)
- **Model B:** `llama3.2` (Ollama backend)
- **Metric:** Standard text metrics + **Emoji Hit Rate** (Did it output 🥹?)

### Comparative Results

| Backend Model  | RAG Status   | BLEU       | ROUGE-L    | BERTScore  | Emoji Accuracy (🥹)      |
| :------------- | :----------- | :--------- | :--------- | :--------- | :---------------------- |
| **Gemma-3-1B** | No RAG       | 0.0000     | 0.1667     | 0.8418     | **Fail** (Suggested 🤔) |
| **Gemma-3-1B** | **With RAG** | **0.3095** | **0.3095** | 0.8907     | **Pass** (Found 🥹)      |
| **Llama 3.2**  | No RAG       | 0.0234     | 0.1333     | 0.8569     | **Fail** (Suggested 🙏) |
| **Llama 3.2**  | **With RAG** | 0.0582     | 0.2677     | **0.9006** | **Pass** (Found 🥹)      |

### Analysis

1.  **Hallucination without RAG:** Both Gemma and Llama failed the task without RAG, guessing "Thinking face" (🤔) or "Praying hands" (🙏). This confirms that neither model knew the private rule.
2.  **RAG Effectiveness across Models:** Once RAG was enabled, **both models** successfully retrieved the correct emoji (🥹) and the specific project definition.
3.  **Model Differences:**
    - **Gemma-3-1B** achieved higher lexical overlap (BLEU 0.30) with the reference answer.
    - **Llama 3.2** achieved a slightly higher semantic score (BERTScore 0.90), suggesting it may have paraphrased the definition more naturally while retaining the correct meaning.

---

## 6. Bonus: Describe Your Own RAG Pipeline

The "Emoji" task described above utilized a custom-built RAG pipeline designed to handle **Private Knowledge Injection**.

### Pipeline Design

1.  **Corpus Creation:** A specialized `emoji_corpus.jsonl` was created containing "In-Jokes" and internal team conventions.
2.  **Query Transformation:** The user question (_"Which emoji should I use if I was criticized..."_) is embedded into a vector.
3.  **Retrieval Logic:** The system searches the vector space for the semantic description of the _situation_, not the emoji character itself.
4.  **Generation:** The retrieved definition is passed to the LLM.

### Why this matters?

This pipeline demonstrates that RAG is not just for "textbook knowledge" but is essential for **Domain Adaptation**. It transforms the LLM from a generic chatbot into a project-specific assistant that understands internal slang and non-public rules (like the 🥹 rule) without requiring expensive fine-tuning.

### Why is this "Emoji Task" significant?

While the technical pipeline (Embedding $\to$ Retrieval $\to$ Generation) is identical to the standard lecture QA task (e.g., processing NCKU wiki text), the **nature of the knowledge** differs fundamentally:

1.  **Public Knowledge Verification (The NCKU Case):**

    - Standard RAG tasks often deal with facts the model _partially knows_ (e.g., NCKU is a university in Taiwan).
    - Here, RAG acts as a **fact-checker** to improve precision and reduce hallucinations on specific details (dates, numbers).

2.  **Private Knowledge Injection (The Emoji Case):**
    - This task deals with **arbitrary, domain-specific rules** that do not exist in the model's pre-training data (e.g., defining 🥹 as "begging for leniency").
    - Without RAG, the model's accuracy is effectively **0%** because it relies on general semantic interpretations of emojis.
    - With RAG, the system successfully **overrides** the model's internal bias (Standard Meaning) with the retrieved context (Project Meaning).

**Conclusion:**
This experiment proves that our RAG pipeline is not merely functioning as a search engine for public facts, but is capable of **Domain Adaptation**: allowing the LLM to function correctly in a highly customized environment (e.g., internal company slang, specific coding conventions) without fine-tuning.
