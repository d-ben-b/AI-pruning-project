# 機器學習講義問答系統：RAG vs. 無 RAG 實驗報告

## 1. 實驗動機與目標

本次實驗的目標是評估 **Retrieval-Augmented Generation (RAG)** 對於「機器學習概念問答」的實際效果：

- 預訓練語言模型本身已經具備一定的機器學習背景知識，理論上就能回答多數基本 ML 概念題。
- 然而，課堂上通常會有**特定用語、簡報風格與定義方式**，希望模型的回答能更貼近講義內容與助教期待的標準解答。
- 因此，比較：
  - **Baseline（無 RAG）**：純 LLM 憑自身知識回答。
  - **RAG（有講義檢索）**：先從課堂 PPT 講義中檢索相關投影片，再把檢索結果 + 問題一起丟給同一個 LLM 回答。

---

## 2. 實驗設定

### 2.1 模型與工具

- 語言模型（LLM）：`google/gemma-3-1b-it`
- 嵌入模型（Embedding）：`all-MiniLM-L6-v2`（SentenceTransformers）
- 向量索引：FAISS `IndexFlatIP` + L2-normalized embeddings（等價於 cosine similarity）
- 評估工具：
  - `evaluate`：BLEU、ROUGE-L
  - `bert-score`：BERTScore（使用預設英文 RoBERTa 模型）

### 2.2 機器學習講義語料（RAG corpus）

資料來源：課堂 PPTX 檔案（部分章節示意）：

- `0. Introduction(1).pptx`
- `1. Single Layer Perceptron.pptx`
- `2. Multilayer perceptron.pptx`
- `Loss_and_Optimization_functions.pptx`
- `(3)Introduction to Deep Learning 2024.pptx`
- `GAN2024.pptx`
- `(16) VAE new 2026.pptx`
- `(11) IntroToTransformer-2024.pptx`
- `13. Diffusion-model_V3 (2).pptx`
- `(14)reinforment learning with NN.pptx`
- `(15)Self-supervised Contrastive Learning_modified (1).pptx`
- `Semi-supervised learning.pptx`
- `CAM calss ativation map.pptx`
- ……等機器學習與深度學習相關講義。

處理流程：

1. 使用 `python-pptx` 逐一讀取所有 `.pptx` 檔案。
2. 對每一張投影片：
   - 擷取所有具有 `shape.text` 的文字框。
   - 將同一張投影片的文字以換行符號 `\n` 串接。
3. 將結果寫成 JSONL 檔案 `ml_lectures_corpus.jsonl`，每行格式為：

   ```json
   { "text": "<slide_text>" }
   ```

4. 使用 `load_doc_from_path()` 將每一行視為一個 `DocChunk`，並建立對應的向量索引：

   - 對 `text` 做 embedding。
   - 正規化後加入 FAISS index。

最終得到：

- `chunk_list_ml`：來自所有講義的文字 chunk（以投影片為單位）。
- `index_ml`：對應的 FAISS 向量索引。

### 2.3 問答資料集（ML QA Dataset）

建立一個小型機器學習概念 QA 資料集 `ml_qa_dataset.jsonl`，共 10 題，每題包含：

- `id`: 題目編號
- `question`: 問題（英文）
- `answer`: 參考標準解答（英文，偏「教科書式定義」）

範例（節錄）：

```json
{"id": 1, "question": "What is overfitting in machine learning?", "answer": "Overfitting happens when a model fits the training data too closely, capturing noise and random fluctuations instead of the underlying pattern, so it performs well on training data but poorly on unseen test data."}
{"id": 2, "question": "What is underfitting?", "answer": "Underfitting occurs when a model is too simple to capture the underlying relationship in the data, so it performs poorly on both training and test data."}
{"id": 3, "question": "What is the purpose of splitting data into training and test sets?", "answer": "The training set is used to learn the model parameters, while the test set is used to evaluate how well the trained model generalizes to unseen data."}
...
```

題目類型涵蓋：

- 過擬合 / 欠擬合（overfitting / underfitting）
- 訓練 / 測試集的用途
- bias–variance tradeoff
- L2 regularization
- classification vs regression
- k-fold cross-validation
- gradient descent
- sigmoid 函數的用途
- SVM 的基本概念

---

## 3. 方法

### 3.1 Baseline：無 RAG 的問答（`predict_no_rag`）

- 輸入：問題文字 `question`。
- 步驟：

  1. 使用 `lm_template(question)` 建立 system + user 的對話格式，system prompt 要求簡潔定義（1–3 句）。
  2. 呼叫 `generate(prompt, tokenizer, model)` 產生回答。

- 不使用任何外部知識或講義內容。

### 3.2 RAG：使用講義檢索的問答（`predict_with_rag_ml`）

- 輸入：問題文字 `question`。

- 步驟：

  1. 將問題包裝為 `LookupQuery`，形成搜尋 query。

  2. 使用嵌入模型 `all-MiniLM-L6-v2` 對 query 做 embedding。

  3. 在 `index_ml` 上以 inner product + L2 normalization 進行向量搜尋，取 Top-k（實驗中使用 `top_k = 3`）。

  4. 將取得的 `DocChunk` 內容串成 context：

     ```text
     Reference 1: <chunk1 content>
     Reference 2: <chunk2 content>
     Reference 3: <chunk3 content>
     ```

  5. 將 context 與 question 一起放進 prompt template：

     ```text
     References:
     <context>

     Question:
     <question>
     Do not use markdown syntax to answer and put the answer after "Answer:".
     ```

  6. 使用同一個 `lm_template()` + `generate()` 呼叫 Gemma-3-1b-it 作答。

- 差別在於：RAG 在作答前先「看到」講義中與問題最相關的投影片文字。

### 3.3 Case Study: Emoji RAG for Internal “In-Joke” Semantics

To further stress-test RAG beyond standard ML lecture QA, I constructed a small **emoji knowledge base** that encodes project-specific “in-joke” meanings which do not exist in public pretraining data.

Concretely, I defined a custom semantic mapping for 🥹 as:

> **Reference meaning**  
> _“In this project, we use the emoji 🥹 to represent the feeling that your assignment was heavily criticized, but you still want to look obedient and are begging the teacher or TA to be lenient.”_

This mapping was stored as one entry in `emoji_corpus.jsonl` and indexed by the same RAG pipeline as before (SentenceTransformers + FAISS).

#### Task setup

- **Corpus**: `emoji_corpus.jsonl`
  - 15 emoji entries, each with a short name, tags, a project-specific meaning, and an example usage.
- **Question (inference)**:

  > _“In our project, if I want to use one emoji to mean ‘my assignment was heavily criticized, but I still pretend to be obedient and beg the teacher or TA to be lenient’, which emoji should I use and why? Please answer in English.”_

- **Reference answer (for automatic metrics)**:

  > _“In this project, we use the emoji 🥹 to represent the feeling that your assignment was heavily criticized, but you still want to look obedient and are begging the teacher or TA to be lenient.”_

- **Systems compared**:
  - **No RAG**: direct prompting of the LLM.
  - **With RAG (emoji)**: same LLM + retrieval over `emoji_corpus.jsonl`.

#### Quantitative results

| System | BLEU   | ROUGE-L | BERTScore (F1) |
| ------ | ------ | ------- | -------------- |
| No RAG | 0.0000 | 0.1690  | 0.8671         |
| RAG    | 0.0000 | 0.2785  | 0.8999         |

BLEU is close to zero for both systems because the reference is very short and the model answers are longer free-form explanations, so exact n-gram overlap is limited. However, **ROUGE-L** and **BERTScore** both improve when RAG is enabled, indicating that the RAG-augmented answer is semantically closer to the reference description of 🥹.

#### Emoji accuracy (task-specific metric)

For this task, the truly critical behaviour is whether the model actually selects the **correct emoji** that we defined in the custom corpus. I therefore add a simple task-specific metric:

- **Expected emoji**: 🥹
- **No RAG**: does not contain 🥹
- **With RAG**: explicitly outputs `Answer: 🥹`

In other words:

- **No RAG** fails to select the correct emoji at all and instead produces a generic description (e.g., an exasperated or thinking face).
- **With RAG** correctly selects 🥹 and provides a justification that closely matches the project-specific definition stored in `emoji_corpus.jsonl`.

This case study illustrates an important property of RAG:

> **RAG is not only helpful for recovering “public” factual knowledge, but also for aligning an LLM with highly local, project-specific semantics (such as internal emoji conventions) that are absent from its pretraining data.**

output of the code

```bash
[No RAG]
The best emoji to represent that situation is a slightly exasperated face with a question mark above it – often represented by 🤔. This conveys a sense of subtle resistance while acknowledging the criticism without directly confronting the issue.

[With RAG]
The “🥹” emoji best represents that feeling of being deeply disappointed but trying to appear compliant, reflecting a situation where your work was significantly criticized but you’re still seeking leniency. It conveys a sense of longing and a desperate attempt to avoid negative consequences. Answer: 🥹
Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

=== Scores vs. reference ===
Reference:
In this project, we use the emoji 🥹 to represent the feeling that your assignment was heavily criticized, but you still want to look obedient and are begging the teacher or TA to be lenient.

[No RAG]
BLEU     : 0.0000
ROUGE-L  : 0.1690
BERTScore: 0.8671

[With RAG]
BLEU     : 0.0000
ROUGE-L  : 0.2785
BERTScore: 0.8999

=== Emoji accuracy ===
Expected emoji: 🥹
No RAG contains it?   False
With RAG contains it? True
```

---

## 4. 評估設定

### 4.1 指標

對每一題，我們得到：

- 參考答案：`answer`
- 模型答案：`prediction`

使用下列指標評估整體表現：

1. **BLEU**

   - 測量 n-gram 重疊程度，偏向字面相似度。
   - 使用 `evaluate.load("bleu")` 實作。

2. **ROUGE-L**

   - 對應 longest common subsequence (LCS) 的重疊程度。
   - 使用 `evaluate.load("rouge")` 取 `rougeL`。

3. **BERTScore (F1)**

   - 基於 contextual embeddings 的語意相似度。
   - 使用 `bert-score` 套件，語言設定為 `lang="en"`。

### 4.2 評估函式

使用同一個 `evaluate_model()` 函式對比不同系統：

- `predict_fn = predict_no_rag` → Baseline（no RAG）
- `predict_fn = predict_with_rag_ml` → With RAG（ML lectures）

---

## 5. 實驗結果

對 10 題 ML 問答的整體結果如下：

| System            | BLEU   | ROUGE-L | BERTScore-F1 |
| ----------------- | ------ | ------- | ------------ |
| Baseline (no RAG) | 0.0278 | 0.1914  | 0.8483       |
| RAG (ML lectures) | 0.1104 | 0.3255  | 0.8882       |

觀察：

- **BLEU：0.0278 → 0.1104**

  - 約提升 4 倍，顯示 RAG 版本在字詞與短語層面上與參考解答重疊度大幅提升。

- **ROUGE-L：0.1914 → 0.3255**

  - LCS 長度明顯增加，代表整體回答結構與標準答案更加相似。

- **BERTScore-F1：0.8483 → 0.8882**

  - 原本 baseline 的語意相似度已經偏高（模型本身懂 ML 概念）。
  - 有了講義檢索後，語意分數進一步提升約 0.04，顯示回答內容在語意上更貼近標準解答。

- **不同模型比較Gemma-3-1B 與 Llama3.2 在有無 RAG 下的表現：**
| Backend            | RAG | BLEU  | ROUGE-L | BERTScore | Emoji 命中 (🥹) |
|-------------------|-----|-------|---------|-----------|-----------------|
| Gemma-3-1B (HF)   | ✗   | 0.0000 | 0.1667   | 0.8418| 否              |
| Gemma-3-1B (HF)   | ✓   | 0.3095 |0.3095 | 0.8907  | 是              |
| Llama3.2 (Ollama) | ✗   | 0.0234| 0.1333  | 0.8569    | 否              |
| Llama3.2 (Ollama) | ✓   | 0.0582| 0.2677  | 0.9006    | 是              |

在這個 emoji 語意任務中，我們刻意選擇一個「只有在本專案的資料庫中才定義清楚」的規則：  
「當作業被改得很慘，但仍然想裝乖、拜託老師或助教手下留情時，要使用 🥹 這個 emoji。」

無論是 Hugging Face 後端（Gemma 3 1B）或 Ollama 後端（Llama3.2），在 *不使用 RAG* 的情況下，
模型都傾向選擇其他「合理但不符合我們規則」的表情，例如 🤔、🙏 等。
這說明僅靠預訓練知識，模型並不知道我們專案中自訂的 emoji 規則。

加入 RAG 後，兩個後端都可以穩定地選擇正確的 🥹，而且在 BLEU / ROUGE-L / BERTScore 指標上
與人工參考答案的距離也明顯縮短，顯示模型確實有利用外部 corpus 中的定義來調整回答內容。


---

## 6. 討論

1. **Baseline 已具備不錯的概念理解能力**

   - BERTScore 在無 RAG 下即有約 0.85，代表預訓練 LLM 本身就能正確描述多數基本 ML 概念。
   - 但 BLEU 與 ROUGE-L 偏低，反映出模型傾向給出較口語化、冗長或風格不同的回答，與標準解答在字面形式上差異較大。

2. **RAG 的主要貢獻：對齊「講義風格」與「標準答案用語」**

   - 將講義投影片文字納入 context 後，模型回答更容易使用簡報中的用字與句型。
   - 這直接帶動 BLEU、ROUGE-L 的提升，也讓 BERTScore 有明顯進步。
   - 從指標可以推測，RAG 在這個設定下主要扮演「讓模型的表達更貼近授課內容」的角色，而不是從零補充模型完全缺乏的概念知識。

3. **RAG 的效益與題目型態有關**

   - 本次 QA 題目多為「標準定義題」，簡報內容通常有明確的一兩句定義。
   - 這種類型題目特別適合 RAG：檢索到相關 slide 後，模型容易生成與標準解答高度重疊的回覆。
   - 之後若設計「定量細節、課堂特有例子或符號」類型的題目，RAG 的效果可能會更明顯。

---

## 7. 結論與未來工作

### 7.1 結論

- 在機器學習講義 QA 任務上，**加入 RAG（講義檢索）可以穩定提升回答品質**：

  - BLEU 與 ROUGE-L 顯著增加，表示字面與結構更貼近標準答案。
  - BERTScore 也有明確提升，顯示語意上更接近教科書式定義。

- 本次實驗驗證了：即使預訓練 LLM 已經懂機器學習，RAG 仍然能夠：

  1. 帶入課程特有的用語與風格。
  2. 幫助模型產生與講義一致的、較「標準答案化」的回答。

### 7.2 未來工作構想

接下來可以進一步設計「**預訓練模型明顯無法回答、但講義有答案**」的題目，讓 RAG 的差異更極端，例如：

- 課程中才出現的特定 notation、符號定義或縮寫。
- 僅存在於講義中、但不常見於一般網路資料的例子或圖表說明。
- 與這門課特定作業、專案說明內容高度綁定的問題。

透過這類題目，可以更清楚分析：

- 在**一般知識題**上，RAG 是否只是「微調表達與用語」；
- 在**模型原本不知道的課程專有知識**上，RAG 是否能讓模型從「不會回答」變成「可以給出接近講義的正確解答」。

這將是下一階段更有趣、也更能凸顯 RAG 價值的實驗方向。
