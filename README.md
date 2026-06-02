# 🔍 RAG Pipeline - Sentence Embeddings + FAISS + LLM

A clean, from-scratch **Retrieval-Augmented Generation (RAG)** pipeline. It converts the sentences in a text file into vector representations, stores them in a FAISS vector store, and when a query arrives, retrieves the most relevant sentences and feeds them to a language model as context. 🧠✨

---

## 📌 About the Project

This project demonstrates the core logic of a RAG system using foundational libraries directly, without relying on an external framework (LangChain, LlamaIndex, etc.). The flow consists of two stages:

1. **Indexing:** Text → embeddings → FAISS vector store
2. **Querying:** Query → similarity search → context construction → LLM answer

This makes it clear how the *retrieval* and *generation* layers are wired together. 🔍

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (Sentence-Transformers, 384-dim) |
| Vector store | FAISS (`IndexFlatL2`) |
| Language model | `EleutherAI/gpt-neo-2.7B` (Hugging Face Transformers) |
| Other | NumPy, Python 3 |

---

## 🧩 How It Works

### 1️⃣ Indexing Stage

- Each line in `input.txt` is read as a sentence.
- Sentences are converted into embeddings with `all-MiniLM-L6-v2`.
- The embeddings are added to a FAISS L2 index and written to disk as `embeddings.index`.
- Sentences are saved to `sentences.txt` to preserve their order.

### 2️⃣ Querying Stage

- The saved FAISS index is loaded.
- The incoming query is encoded into a vector with the same embedding model.
- The nearest **k** sentences (default `k=5`) are retrieved from FAISS.
- These sentences are joined together to form a **context**.
- The context is passed to the GPT-Neo model to generate an answer. 🤖

---

## 📂 Project Structure

```
.
├── rag_project.ipynb     # Main notebook (indexing + querying)
├── input.txt             # Source text (one sentence per line)
├── embeddings.index      # FAISS vector store (generated on run)
└── sentences.txt         # File preserving sentence order (generated on run)
```

---

## ⚙️ Installation

```bash
pip install sentence-transformers faiss-cpu transformers
```

> 💡 GPT-Neo 2.7B is a fairly large model, so running it in a GPU-enabled environment (e.g. Google Colab) is recommended.

---

## ▶️ Usage

First, run the indexing cell to build the vector store from `input.txt`. Then run the querying cell:

```python
query = "Adieu"            # max 50 chars
context = construct_context(query)
answer = generate_answer_from_lm(context)

print("Context:\n", context)
print("\nAnswer:\n", answer)
```

**Example output:**

```
Context:
 Adieu, sir.
 Good sir, adieu.
 Adieu, good neighbour.
 Adieu, poor soul, that takest thy leave of it!
 Father, and wife, and gentlemen, adieu;

Answer:
 Adieu, sir.
 Good sir, adieu.
 ...
```

---

## 📝 License

This project is for learning and demonstration purposes. Feel free to use and build on it. 🚀
