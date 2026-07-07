# Chat with Your Documents

A private AI assistant that answers questions about your documents — with citations, and without sending anything to the cloud.

Upload PDFs or text files and ask questions in plain language. The assistant finds the most relevant passages and answers using **only** those, citing the exact document and page. If the answer isn't in your documents, it says so — instead of making something up.

<img width="1512" height="982" alt="Screenshot 2026-07-07 at 6 11 20 PM" src="https://github.com/user-attachments/assets/06953a3a-c9fc-46a0-a8dd-6e534fcd008d" />

## Why it's different

- **Grounded, cited answers.** Every response links back to the source document and page, so answers are verifiable — no hallucinated facts.
- **Runs entirely on your machine.** Powered by local models via [Ollama](https://ollama.com), your documents never leave your computer. Safe for confidential, legal, or sensitive material.
- **Works with your files.** PDFs and text files, several at once.

Two sample PDFs (`ACME_Employee_Handbook.pdf`, `ACME_Customer_Policies.pdf`) are included so you can try it in under a minute.

## How it works

A standard Retrieval-Augmented Generation (RAG) pipeline:

1. **Load** documents (PDF or text).
2. **Split** them into overlapping chunks.
3. **Embed** each chunk into a vector.
4. **Retrieve** the chunks most similar to your question.
5. **Answer** from only those chunks, with the model instructed to cite its sources and to admit when the answer isn't in the documents.

## Quickstart

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- Pull the two models (one for embeddings, one for answering):
  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen2.5:7b
  ```

### Install and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the app, upload one or more documents in the sidebar, click **Index**, and start asking questions.

## Tech stack

- **Python** + **Streamlit** for the interface
- **Ollama** for local embeddings (`nomic-embed-text`) and generation (`qwen2.5`)
- **LangChain** model interfaces
- In-memory cosine-similarity retrieval — simple and transparent, and swappable for a dedicated vector database (FAISS, Chroma, pgvector) for larger document sets

## Using a hosted model instead

The embedding and chat models are injectable, so you can trade the local models for a hosted provider (e.g. OpenAI or Anthropic) when you want higher answer quality — the rest of the pipeline stays the same:

```python
from rag import RAGPipeline
# rag = RAGPipeline(embeddings=YourEmbeddings(), llm=YourChatModel())
```

## Project structure

```
rag-assistant/
├── rag.py             # the RAG engine (load, chunk, embed, retrieve, answer)
├── app.py             # Streamlit UI
├── requirements.txt
├── docs/ACME_Employee_Handbook.pdf    # sample document
└── docs/ACME_Customer_Policies.pdf    # sample document
```

## About

Built by **Muhammad Adeel Tahir**. I build custom AI assistants and automations for businesses — document Q&A, workflow automation, and integrations with the tools you already use.

📫 adeel.tahir98@yahoo.com · https://www.linkedin.com/in/adeeltahir1998
