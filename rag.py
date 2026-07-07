"""
rag.py — a small, transparent Retrieval-Augmented Generation (RAG) pipeline.

Portfolio piece: "Chat with your documents."

Flow:
  1. Load documents (PDF or plain text)
  2. Split them into overlapping chunks
  3. Embed each chunk into a vector
  4. Store the vectors in a simple in-memory index
  5. At query time: embed the question, find the most similar chunks, and
     have an LLM answer using ONLY those chunks — citing its sources.

Runs fully locally on Ollama by default (no API key, private, works offline).
Both models are injectable, so you can swap in a hosted model for higher
quality, or a fake one for tests.

The vector store here is a straightforward in-memory cosine-similarity
search — easy to read and perfect for demos. For a large corpus you'd swap
it for a dedicated vector database (FAISS, Chroma, pgvector); the surrounding
interface would not change.
"""

from pathlib import Path
import numpy as np
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You answer questions using ONLY the provided context.
- If the answer is not in the context, say you don't know from the documents.
- Cite the sources you used by their [number].
- Be concise and accurate."""


def split_text(text, chunk_size=800, overlap=120):
    """Split text into overlapping chunks, breaking on whitespace when possible."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # prefer to end on a word boundary
            ws = text.rfind(" ", start, end)
            if ws > start:
                end = ws
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, 0)
    return chunks


class RAGPipeline:
    def __init__(self, embeddings=None, llm=None, chunk_size=800, overlap=120, k=4):
        # Default to local Ollama models; inject your own to swap them out.
        self.embeddings = embeddings or OllamaEmbeddings(model="nomic-embed-text")
        self.llm = llm or ChatOllama(model="qwen2.5:7b", temperature=0)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.k = k

        self._texts = []       # chunk texts
        self._metas = []       # chunk metadata (source, page)
        self._vectors = None   # np.ndarray of shape [n_chunks, dim]

    # ---- ingestion ----------------------------------------------------
    def _read_file(self, path):
        """Return a list of (text, metadata) pairs for one file."""
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return [
                (page.extract_text() or "", {"source": path.name, "page": i + 1})
                for i, page in enumerate(reader.pages)
            ]
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [(text, {"source": path.name})]

    def add_documents(self, paths):
        """Load, chunk, embed, and index the given files. Returns chunk count."""
        new_texts, new_metas = [], []
        for p in paths:
            for text, meta in self._read_file(p):
                for chunk in split_text(text, self.chunk_size, self.overlap):
                    new_texts.append(chunk)
                    new_metas.append(meta)

        if not new_texts:
            return 0

        vectors = np.array(self.embeddings.embed_documents(new_texts), dtype="float32")
        self._texts.extend(new_texts)
        self._metas.extend(new_metas)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        return len(new_texts)

    # ---- retrieval ----------------------------------------------------
    def _most_similar(self, question):
        """Return the indices of the top-k chunks most similar to the question."""
        q = np.array(self.embeddings.embed_query(question), dtype="float32")
        mat = self._vectors
        sims = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-8)
        return np.argsort(-sims)[: self.k]

    def query(self, question):
        """Retrieve relevant chunks and answer the question from them."""
        if self._vectors is None:
            raise ValueError("No documents indexed yet — call add_documents() first.")

        context_parts, sources = [], []
        for n, i in enumerate(self._most_similar(question), start=1):
            meta = self._metas[i]
            label = meta.get("source", "document")
            if "page" in meta:
                label += f" p.{meta['page']}"
            context_parts.append(f"[{n}] (from {label})\n{self._texts[i]}")
            sources.append({"n": n, "source": label})

        context = "\n\n".join(context_parts)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        response = self.llm.invoke(messages)
        answer = getattr(response, "content", str(response))
        return {"answer": answer, "sources": sources}
