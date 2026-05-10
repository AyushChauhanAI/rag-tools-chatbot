import logging
import os
import shutil
import tempfile

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

log = logging.getLogger("chatbot.rag")


CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))


_embeddings = None
_vectorstore = None
_current_doc_name: str | None = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
    return _vectorstore


def _reset_vectorstore() -> None:
    """Wipe the entire Chroma store so a new PDF replaces the old one."""
    global _vectorstore
    _vectorstore = None
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    os.makedirs(CHROMA_DIR, exist_ok=True)


def ingest_file(file_bytes: bytes, filename: str) -> dict:
    """Replace the current document. Returns ingest stats."""
    global _current_doc_name

    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in (".pdf", ".txt"):
        raise ValueError("Only .pdf and .txt files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    log.info("INGEST | filename=%s | raw_pages=%d", filename, len(docs))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source"] = filename

    log.info("INGEST | split into %d chunks (size=%d overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)

    _reset_vectorstore()
    store = _get_vectorstore()
    store.add_documents(chunks)
    log.info("INGEST | embedded %d chunks into Chroma collection=%s", len(chunks), CHROMA_COLLECTION)

    _current_doc_name = filename

    return {
        "filename": filename,
        "num_chunks": len(chunks),
    }


def retrieve(query: str) -> list[dict]:
    """Return top-k chunks for the query. The LLM decides relevance — no
    score-based filtering here, because short conversational queries often
    have weak similarity scores even when the answer is in the document."""
    if _current_doc_name is None and not os.path.isdir(CHROMA_DIR):
        log.info("RETRIEVE | skipped (no document uploaded)")
        return []

    store = _get_vectorstore()
    try:
        results = store.similarity_search_with_relevance_scores(query, k=RETRIEVAL_K)
    except Exception as e:
        log.warning("RETRIEVE | error querying Chroma: %s", e)
        return []

    log.info("RETRIEVE | query=%r | k=%d | got %d results", query[:100], RETRIEVAL_K, len(results))

    chunks = []
    for i, (doc, score) in enumerate(results, 1):
        score_val = float(score) if score is not None else 0.0
        chunk = {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": score_val,
        }
        chunks.append(chunk)
        preview = doc.page_content[:120].replace("\n", " ")
        log.info("RETRIEVE | chunk %d | score=%.3f | source=%s | %s...",
                 i, score_val, chunk["source"], preview)

    return chunks


def current_document() -> str | None:
    return _current_doc_name


def clear_document() -> None:
    global _current_doc_name
    _reset_vectorstore()
    _current_doc_name = None
