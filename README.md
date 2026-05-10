# RAG + Tools Chatbot

A streaming, multi-turn chatbot that answers questions using three sources:

1. **Conversation history** — remembers previous messages in the same session.
2. **An uploaded document** — PDF or TXT, retrieved from a vector database (RAG).
3. **External tools** the model calls on its own:
   - `web_search` — Tavily Search API
   - `calculator` — math.js public API

Built with **FastAPI**, **LangGraph**, **LangChain**, **ChromaDB**, **OpenAI**, and an optional **Streamlit** UI.

---

## Table of contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the app](#running-the-app)
8. [Streamlit UI guide](#streamlit-ui-guide)
9. [REST API reference](#rest-api-reference)
10. [Example requests](#example-requests)
11. [How it works](#how-it-works)
12. [Design decisions and trade-offs](#design-decisions-and-trade-offs)
13. [Troubleshooting](#troubleshooting)
14. [License and credits](#license-and-credits)

---

## Features

- **Streaming responses (SSE)** — tokens are pushed to the client as they are produced; no buffering.
- **Multi-turn conversation memory** — each chat is identified by a `thread_id`; history is persisted across turns by LangGraph's `InMemorySaver`.
- **Document upload + RAG** — upload a PDF or TXT once, then ask any question about it. Chunks are embedded with OpenAI `text-embedding-3-small` and stored in Chroma.
- **Tool-calling agent** — the model decides on its own when to call `web_search` or `calculator`; tool results are folded back into the answer automatically.
- **Polite grounding fallback** — if a document is uploaded but the answer is not in it, the bot says so clearly and offers to search the web.
- **Two front-ends, one backend** — same agent runs behind both a FastAPI streaming endpoint (for programmatic use) and a Streamlit UI (for human use).
- **Detailed logs** — every retrieval, every LLM call, and every tool invocation is logged with previews so you can trace exactly what the model saw.

---

## Architecture

```
                            +-----------------------+
                            |    Browser / Client   |
                            +-----------+-----------+
                                        |
                           SSE (stream) | HTTPS
                                        v
   +------------+              +----------------------+
   | Streamlit  |  imports     |       FastAPI        |
   |   app.py   +<------+----->+        main.py       |
   +------------+       |      +----------+-----------+
                        |                 |
                        v                 v
                +-----------------------------------+
                |         LangGraph agent           |
                |             graph.py              |
                |                                   |
                |   START -> retrieve -> agent ---  |
                |                          ^     |  |
                |                          |     v  |
                |                       (loop) tools|
                |                                   |
                +---------+----------+--------------+
                          |          |
                          v          v
                  +---------------+ +-------------------------+
                  |    rag.py     | |        tools.py         |
                  |               | |                         |
                  |  PyPDF +      | |  web_search  -> Tavily  |
                  |  Splitter +   | |  calculator  -> mathjs  |
                  |  Chroma +     | |                         |
                  |  OpenAI emb.  | |                         |
                  +---------------+ +-------------------------+
```

The LangGraph agent has three nodes:

| Node | Responsibility |
|---|---|
| `retrieve` | Embed the latest user message, query Chroma for top-k chunks, store the formatted CONTEXT in state. |
| `agent` | Call `gpt-4o-mini` with the system prompt + CONTEXT + conversation history. The LLM either produces a final answer or requests a tool call. |
| `tools` (`ToolNode`) | Execute the requested tool(s), return the result to the `agent`. The `agent` then continues until it produces a final answer. |

---

## Project structure

```
.
├── main.py             FastAPI app, endpoints, SSE streaming
├── app.py              Streamlit UI (alternative to FastAPI)
├── graph.py            LangGraph agent: retrieve -> agent <-> tools
├── rag.py              PDF/TXT ingest, chunking, Chroma retrieval
├── tools.py            web_search (Tavily) + calculator (math.js)
├── requirements.txt    Pinned Python dependencies
├── .env.example        Template for environment variables
├── .gitignore
├── README.md
└── data/               Chroma persistent store (auto-created, gitignored)
```

---

## Prerequisites

- **Python 3.11+** (developed against 3.12)
- An **OpenAI API key** with access to `gpt-4o-mini` and `text-embedding-3-small`
- A **Tavily API key** (free tier: https://tavily.com)

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/AyushChauhanAI/rag-tools-chatbot.git
cd rag-tools-chatbot

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your local .env from the template
cp .env.example .env
# then edit .env and paste your API keys
```

---

## Configuration

All configuration is via environment variables, loaded from `.env` at startup.

| Variable | Required | Purpose | Default |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | OpenAI API key | — |
| `TAVILY_API_KEY` | yes | Tavily search API key | — |
| `LLM_MODEL` | no | OpenAI chat model | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | no | OpenAI embedding model | `text-embedding-3-small` |
| `CHROMA_DIR` | no | Persistent Chroma directory | `./data/chroma` |
| `CHROMA_COLLECTION` | no | Collection name in Chroma | `documents` |
| `CHUNK_SIZE` | no | Document chunk size, in chars | `1000` |
| `CHUNK_OVERLAP` | no | Overlap between adjacent chunks, in chars | `150` |
| `RETRIEVAL_K` | no | Top-k chunks returned by Chroma | `4` |
| `LOG_LEVEL` | no | Python logging level (`DEBUG`, `INFO`, ...) | `INFO` |

See `.env.example` for the template.

---

## Running the app

The same agent (`graph.py`) is exposed by two interfaces. Pick whichever fits your use case.

### Option A — Streamlit UI (recommended for testing)

```bash
streamlit run app.py
```

- Opens at http://localhost:8501
- Upload a document from the sidebar, then chat in the main area.
- Tokens stream live with a cursor.
- Each tool call is shown inline in a collapsible "wrench" panel with its inputs and outputs.

### Option B — FastAPI (for programmatic use / SSE clients)

```bash
uvicorn main:app --reload
```

- Server runs on http://localhost:8000
- Interactive Swagger docs at http://localhost:8000/docs
- See [REST API reference](#rest-api-reference) for the endpoints.

You can run **either or both** — they share the same Chroma store and the same in-memory chat history (within one Python process).

---

## Streamlit UI guide

The Streamlit interface mirrors a typical chat application:

| Area | What it does |
|---|---|
| **Sidebar — Document** | Upload a PDF or TXT, click `Ingest into vector DB`. Shows the current document; click `Clear document` to remove it. |
| **Sidebar — Chats** | List of all threads in this session. Click a thread to switch back to it. Click `+ New chat` to start a fresh thread. |
| **Main area** | Standard chat layout. Messages stream in real time. Tool calls appear as expandable panels under the assistant's reply. |

The Streamlit app imports `graph` directly from `graph.py` and calls
`graph.astream_events(...)` inside an `asyncio.run(...)` block — no HTTP, no
FastAPI required.

---

## REST API reference

All endpoints are under the FastAPI server (`http://localhost:8000` by default).

### `POST /chat` — streaming chat (Server-Sent Events)

**Request body**
```json
{
  "message": "What does the contract say about late fees?",
  "thread_id": "optional-uuid-or-omit-for-new-thread"
}
```

**Response** — `Content-Type: text/event-stream`

| Event | Payload | When |
|---|---|---|
| `thread` | `{ "thread_id": "..." }` | Once at the start of every response |
| `token` | `{ "text": "..." }` | Per streamed token from the LLM (final answer only — tool-call JSON is filtered out) |
| `tool_start` | `{ "name": "...", "input": ... }` | When the LLM invokes a tool |
| `tool_end` | `{ "name": "...", "output": "..." }` | When a tool returns |
| `done` | `{}` | After the final answer is delivered |
| `error` | `{ "error": "..." }` | If something fails mid-stream |

### `POST /upload` — upload a document (replaces the current one)

Multipart form upload with field name `file`. Accepts `.pdf` and `.txt`.

**Response**
```json
{ "status": "ok", "filename": "Resume.pdf", "num_chunks": 4 }
```

### `GET /document` — current document

```json
{ "current": "Resume.pdf" }
```

### `DELETE /document` — clear the current document

Wipes the Chroma collection. Returns `{ "status": "ok" }`.

### `GET /threads` — list past threads (in-memory)

```json
{
  "threads": [
    { "thread_id": "abc-123", "title": "who is ayush" }
  ]
}
```

### `GET /threads/{thread_id}` — full message history of a thread

```json
{
  "thread_id": "abc-123",
  "title": "who is ayush",
  "messages": [
    { "role": "user", "content": "who is ayush" },
    { "role": "assistant", "content": "Ayush Chauhan is..." }
  ]
}
```

### `GET /` — health / discovery

Returns app metadata and the list of available endpoints.

---

## Example requests

```bash
# 1. Upload a document
curl -X POST http://localhost:8000/upload -F "file=@./Resume.pdf"

# 2. Ask a question about the document (streamed answer)
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What programming languages does Ayush know?"}'

# 3. Math (calculator tool fires automatically)
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Compute (1234 * 56) / 7"}'

# 4. Current events (web_search tool fires automatically)
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Latest news on AI regulation 2026"}'

# 5. Continue an existing thread
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "And what was my first question?", "thread_id": "abc-123"}'

# 6. List threads
curl http://localhost:8000/threads

# 7. Inspect a thread's full history
curl http://localhost:8000/threads/abc-123
```

---

## How it works

### Document ingest

When a file is uploaded:

1. `rag.py:ingest_file` reads the bytes into a temp file.
2. `PyPDFLoader` (or `TextLoader`) extracts the raw text page-by-page.
3. `RecursiveCharacterTextSplitter` splits the text into ~1000-character chunks with 150 chars of overlap.
4. Each chunk is embedded with `text-embedding-3-small` and stored in a Chroma collection on disk (`./data/chroma`).
5. The previous collection (if any) is wiped first — only one document lives in the DB at a time.

### Retrieval

Every chat message triggers retrieval:

1. The user's message is embedded.
2. Chroma returns the top-k similar chunks (default `k=4`) along with cosine relevance scores.
3. The chunks are formatted into a single CONTEXT block and stored in the graph state under `context`. They are **not** added to the message history (that would pollute future turns).

### The agent loop

```
START
  -> retrieve   (build CONTEXT)
  -> agent      (LLM sees: system prompt + CONTEXT + history)
        -> tool calls? --no--> END
        -> tool calls? --yes-> tools -> agent (loop until no tool calls)
```

The LLM (`gpt-4o-mini`) is given two tools (`web_search`, `calculator`) and decides on its own whether and which to call. Multiple tool calls in a single turn are supported.

### Streaming

Streaming is end-to-end:

| Layer | Mechanism |
|---|---|
| OpenAI API | `ChatOpenAI(streaming=True)` -> HTTP chunked responses |
| LangGraph | `graph.astream_events(version="v2")` -> Python async generator |
| FastAPI | `StreamingResponse` -> SSE on the wire |
| Streamlit | Same async generator, rendered as it produces tokens |

Tool-call argument chunks are filtered out of the user-visible stream (`if not chunk.tool_call_chunks`) so users never see raw JSON.

### Memory

Chat history is stored by LangGraph's `InMemorySaver`, keyed by `thread_id`.
Each call to the graph passes `config={"configurable": {"thread_id": ...}}`, and
the saver appends new messages to the existing thread state. Restarting the
server clears all threads — this is a deliberate simplification; switch to
`SqliteSaver` for durable history.

---

## Design decisions and trade-offs

**Single `/chat` endpoint, agent decides routing.** RAG context is always
prepared; the LLM calls `web_search` or `calculator` only when warranted. This
satisfies the "chatbot decides when to use tools" requirement without exposing
RAG itself as a tool.

**RAG runs every turn, not as a tool.** Embedding the user's query is fast and
cheap. Always retrieving keeps the agent's behavior simple and predictable. If
nothing relevant is found, the system prompt's "polite couldn't find" branch
takes over.

**No similarity threshold filter.** Short conversational queries (e.g. *"what
is my name"*) often have weak cosine similarity scores against
information-dense chunks even when the answer is right there. Filtering by
score caused real questions to fail. Top-k chunks are always passed to the LLM,
which judges relevance from content.

**One document at a time.** New uploads wipe the Chroma collection. This keeps
retrieval scope obvious and avoids cross-document confusion. Multi-document
support is a small extension (collection-per-doc + a dropdown).

**`InMemorySaver` for chat history.** Keeps history in RAM. Restarting the
server clears all threads. This was chosen for simplicity; swap to
`SqliteSaver` (one-line change in `graph.py`) for persistent history.

**Streaming via `astream_events` v2.** Gives clean access to LLM token events
*and* tool start/end events. Tool-call-argument chunks are filtered out of the
token stream so users never see raw JSON.

**Web search uses Tavily.** Tavily has a free tier, an official Python SDK, and
is purpose-built for LLM/RAG use cases. We initially used DuckDuckGo (no key
needed) but ran into 202 rate-limit errors on repeated queries — switching to
Tavily resolved this with a stable, documented quota.

**Calculator calls a free public API (`api.mathjs.org`).** No API key, no local
`eval()`. Keeps both tools symmetric (each is an HTTP call) and gives the LLM
access to full math.js features (trig, sqrt, log, constants) without writing a
parser.

**Polite grounding response.** When a document is uploaded but doesn't contain
the answer, the bot replies with a polite, explicit message and offers to search
the web instead — clearer for the user than a curt "not found".

**No auth, no rate limiting, no file-size cap.** Out of scope for this assignment.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: tavily` | `tavily-python` not installed in active venv | `pip install -r requirements.txt` |
| `RuntimeError: TAVILY_API_KEY not set` | Missing env var | Add `TAVILY_API_KEY=...` to `.env` and restart |
| Web search returns `202 Ratelimit` | DuckDuckGo rate limit (only if you reverted to the DDG version) | Use Tavily as configured |
| Bot says "I couldn't find this in the uploaded document" for an obvious doc question | Retrieval didn't surface the right chunk | Try smaller `CHUNK_SIZE` (e.g. 300), larger `RETRIEVAL_K` (e.g. 8) |
| Chat history disappears after restart | `InMemorySaver` is RAM-only | Switch to `SqliteSaver` in `graph.py` for persistence |
| Streamed response looks ugly in Swagger | Swagger doesn't render SSE | Use `curl -N`, the Streamlit UI, or a real SSE client |
| `git push` rejected with "non-fast-forward" | Remote has commits you don't have | `git pull --rebase origin main` then push |

---

## License and credits

This project was built as a take-home assignment.

Notable libraries used:

- [FastAPI](https://fastapi.tiangolo.com/) — async web framework
- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [LangChain](https://python.langchain.com/) — LLM, embeddings, retriever wrappers
- [ChromaDB](https://www.trychroma.com/) — persistent vector store
- [Streamlit](https://streamlit.io/) — UI front-end
- [Tavily](https://tavily.com/) — web search API
- [math.js API](https://api.mathjs.org/) — calculator backend
- [pypdf](https://pypdf.readthedocs.io/) — PDF text extraction
