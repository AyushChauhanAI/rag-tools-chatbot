# RAG + Tools Chatbot

A streaming chatbot that combines:
- **Conversational chat** with multi-turn history per session (ChatGPT-style threads)
- **RAG** over an uploaded PDF or text file (ChromaDB + OpenAI embeddings)
- **Tools** the agent calls on its own:
  - `web_search` — DuckDuckGo
  - `calculator` — safe math evaluator

Built with FastAPI + LangGraph + LangChain + ChromaDB + OpenAI.

---

## Project structure

```
.
├── main.py             # FastAPI app, endpoints, SSE streaming
├── graph.py            # LangGraph agent: retrieve -> agent <-> tools
├── rag.py              # PDF/TXT ingest, chunking, Chroma retrieval
├── tools.py            # web_search + calculator tools
├── requirements.txt
├── .env.example
└── data/               # Chroma persistent dir (auto-created)
```

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and put your OPENAI_API_KEY
```

## Run

```bash
uvicorn main:app --reload
```

Server runs on `http://localhost:8000`.

---

## Required environment variables

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key (required) | — |
| `LLM_MODEL` | Chat model | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `CHROMA_DIR` | Vector DB persistent dir | `./data/chroma` |
| `CHROMA_COLLECTION` | Collection name | `documents` |
| `CHUNK_SIZE` | Chunk size in chars | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap in chars | `150` |
| `RETRIEVAL_K` | Top-k chunks to retrieve | `4` |
| `SIMILARITY_THRESHOLD` | Min relevance score (0–1) | `0.3` |

See `.env.example` for the full list.

---

## API

### `POST /chat` — streaming chat (SSE)

Request body:
```json
{ "message": "What is 234 * 17?", "thread_id": "optional-uuid" }
```

If `thread_id` is omitted, the server creates a new one and returns it in the first SSE event. Pass it back on follow-up calls to continue the same conversation.

The response is `text/event-stream` with these event types:
- `thread` — emitted once at the start: `{ "thread_id": "..." }`
- `token` — incremental answer tokens: `{ "text": "..." }`
- `tool_start` — agent called a tool: `{ "name": "...", "input": ... }`
- `tool_end` — tool finished: `{ "name": "...", "output": "..." }`
- `done` — stream finished
- `error` — something failed: `{ "error": "..." }`

Example with curl:
```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 234 * 17?"}'
```

### `POST /upload` — replace the uploaded document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@./mydoc.pdf"
```

Response:
```json
{ "status": "ok", "filename": "mydoc.pdf", "num_chunks": 42 }
```

Each upload **replaces** the previous document (only one document is held at a time).

### `GET /document` — what's currently uploaded

```json
{ "current": "mydoc.pdf" }
```

### `DELETE /document` — clear the uploaded document

### `GET /threads` — list past chat threads (in-memory)

### `GET /threads/{thread_id}` — full message history of one thread

---

## Example session

```bash
# 1. Upload a document
curl -X POST http://localhost:8000/upload -F "file=@./contract.pdf"

# 2. Ask a question about it
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does the contract say about late fees?"}'

# 3. Math (calculator tool fires)
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Compute (1234 * 56) / 7"}'

# 4. Current events (web_search tool fires)
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Latest news on AI regulation"}'
```

---

## Design notes & trade-offs

**Single `/chat` endpoint, agent decides routing.** RAG context is always
injected, but the LLM calls `web_search` or `calculator` only when the question
warrants it. This satisfies the "chatbot decides when to use tools" requirement
without exposing RAG itself as a tool.

**RAG is always-on, not a tool.** On every chat message we embed the user's
query, fetch top-k similar chunks from Chroma, and inject them as a
`SystemMessage` before the LLM call. The LLM decides whether the chunks are
actually relevant.

**No similarity threshold filter.** Short conversational queries
(e.g. "what is my name") often have weak cosine similarity scores against
information-dense chunks even when the answer is right there. Filtering by
score caused real questions to fail. Instead, top-k chunks are always passed
to the LLM, and the LLM judges relevance from content. The system prompt
instructs the LLM to reply *"I could not find this in the uploaded document."*
when the answer isn't in CONTEXT.

**One document at a time.** New uploads wipe the Chroma collection. This keeps
retrieval scope obvious and avoids cross-document confusion. Multi-document
support is a small extension (collection-per-doc + a dropdown).

**InMemorySaver for chat history.** LangGraph's `InMemorySaver` keeps thread
history in RAM. **Restarting the server clears all threads.** This was a
deliberate simplification; swapping in `SqliteSaver` is a one-line change if
durability is needed.

**Streaming via `astream_events` v2.** This gives clean access to LLM token
events and tool start/end events. Tool-call-argument chunks are filtered out of
the token stream (`if not chunk.tool_call_chunks`) so users never see raw JSON.

**Calculator calls a free public API (`api.mathjs.org`).** No API key, no
local `eval()`. This keeps the two tools symmetric (both are external API
calls) and gives the LLM access to full math.js features (trig, sqrt, log,
constants like `pi` and `e`) without writing a parser.

**File size limits.** None enforced — `UploadFile` streams to disk-backed
temporary storage. Add a `Content-Length` check in production.

**No auth, no rate limiting.** Out of scope for the assignment.

---

## Quick smoke test

```bash
# 1. Health
curl http://localhost:8000/

# 2. Math (no PDF needed)
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "(15 * 4) / 6"}'

# Expect: tool_start + tool_end events for calculator, then streamed answer.
```
