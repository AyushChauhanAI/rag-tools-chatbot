import json
import logging
import os
import uuid
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chatbot.main")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from graph import graph
from rag import ingest_file, current_document, clear_document


app = FastAPI(title="RAG + Tools Chatbot")


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


_thread_titles: dict[str, str] = {}


def _sse(event: str, data: dict | str) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, ensure_ascii=False)
    else:
        data = json.dumps({"text": data}, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


@app.post("/chat")
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    is_new = thread_id not in _thread_titles
    if is_new:
        _thread_titles[thread_id] = req.message[:60]

    log.info("=" * 80)
    log.info("CHAT REQUEST | thread=%s | new=%s", thread_id, is_new)
    log.info("CHAT REQUEST | user message: %r", req.message)
    log.info("=" * 80)

    async def event_stream():
        yield _sse("thread", {"thread_id": thread_id})
        full_answer_parts: list[str] = []
        try:
            async for event in graph.astream_events(
                {"messages": [HumanMessage(content=req.message)]},
                config,
                version="v2",
            ):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node")

                if kind == "on_chat_model_stream" and node == "agent":
                    chunk = event["data"]["chunk"]
                    if chunk.content and not getattr(chunk, "tool_call_chunks", None):
                        full_answer_parts.append(chunk.content)
                        yield _sse("token", chunk.content)

                elif kind == "on_tool_start":
                    log.info("TOOL START | name=%s | input=%s", event["name"], event["data"].get("input"))
                    yield _sse("tool_start", {
                        "name": event["name"],
                        "input": event["data"].get("input"),
                    })

                elif kind == "on_tool_end":
                    output = event["data"].get("output")
                    output_str = output.content if hasattr(output, "content") else str(output)
                    log.info("TOOL END   | name=%s | output=%s", event["name"], output_str[:200].replace("\n", " "))
                    yield _sse("tool_end", {
                        "name": event["name"],
                        "output": output_str[:500],
                    })

            final = "".join(full_answer_parts)
            log.info("-" * 80)
            log.info("CHAT ANSWER | thread=%s | %d chars:\n>>> %s", thread_id, len(final), final)
            log.info("=" * 80)
            yield _sse("done", {})
        except Exception as e:
            log.exception("CHAT ERROR | thread=%s", thread_id)
            yield _sse("error", {"error": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename missing")

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in (".pdf", ".txt"):
        raise HTTPException(status_code=400, detail="only .pdf or .txt supported")

    contents = await file.read()
    log.info("UPLOAD REQUEST | filename=%s | bytes=%d", file.filename, len(contents))
    try:
        stats = ingest_file(contents, file.filename)
    except Exception as e:
        log.exception("UPLOAD FAILED | filename=%s", file.filename)
        raise HTTPException(status_code=400, detail=f"ingest failed: {e}")

    log.info("UPLOAD OK | filename=%s | chunks=%d", stats["filename"], stats["num_chunks"])
    return {"status": "ok", **stats}


@app.delete("/document")
async def delete_document():
    clear_document()
    return {"status": "ok"}


@app.get("/document")
async def get_document():
    return {"current": current_document()}


@app.get("/threads")
async def list_threads():
    return {
        "threads": [
            {"thread_id": tid, "title": title}
            for tid, title in _thread_titles.items()
        ]
    }


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="thread not found")

    messages = []
    for m in state.values.get("messages", []):
        mtype = getattr(m, "type", "")
        if mtype not in ("human", "ai"):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content)
        if not content:
            continue
        messages.append({"role": "user" if mtype == "human" else "assistant", "content": content})

    return {
        "thread_id": thread_id,
        "title": _thread_titles.get(thread_id, ""),
        "messages": messages,
    }


@app.get("/")
async def root():
    return {
        "name": "RAG + Tools Chatbot",
        "endpoints": ["/chat", "/upload", "/document", "/threads", "/threads/{thread_id}"],
        "current_document": current_document(),
    }
