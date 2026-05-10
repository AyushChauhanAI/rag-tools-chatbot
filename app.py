import asyncio
import logging
import os
import uuid

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

import streamlit as st
from langchain_core.messages import HumanMessage

from graph import graph
from rag import ingest_file, current_document, clear_document


st.set_page_config(page_title="RAG + Tools Chatbot", page_icon=":speech_balloon:", layout="wide")


def _new_thread() -> str:
    return str(uuid.uuid4())


if "threads" not in st.session_state:
    st.session_state.threads = {}
if "active_thread" not in st.session_state:
    tid = _new_thread()
    st.session_state.active_thread = tid
    st.session_state.threads[tid] = {"title": "New chat", "messages": []}


def active() -> dict:
    return st.session_state.threads[st.session_state.active_thread]


with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded is not None:
        if st.button("Ingest into vector DB", type="primary", use_container_width=True):
            with st.spinner("Chunking and embedding..."):
                stats = ingest_file(uploaded.read(), uploaded.name)
            st.success(f"Ingested {stats['filename']} ({stats['num_chunks']} chunks)")

    current = current_document()
    if current:
        st.info(f"Current document: **{current}**")
        if st.button("Clear document", use_container_width=True):
            clear_document()
            st.rerun()
    else:
        st.caption("No document uploaded — chatbot will use tools and history only.")

    st.divider()

    st.header("Chats")
    if st.button(":heavy_plus_sign: New chat", use_container_width=True):
        tid = _new_thread()
        st.session_state.active_thread = tid
        st.session_state.threads[tid] = {"title": "New chat", "messages": []}
        st.rerun()

    for tid, data in list(st.session_state.threads.items())[::-1]:
        label = data["title"][:40] or "New chat"
        is_active = tid == st.session_state.active_thread
        if st.button(
            ("> " if is_active else "  ") + label,
            key=f"thread-{tid}",
            use_container_width=True,
        ):
            st.session_state.active_thread = tid
            st.rerun()


st.title("RAG + Tools Chatbot")
st.caption(f"Thread: `{st.session_state.active_thread[:8]}` | Tools: web_search (Tavily), calculator (math.js)")

for msg in active()["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for tool in msg.get("tools", []):
            with st.expander(f":wrench: {tool['name']}", expanded=False):
                st.markdown(f"**Input:** `{tool['input']}`")
                st.markdown(f"**Output:**\n```\n{tool['output']}\n```")


async def _run_graph(user_input: str, thread_id: str, answer_box, status_box):
    config = {"configurable": {"thread_id": thread_id}}
    answer = ""
    tools_used: list[dict] = []
    pending_tool_idx: dict[str, int] = {}

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        version="v2",
    ):
        kind = event["event"]
        node = event.get("metadata", {}).get("langgraph_node")

        if kind == "on_chat_model_stream" and node == "agent":
            chunk = event["data"]["chunk"]
            if chunk.content and not getattr(chunk, "tool_call_chunks", None):
                answer += chunk.content
                answer_box.markdown(answer + " :black_small_square:")

        elif kind == "on_tool_start":
            tools_used.append({
                "name": event["name"],
                "input": event["data"].get("input"),
                "output": None,
            })
            pending_tool_idx[event["name"]] = len(tools_used) - 1
            status_box.info(f":wrench: Calling **{event['name']}**...")

        elif kind == "on_tool_end":
            output = event["data"].get("output")
            output_str = output.content if hasattr(output, "content") else str(output)
            idx = pending_tool_idx.get(event["name"])
            if idx is not None and tools_used[idx]["output"] is None:
                tools_used[idx]["output"] = output_str[:1500]
            status_box.empty()

    answer_box.markdown(answer)
    return answer, tools_used


if user_input := st.chat_input("Ask anything..."):
    if active()["title"] == "New chat":
        active()["title"] = user_input[:40]

    active()["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_box = st.empty()
        answer_box = st.empty()
        try:
            answer, tools_used = asyncio.run(
                _run_graph(user_input, st.session_state.active_thread, answer_box, status_box)
            )
        except Exception as e:
            answer = f"Error: {e}"
            tools_used = []
            answer_box.error(answer)

        for tool in tools_used:
            with st.expander(f":wrench: {tool['name']}", expanded=False):
                st.markdown(f"**Input:** `{tool['input']}`")
                st.markdown(f"**Output:**\n```\n{tool['output']}\n```")

    active()["messages"].append({
        "role": "assistant",
        "content": answer,
        "tools": tools_used,
    })
    st.rerun()
