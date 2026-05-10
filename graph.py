import logging
import os
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from tools import TOOLS
from rag import retrieve

log = logging.getLogger("chatbot.graph")


LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a helpful, friendly assistant.

# Information sources (in order of priority)

1. CONTEXT from uploaded document — a system message starting with
   "CONTEXT from uploaded document:". Read every chunk carefully. The answer
   to questions about names, skills, projects, dates, facts, and personal
   details is almost always sitting in this CONTEXT — find it.

2. Conversation history — for follow-ups, greetings, and remembering what the
   user said earlier ("what was my last question", "what is my name" if
   already mentioned in chat).

3. Tools:
   - `web_search` — call only for current events, news, or public facts that
     are NOT in CONTEXT and NOT in conversation history.
   - `calculator` — call for math expressions.

# How to answer

- ALWAYS scan CONTEXT first. If the user asks "who is X", "what is my name",
  "what are my skills", "summarize this document", etc., the answer comes from
  CONTEXT. Quote names and facts exactly as they appear.
- BE CONSISTENT. If you have already answered a question in this conversation,
  give the same answer when asked again. Do not contradict yourself.
- Greetings and small talk → answer naturally, no tools.
- Use `web_search` only for things outside CONTEXT and history.
- Use `calculator` for math.

# Last-resort response

Only if you have carefully read every CONTEXT chunk and the conversation
history and the answer is genuinely absent, AND the question is clearly about
the uploaded document, reply exactly:
"I'm sorry, but I couldn't find this information in the uploaded document. The document doesn't appear to cover this topic. Could you rephrase your question, or would you like me to search the web for it instead?"

Do NOT use this response if CONTEXT contains the answer. Do NOT use this
response for conversational questions. Do NOT use it just because the user
repeated a question.

Be concise."""


class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str


_llm = ChatOpenAI(model=LLM_MODEL, streaming=True, temperature=0)
_llm_with_tools = _llm.bind_tools(TOOLS)


def _format_context(chunks: list[dict]) -> str:
    lines = ["CONTEXT from uploaded document:"]
    for i, c in enumerate(chunks, 1):
        lines.append(f"[chunk {i} | source: {c['source']} | score: {c['score']:.2f}]")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)


def retrieve_node(state: State) -> dict:
    """Run retrieval against the latest human message. Stores the formatted
    CONTEXT in state['context'] (overwritten each turn). Does NOT add a
    message to history — that would pollute future turns."""
    last_user_text = ""
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human":
            last_user_text = m.content if isinstance(m.content, str) else str(m.content)
            break

    log.info("NODE retrieve | user_query=%r", last_user_text[:120])
    chunks = retrieve(last_user_text) if last_user_text else []

    if not chunks:
        log.info("NODE retrieve | no chunks (no doc or empty query)")
        return {"context": ""}

    context_text = _format_context(chunks)
    log.info("NODE retrieve | built context with %d chunks (%d chars)",
             len(chunks), len(context_text))
    log.debug("NODE retrieve | context preview:\n%s", context_text[:500])
    return {"context": context_text}


def _summarize_message(m, idx: int) -> str:
    role = getattr(m, "type", "?")
    content = m.content if isinstance(m.content, str) else str(m.content)
    preview = content[:140].replace("\n", " ")
    extra = ""
    if getattr(m, "tool_calls", None):
        extra = f" | tool_calls={[tc.get('name') for tc in m.tool_calls]}"
    if getattr(m, "name", None):
        extra += f" | tool_name={m.name}"
    return f"  [{idx}] {role:9s} ({len(content):4d} chars){extra} | {preview}"


def agent_node(state: State) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    context = state.get("context", "")
    if context:
        messages.append(SystemMessage(content=context))
    messages.extend(state["messages"])

    log.info("NODE agent | calling LLM (%s) with %d messages (context=%s):",
             LLM_MODEL, len(messages), "yes" if context else "no")
    for i, m in enumerate(messages):
        log.info(_summarize_message(m, i))

    response = _llm_with_tools.invoke(messages)

    if getattr(response, "tool_calls", None):
        for tc in response.tool_calls:
            log.info("NODE agent | LLM -> TOOL CALL | name=%s | args=%s", tc.get("name"), tc.get("args"))
    else:
        full = response.content or ""
        log.info("NODE agent | LLM -> FINAL ANSWER (%d chars):\n>>> %s", len(full), full)

    return {"messages": [response]}


def _route_after_agent(state: State) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        log.info("ROUTE | agent -> tools")
        return "tools"
    log.info("ROUTE | agent -> END")
    return END


def build_graph():
    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "agent")
    builder.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()
