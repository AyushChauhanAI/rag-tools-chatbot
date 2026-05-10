import logging

import httpx

from langchain_core.tools import tool
from duckduckgo_search import DDGS


log = logging.getLogger("chatbot.tools")

MATHJS_API_URL = "https://api.mathjs.org/v4/"


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression by calling the math.js API.
    Supports arithmetic (+ - * / ^), parentheses, and functions like
    sqrt, sin, cos, log, etc.
    Example inputs: "2 + 2", "(15 * 4) / 6", "sqrt(144)", "sin(pi/2)"."""
    log.info("TOOL calculator | expression=%r", expression)
    try:
        response = httpx.get(
            MATHJS_API_URL,
            params={"expr": expression},
            timeout=10.0,
        )
    except httpx.RequestError as e:
        log.warning("TOOL calculator | network error: %s", e)
        return f"Error reaching calculator API: {e}"

    if response.status_code != 200:
        log.warning("TOOL calculator | api error %d: %s", response.status_code, response.text.strip())
        return f"Error: invalid expression ({response.text.strip()})"

    result = response.text.strip()
    log.info("TOOL calculator | result=%s", result)
    return result


@tool
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo and return the top results.
    Use this for questions about current events, news, or anything not in the
    uploaded document."""
    log.info("TOOL web_search | query=%r", query)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        log.warning("TOOL web_search | error: %s", e)
        return f"Error performing web search: {e}"

    log.info("TOOL web_search | %d results", len(results))
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"{i}. {title}\n   {body}\n   {href}")
    return "\n".join(lines)


TOOLS = [web_search, calculator]
