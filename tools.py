import logging
import os

import httpx

from langchain_core.tools import tool
from tavily import TavilyClient


log = logging.getLogger("chatbot.tools")

MATHJS_API_URL = "https://api.mathjs.org/v4/"

_tavily_client: TavilyClient | None = None


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set in environment")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


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
    """Search the web with Tavily and return the top results.
    Use this for questions about current events, news, or anything not in the
    uploaded document."""
    log.info("TOOL web_search | query=%r", query)
    try:
        response = _get_tavily().search(query, max_results=5)
    except Exception as e:
        log.warning("TOOL web_search | error: %s", e)
        return f"Error performing web search: {e}"

    results = response.get("results", []) if isinstance(response, dict) else []
    log.info("TOOL web_search | %d results", len(results))
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        lines.append(f"{i}. {title}\n   {content}\n   {url}")
    return "\n".join(lines)


TOOLS = [web_search, calculator]
