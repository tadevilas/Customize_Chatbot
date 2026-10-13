import os
from typing import List, Optional
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

tavily_api_key = os.getenv("TAVILY_API_KEY", "")
mcp = FastMCP("tavily-search-server")


def get_tavily_client() -> Optional[TavilyClient]:
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)


@mcp.tool()
def tavily_search(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> str:
    """Performs a real-time web search using Tavily API.
    
    Args:
        query: The search query string.
        search_depth: 'basic' for fast search or 'advanced' for deeper research.
        max_results: Maximum number of search results to return (default 5).
        include_domains: Optional list of domains to include.
        exclude_domains: Optional list of domains to exclude.
    """
    client = get_tavily_client()
    if not client:
        return "Error: TAVILY_API_KEY environment variable is not set. Please configure it in your .env file."

    try:
        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_domains=include_domains or [],
            exclude_domains=exclude_domains or []
        )
        results = response.get("results", [])
        if not results:
            return f"No results found for query: '{query}'"

        formatted = []
        for i, res in enumerate(results, start=1):
            title = res.get("title", "No Title")
            url = res.get("url", "")
            content = res.get("content", "")
            formatted.append(f"[{i}] {title}\nURL: {url}\nSnippet: {content}\n")

        return "\n".join(formatted)
    except Exception as e:
        return f"Error executing Tavily search: {str(e)}"


@mcp.tool()
def tavily_qna_search(query: str) -> str:
    """Searches the web and directly returns a synthesized answer with references."""
    client = get_tavily_client()
    if not client:
        return "Error: TAVILY_API_KEY environment variable is not set."

    try:
        answer = client.qna_search(query=query)
        return answer
    except Exception as e:
        return f"Error executing Tavily Q&A search: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
