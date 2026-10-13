"""
Phase 4C — Sub-Agent Nodes

Three specialised async agent nodes:
  sql_agent_node     → queries the SQLite business database via MCP tools
  research_agent_node → searches the web via Tavily MCP tools
  rag_agent_node     → searches uploaded documents via ChromaDB + reranker

Each node:
  - reads state['user_input'] and state['messages']
  - calls the appropriate tool(s)
  - returns its output as a string in the state field it owns
"""

import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from backend.config import settings
from backend.agents.llm_factory import get_llm, ainvoke_with_fallback
from backend.graphs.state import AgentState


# ---------------------------------------------------------------------------
# SQL Agent
# ---------------------------------------------------------------------------

async def sql_agent_node(state: AgentState) -> dict:
    """
    Queries the SQLite business database.
    Uses MCP tools: list_tables, describe_table, read_query.
    """
    from backend.mcp.sqlite_client import MCPClientManager

    system_prompt = """<sql_agent>
  <role>SQL Database Expert</role>
  <instructions>
    - You have access to a SQLite business database with products, customers, and sales_orders tables.
    - Always start by calling list_tables, then describe_table to understand the schema.
    - Write precise SQL queries using read_query.
    - Return a clear, structured summary of the results with key numbers highlighted.
    - If the query returns no data, say so clearly.
  </instructions>
</sql_agent>"""

    try:
        mcp = MCPClientManager()
        tools = await mcp.get_tools()
        # Keep only SQL tools
        sql_tools = [t for t in tools if t.name in {"list_tables", "describe_table", "read_query"}]

        llm = get_llm()
        tool_map = {t.name: t for t in sql_tools}

        chat: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_input"])
        ]

        tools_used = []
        for _ in range(5):
            response, _ = await ainvoke_with_fallback(llm, chat, tools=sql_tools)
            chat.append(response)

            if not response.tool_calls:
                return {
                    "sql_output": response.content or "No results found.",
                    "tools_used": tools_used
                }

            for tc in response.tool_calls:
                tools_used.append({"agent": "sql", "tool": tc["name"], "args": tc["args"]})
                result = await tool_map[tc["name"]].ainvoke(tc["args"]) if tc["name"] in tool_map else "Tool not found"
                chat.append(ToolMessage(tool_call_id=tc["id"], content=str(result)))

        return {"sql_output": "SQL agent reached max steps without a final answer.", "tools_used": tools_used}

    except Exception as e:
        return {"sql_output": f"SQL agent error: {str(e)}", "tools_used": []}


# ---------------------------------------------------------------------------
# Research Agent (Web Search)
# ---------------------------------------------------------------------------

async def research_agent_node(state: AgentState) -> dict:
    """
    Searches the web for real-time information using Tavily MCP tools.
    """
    from backend.mcp.sqlite_client import MCPClientManager

    system_prompt = """<research_agent>
  <role>Web Research Expert</role>
  <instructions>
    - Use tavily_search or tavily_qna_search to find current, accurate information.
    - Summarise findings clearly with source URLs where available.
    - Focus on the most recent and relevant results.
    - If search returns no results, say so and suggest what the user could try.
  </instructions>
</research_agent>"""

    try:
        mcp = MCPClientManager()
        tools = await mcp.get_tools()
        search_tools = [t for t in tools if t.name in {"tavily_search", "tavily_qna_search"}]

        llm = get_llm()
        tool_map = {t.name: t for t in search_tools}

        chat: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_input"])
        ]

        tools_used = []
        for _ in range(3):
            response, _ = await ainvoke_with_fallback(llm, chat, tools=search_tools)
            chat.append(response)

            if not response.tool_calls:
                return {
                    "search_output": response.content or "No search results found.",
                    "tools_used": tools_used
                }

            for tc in response.tool_calls:
                tools_used.append({"agent": "search", "tool": tc["name"], "args": tc["args"]})
                result = await tool_map[tc["name"]].ainvoke(tc["args"]) if tc["name"] in tool_map else "Tool not found"
                chat.append(ToolMessage(tool_call_id=tc["id"], content=str(result)))

        return {"search_output": "Research agent reached max steps.", "tools_used": tools_used}

    except Exception as e:
        return {"search_output": f"Research agent error: {str(e)}", "tools_used": []}


# ---------------------------------------------------------------------------
# RAG Agent (Document Search)
# ---------------------------------------------------------------------------

async def rag_agent_node(state: AgentState) -> dict:
    """
    Searches uploaded documents using hybrid retrieval + reranker.
    """
    from backend.mcp.rag_server import rag_search

    system_prompt = """<rag_agent>
  <role>Document Retrieval Expert</role>
  <instructions>
    - Use rag_search to find relevant passages from uploaded documents.
    - Always include source citations: document name and page number.
    - If the document does not contain relevant information, say so clearly.
    - Quote directly from the retrieved passages where appropriate.
  </instructions>
</rag_agent>"""

    try:
        llm = get_llm()
        tool_map = {"rag_search": rag_search}

        chat: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_input"])
        ]

        tools_used = []
        for _ in range(3):
            response, _ = await ainvoke_with_fallback(llm, chat, tools=[rag_search])
            chat.append(response)

            if not response.tool_calls:
                return {
                    "rag_output": response.content or "No document content found.",
                    "tools_used": tools_used
                }

            for tc in response.tool_calls:
                tools_used.append({"agent": "rag", "tool": tc["name"], "args": tc["args"]})
                result = await tool_map[tc["name"]].ainvoke(tc["args"]) if tc["name"] in tool_map else "Tool not found"
                chat.append(ToolMessage(tool_call_id=tc["id"], content=str(result)))

        return {"rag_output": "RAG agent reached max steps.", "tools_used": tools_used}

    except Exception as e:
        return {"rag_output": f"RAG agent error: {str(e)}", "tools_used": []}
