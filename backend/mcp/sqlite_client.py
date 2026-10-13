import os
import sys
import asyncio
from typing import Dict, Any, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from backend.config import settings


class MCPClientManager:
    """Manages connections to multiple MCP servers (SQLite + Tavily Search)."""

    def __init__(self, db_path: Optional[str] = None, tavily_api_key: Optional[str] = None):
        self.db_path = os.path.abspath(db_path or settings.BUSINESS_DB_PATH)
        self.tavily_api_key = tavily_api_key or settings.TAVILY_API_KEY
        
        sqlite_runner = os.path.abspath(os.path.join(os.path.dirname(__file__), "madnh_sqlite_runner.py"))
        tavily_server = os.path.abspath(os.path.join(os.path.dirname(__file__), "tavily_server.py"))
        python_exe = sys.executable

        server_configs: Dict[str, Any] = {
            "sqlite": {
                "command": python_exe,
                "args": [
                    sqlite_runner,
                    "--db-path",
                    self.db_path
                ],
                "transport": "stdio"
            }
        }

        # Add Tavily MCP server config
        env_vars = os.environ.copy()
        if self.tavily_api_key:
            env_vars["TAVILY_API_KEY"] = self.tavily_api_key

        server_configs["tavily"] = {
            "command": python_exe,
            "args": [
                tavily_server
            ],
            "env": env_vars,
            "transport": "stdio"
        }

        self.client = MultiServerMCPClient(server_configs)
        self._tools = None

    async def get_tools(self):
        """Retrieves and caches LangChain-compatible tools across all registered MCP servers."""
        if self._tools is None:
            self._tools = await self.client.get_tools()
        return self._tools


# Backwards compatibility alias
SQLiteMCPManager = MCPClientManager


async def main():
    mgr = MCPClientManager()
    tools = await mgr.get_tools()
    print(f"Loaded {len(tools)} tools across all MCP Servers:")
    for tool in tools:
        print(f" - {tool.name}: {tool.description}")


if __name__ == "__main__":
    asyncio.run(main())
