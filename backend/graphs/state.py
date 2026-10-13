"""
Phase 4A — AgentState

The typed state object that flows through every node in the LangGraph graph.
Every node reads from this state and returns a partial update to it.
"""

from typing import List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """
    Central state shared across all LangGraph nodes.

    Fields:
      messages        : full conversation history (HumanMessage, AIMessage, ToolMessage)
      user_input      : the raw user question for this turn
      route           : which agent(s) the Supervisor chose: "sql" | "search" | "rag" | "direct"
      sql_output      : result from the SQL Agent
      search_output   : result from the Research Agent
      rag_output      : result from the RAG Agent
      final_response  : synthesized final answer
      tools_used      : list of tool call dicts logged during this turn
      model_used      : which Groq model was used
      error           : any error message (empty string if none)
    """
    messages:        Annotated[List[BaseMessage], operator.add]
    user_input:      str
    route:           List[str]          # can route to multiple agents
    sql_output:      str
    search_output:   str
    rag_output:      str
    final_response:  str
    tools_used:      Annotated[List[Dict[str, Any]], operator.add]
    model_used:      str
    error:           str
