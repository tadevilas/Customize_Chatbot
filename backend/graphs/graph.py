"""
Phase 4E — LangGraph Graph Builder

Assembles the Supervisor + Sub-Agent + Synthesis nodes into a
compiled LangGraph StateGraph.

Graph topology:
                    ┌─────────────┐
                    │  SUPERVISOR │  (routes to 1+ agents)
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐   ┌──────────┐   ┌──────────┐
      │   SQL   │   │ RESEARCH │   │   RAG    │
      │  Agent  │   │  Agent   │   │  Agent   │
      └────┬────┘   └────┬─────┘   └────┬─────┘
           └─────────────┼──────────────┘
                         ▼
                  ┌─────────────┐
                  │  SYNTHESIS  │  (or DIRECT if no agents)
                  └─────────────┘
                         ▼
                      END
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from backend.graphs.state import AgentState
from backend.graphs.supervisor import supervisor_node
from backend.graphs.agents import sql_agent_node, research_agent_node, rag_agent_node
from backend.graphs.synthesis import synthesis_node, direct_node


# ---------------------------------------------------------------------------
# Routing function — runs after Supervisor to branch to sub-agents
# ---------------------------------------------------------------------------

def route_after_supervisor(state: AgentState) -> list[str]:
    """
    LangGraph conditional edge: returns the list of next nodes to run in parallel.
    Maps route names → node names.
    """
    route = state.get("route", ["direct"])

    route_map = {
        "sql":    "sql_agent",
        "search": "research_agent",
        "rag":    "rag_agent",
        "direct": "direct",
    }

    next_nodes = [route_map[r] for r in route if r in route_map]
    return next_nodes if next_nodes else ["direct"]


def route_after_agents(state: AgentState) -> Literal["synthesis", "direct"]:
    """
    After sub-agents run: if any agent output exists → synthesise.
    If only direct route was chosen → already done in direct_node.
    """
    route = state.get("route", ["direct"])
    if route == ["direct"]:
        return "direct"
    return "synthesis"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Build and compile the multi-agent StateGraph.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor",      supervisor_node)
    graph.add_node("sql_agent",       sql_agent_node)
    graph.add_node("research_agent",  research_agent_node)
    graph.add_node("rag_agent",       rag_agent_node)
    graph.add_node("synthesis",       synthesis_node)
    graph.add_node("direct",          direct_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor → sub-agents (conditional fan-out)
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "sql_agent":      "sql_agent",
            "research_agent": "research_agent",
            "rag_agent":      "rag_agent",
            "direct":         "direct",
        }
    )

    # Sub-agents → synthesis
    graph.add_edge("sql_agent",      "synthesis")
    graph.add_edge("research_agent", "synthesis")
    graph.add_edge("rag_agent",      "synthesis")

    # Synthesis → END
    graph.add_edge("synthesis", END)

    # Direct → END
    graph.add_edge("direct", END)

    return graph.compile()


# Module-level compiled graph (loaded once, reused across requests)
_compiled_graph = None


def get_graph():
    """Return the compiled graph, building it on first call."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
