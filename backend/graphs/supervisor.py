"""
Phase 4B — Supervisor Agent Node

Analyses the user's question and decides which sub-agents to invoke.
Returns a route list: any combination of ["sql", "search", "rag", "direct"].

Uses the Groq LLM with a structured routing prompt — no tools called here,
just a routing decision in JSON.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage
from backend.config import settings
from backend.agents.llm_factory import get_llm
from backend.graphs.state import AgentState


SUPERVISOR_PROMPT = """<supervisor>
  <role>Query Router — Analyse the user question and decide which agents to invoke.</role>

  <agents>
    <agent name="sql">
      Use when: questions about sales, orders, products, customers, revenue, inventory,
      database records, or any structured business data query.
    </agent>
    <agent name="search">
      Use when: questions about recent news, current events, live market data,
      external web information, or anything requiring real-time lookup.
    </agent>
    <agent name="rag">
      Use when: questions about uploaded documents, reports, manuals, PDFs,
      or any file-based knowledge the user has provided.
    </agent>
    <agent name="direct">
      Use when: the question is general knowledge, conversational, a greeting,
      or can be answered without any tools.
    </agent>
  </agents>

  <instructions>
    - You MUST return ONLY a valid JSON object — no prose, no markdown, no explanation.
    - Multiple agents can be selected for multi-part questions.
    - Format: {"route": ["agent1", "agent2"], "reasoning": "one sentence why"}
    - Examples:
        Q: "Show total sales by product"
        A: {"route": ["sql"], "reasoning": "Requires database query on sales_orders table."}

        Q: "What is the latest news on AI and what does our uploaded report say?"
        A: {"route": ["search", "rag"], "reasoning": "Needs live news and document retrieval."}

        Q: "Hello, how are you?"
        A: {"route": ["direct"], "reasoning": "Simple greeting, no tools needed."}
  </instructions>
</supervisor>"""


def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor node: reads user_input, returns updated route.
    """
    llm = get_llm(temperature=0.0)   # deterministic routing

    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f"User question: {state['user_input']}")
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        route = parsed.get("route", ["direct"])
        reasoning = parsed.get("reasoning", "")

        # Validate — only allow known agents
        valid = {"sql", "search", "rag", "direct"}
        route = [r for r in route if r in valid] or ["direct"]

        print(f"[Supervisor] Route: {route} | Reason: {reasoning}")
        return {"route": route}

    except Exception as e:
        print(f"[Supervisor] Routing failed: {e} — defaulting to direct")
        return {"route": ["direct"]}
