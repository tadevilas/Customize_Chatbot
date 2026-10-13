"""
Phase 4D — Response Synthesis Node + Direct Answer Node

synthesis_node:
  Combines outputs from SQL, Research, and RAG agents into one
  coherent, well-structured final answer with citations.

direct_node:
  Handles questions that don't need any tools — answers directly
  from the LLM's knowledge (greetings, general facts, etc.)
"""

from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.llm_factory import get_llm, ainvoke_with_fallback
from backend.graphs.state import AgentState


SYNTHESIS_PROMPT = """<synthesizer>
  <role>Response Synthesizer — Combine multi-agent outputs into one final answer.</role>
  <instructions>
    - You will receive outputs from one or more specialised agents.
    - Combine them into a single, coherent, well-structured response.
    - Remove duplicates and contradictions — prefer the most specific and recent data.
    - Always include source citations: database table names, document names/pages, or URLs.
    - Use markdown formatting: headers, bullet points, tables where helpful.
    - Be concise — do not repeat the same information multiple times.
    - If any agent returned an error or no data, note it briefly but focus on what was found.
  </instructions>
</synthesizer>"""


async def synthesis_node(state: AgentState) -> dict:
    """
    Synthesises outputs from SQL, Research, and RAG agents.
    Called only when at least one sub-agent was invoked.
    """
    llm = get_llm(temperature=0.2)

    # Build context from whichever agents ran
    context_parts = []

    if state.get("sql_output"):
        context_parts.append(f"<sql_results>\n{state['sql_output']}\n</sql_results>")

    if state.get("search_output"):
        context_parts.append(f"<web_search_results>\n{state['search_output']}\n</web_search_results>")

    if state.get("rag_output"):
        context_parts.append(f"<document_results>\n{state['rag_output']}\n</document_results>")

    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(content=(
            f"User question: {state['user_input']}\n\n"
            f"Agent outputs to synthesise:\n{context}"
        ))
    ]

    try:
        response, model_used = await ainvoke_with_fallback(llm, messages)
        return {
            "final_response": response.content or "I could not generate a response.",
            "model_used": model_used
        }
    except Exception as e:
        return {
            "final_response": f"Synthesis error: {str(e)}",
            "model_used": "error"
        }


DIRECT_PROMPT = """<direct_agent>
  <role>General Knowledge Assistant</role>
  <instructions>
    - Answer the user's question directly from your knowledge.
    - Be conversational, helpful, and concise.
    - If the question is a greeting or small talk, respond naturally.
    - If the question requires real-time data you don't have, say so and suggest using search.
  </instructions>
</direct_agent>"""


async def direct_node(state: AgentState) -> dict:
    """
    Answers directly from LLM knowledge — no tools needed.
    """
    llm = get_llm(temperature=0.2)

    # Include conversation history for context
    messages = [SystemMessage(content=DIRECT_PROMPT)]
    for msg in state.get("messages", []):
        messages.append(msg)

    # Ensure the current user input is the last message
    if not messages or not isinstance(messages[-1], HumanMessage):
        messages.append(HumanMessage(content=state["user_input"]))

    try:
        response, model_used = await ainvoke_with_fallback(llm, messages)
        return {
            "final_response": response.content or "I'm not sure how to answer that.",
            "model_used": model_used
        }
    except Exception as e:
        return {
            "final_response": f"Error: {str(e)}",
            "model_used": "error"
        }
