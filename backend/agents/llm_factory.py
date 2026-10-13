"""
LLM Factory — shared Groq LLM builder for all graph nodes.

Provides:
  get_llm(temperature)       → primary model (llama-3.3-70b-versatile)
  get_fallback_llm()         → fallback model (llama-3.1-8b-instant)
  get_llm_with_fallback()    → tries primary, returns fallback on error

All nodes import from here so model names and settings are in one place.
"""

from langchain_groq import ChatGroq
from backend.config import settings


def get_llm(temperature: float = 0.1) -> ChatGroq:
    """Return the primary Groq LLM."""
    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=settings.PRIMARY_MODEL,
        temperature=temperature,
    )


def get_fallback_llm(temperature: float = 0.1) -> ChatGroq:
    """Return the fallback Groq LLM (smaller, faster model)."""
    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=settings.FALLBACK_MODEL,
        temperature=temperature,
    )


async def ainvoke_with_fallback(llm: ChatGroq, messages: list, tools: list = None):
    """
    Invoke an LLM with automatic fallback to the smaller model on failure.

    Returns: (response, model_name_used)
    """
    bound = llm.bind_tools(tools) if tools else llm
    try:
        response = await bound.ainvoke(messages)
        return response, settings.PRIMARY_MODEL
    except Exception:
        fallback = get_fallback_llm(temperature=llm.temperature)
        fallback_bound = fallback.bind_tools(tools) if tools else fallback
        response = await fallback_bound.ainvoke(messages)
        return response, settings.FALLBACK_MODEL
