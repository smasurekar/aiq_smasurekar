"""Deep Research Agent Example.

This module demonstrates building a research agent using the deepagents package
with custom tools for web search and strategic thinking.
"""
# Ported from the LangChain DeepAgents deep_research example
# (deepagents/examples/deep_research/research_agent/__init__.py). Upstream uses absolute
# `research_agent.*` imports because it ships as a top-level package; nested under aiq_agent those
# become relative. That prefix is the only change.

from .prompts import RESEARCH_WORKFLOW_INSTRUCTIONS
from .prompts import RESEARCHER_INSTRUCTIONS
from .prompts import SUBAGENT_DELEGATION_INSTRUCTIONS
from .tools import tavily_search
from .tools import think_tool

__all__ = [
    "RESEARCHER_INSTRUCTIONS",
    "RESEARCH_WORKFLOW_INSTRUCTIONS",
    "SUBAGENT_DELEGATION_INSTRUCTIONS",
    "tavily_search",
    "think_tool",
]
