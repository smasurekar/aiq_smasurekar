# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in routing probe for the autonomous orchestrator.

This asserts **which route the orchestrator takes on its first turn**, not whether the answer is
correct. It exists because DeepSearchQA cannot validate two of the three behaviors the prompt was
rewritten for: essentially no DSQA item is a greeting, and essentially none is report-shaped, so
planner usage on DSQA should stay near zero even when routing is working correctly. Measuring
planner activation on DSQA is chasing a metric that ought to be ~0. This probe measures it
directly instead.

The source tools are stubbed, so no search traffic and no Tavily key are needed — retrieval
quality is irrelevant here. The orchestrator LLM is real, because the thing under test is what the
model decides. The run is cut off after the first assistant turn, so no researcher subagent, no
planner, and no writer ever executes: one model call per probe item.

Run it::

    export NVIDIA_API_KEY=...            # or: set -a; . deploy/.env; set +a
    AIQ_ROUTING_PROBE=1 uv run pytest \\
        tests/aiq_agent/agents/autonomous_researcher/test_routing_probe.py -v

Useful knobs::

    AIQ_ROUTING_PROBE_MODEL=nvidia/nvidia/nemotron-3-ultra
    AIQ_ROUTING_PROBE_BASE_URL=https://inference-api.nvidia.com/v1
    AIQ_ROUTING_PROBE_TEMPERATURE=0      # default 0; production config uses 0.7

Temperature defaults to 0 rather than the production 0.7 so prompt iteration is not fighting
sampling noise. Routing that only holds at temperature 0 is not routing that will hold in the
eval, so re-run at 0.7 before believing a green board.

These are live-model assertions, so treat a single red item as signal to inspect, not as a build
break. The bucket-level pass rates are the number that matters when iterating on the prompt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from aiq_agent.agents.autonomous_researcher.agent import AutonomousResearcherAgent
from aiq_agent.agents.autonomous_researcher.custom_middleware import AutonomousFinalReportCommitTracker
from aiq_agent.agents.autonomous_researcher.models import AutonomousResearchAgentState
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole

_PROBE_ENABLED = os.getenv("AIQ_ROUTING_PROBE") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _PROBE_ENABLED,
        reason="Set AIQ_ROUTING_PROBE=1 (and NVIDIA_API_KEY) to run the live routing probe.",
    ),
]


# =================================================================================================
# Stub source tools
# =================================================================================================
# Names and descriptions mirror configs/config_autonomous_frag.yml and the real Tavily registration
# (sources/tavily_web_search/src/register.py) because the tool description is rendered into the
# orchestrator prompt and is therefore part of what routes the model. A stub with a different
# description would be probing a different prompt. Only the *body* is fake.


@tool
async def web_search_tool(question: str) -> str:
    """Retrieves relevant contexts from web search (using Tavily) for the given question.

    Args:
        question (str): The question to be answered. Will be truncated to 400 characters if longer.

    Returns:
        str: The web search results containing relevant documents and their URLs.
    """
    return f"[stub] web results for: {question}"


@tool
async def advanced_web_search_tool(question: str) -> str:
    """Retrieves relevant contexts from web search (using Tavily) for the given question.

    Args:
        question (str): The question to be answered. Will be truncated to 400 characters if longer.

    Returns:
        str: The web search results containing relevant documents and their URLs.
    """
    return f"[stub] advanced web results for: {question}"


@tool
async def fetch_url_tool(urls: list[str], query: str | None = None, start_line: int = 0) -> str:
    """Placeholder; the real description is attached below."""
    return f"[stub] page content for: {urls}"


# The fetch tool's description is the routing contract this probe most needs to exercise, so it is
# taken from the live registration rather than retyped. A hand-copied description would drift and
# the probe would then be measuring a prompt that is not shipped.
try:
    from web_page_fetch.register import _DESCRIPTION_FOR_PROBE

    fetch_url_tool.description = _DESCRIPTION_FOR_PROBE
except ImportError:  # pragma: no cover - probe still runs without the package installed
    pass


_STUB_SOURCE_TOOLS = [web_search_tool, advanced_web_search_tool, fetch_url_tool]
_SOURCE_TOOL_NAMES = frozenset(t.name for t in _STUB_SOURCE_TOOLS)


# =================================================================================================
# Probe corpus — 36 items, 6 per bucket
# =================================================================================================
# Buckets are named for the ROUTE they should elicit, not for a category the request belongs to.
# The prompt states three properties (could one agent finish the whole thing; do the unknowns
# depend on each other; is the deliverable's structure part of the ask) and deliberately defines
# no request taxonomy, so a bucket here is an expectation about tool selection, not a label the
# model is asked to produce.
#
# EASY_SINGLE_AGENT used to be ONE_INDEPENDENT_UNKNOWN and used to expect a one-query
# run_research_batch. The shallow-researcher sub-agent is strictly cheaper for exactly this shape
# — one bounded run that returns the finished answer, versus a batch plus a composition turn — so
# the expectation moved with the capability. `high` depth and direct-search assertions are gone
# from that bucket for the same reason: neither is reachable through this route.

NO_RESEARCH = [
    "Hi.",
    "What can you do?",
    "Thanks, that was helpful.",
    "Who are you?",
    "Can you explain what kinds of questions you're able to answer?",
    "Good morning!",
]

EASY_SINGLE_AGENT = [
    "Who is the current CEO of Intel?",
    "What was NVIDIA's data-center revenue in its most recent reported quarter?",
    "What is the current population of Lagos?",
    "When did the James Webb Space Telescope launch?",
    "What is the latest stable release of PostgreSQL?",
    "How tall is the Jeddah Tower as currently built?",
]

SEVERAL_INDEPENDENT_UNKNOWNS = [
    "Compare the current corporate tax rates in Ireland, Singapore, and Estonia.",
    "What are the 2025 flagship GPUs from NVIDIA, AMD, and Intel, and what is each one's memory bandwidth?",
    "Compare the current population of Tokyo, Delhi, and Shanghai.",
    "What are the current central bank interest rates in the US, the UK, and Japan?",
    "How do the launch costs of Falcon 9, Electron, and Ariane 6 compare?",
    "What are the current minimum wages in Germany, France, and Spain?",
]

# The two buckets below split on the SHAPE OF THE ANSWER SET, which is what
# _SHALLOW_SUBAGENT_DESCRIPTION keys on since 2026-08-31. They are the probe for recommendation N2
# (see misc/autonomous_researcher/autonomous-researcher-f1-and-token-recommendations.md 8.3-8.4).
#
# Both buckets stack conditions and both have short answers, so the OLD rule ("narrows a set in
# steps ... two number conditions stacked") would have sent every item in both of them away from
# shallow-researcher. Only the winner-selection half should go. Items are written fresh rather than
# lifted from DeepSearchQA, per the no-worked-examples convention above _SHALLOW_SUBAGENT_DESCRIPTION.

ENUMERATE_SET = [
    "Which EU member states have both a statutory minimum wage above EUR 1,500 per month and "
    "an unemployment rate below 6%?",
    "List every US state that has no individual income tax and a population above 3 million.",
    "Which G20 countries have ratified the High Seas Treaty and also set a net-zero target before 2050?",
    "Give me all the Premier League clubs that have won the competition at least twice and "
    "currently play in a stadium holding more than 40,000.",
    "Which national parks in Canada are both UNESCO World Heritage sites and larger than 10,000 square kilometres?",
    "Name every element in the periodic table that is liquid at 300 K and has an atomic number below 90.",
]

SELECT_ONE_WINNER = [
    "Which EU member state has the highest statutory minimum wage per month?",
    "Of the US states with no individual income tax, which has the largest population?",
    "Which G20 country cut its greenhouse gas emissions the most between 2015 and 2023?",
    "Which Premier League club has the largest stadium capacity?",
    "Which Canadian national park is the largest by area?",
    "Among the world's container ports, which handled the most TEU in the most recent full year?",
]

STRUCTURE_FIXED = [
    (
        "Write me a briefing on solid-state battery commercialization with three sections: "
        "cell chemistry, manufacturing readiness, and the top five players by funding."
    ),
    (
        "Produce a report on the EU AI Act covering, in this order: scope, risk tiers, "
        "obligations for general-purpose models, and the enforcement timeline."
    ),
    (
        "Build me a comparison matrix of Snowflake, Databricks, and BigQuery across pricing model, "
        "governance features, and ML tooling, then summarize the trade-offs."
    ),
    (
        "I need a market overview of the humanoid robotics sector: an executive summary, a section "
        "per major player, a funding table, and a closing outlook."
    ),
    (
        "Draft a technical briefing on post-quantum cryptography migration with sections on the "
        "NIST standards, hardware readiness, and a recommended migration sequence."
    ),
    (
        "Give me a structured report on offshore wind in the North Sea: current capacity, "
        "planned projects, grid constraints, and the main policy blockers, each as its own section."
    ),
]


# =================================================================================================
# Driver
# =================================================================================================


def _build_probe_llm() -> Any:
    """Build the orchestrator model, mirroring config_autonomous_frag.yml's `_type: nim`.

    NAT resolves `_type: nim` to `ChatNVIDIA` (nat.plugins.langchain.llm), so the probe uses the
    same class and the same `chat_template_kwargs` the config sets. Temperature is the deliberate
    exception; see the module docstring.
    """
    pytest.importorskip("langchain_nvidia_ai_endpoints")
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    if not os.getenv("NVIDIA_API_KEY"):
        pytest.skip("NVIDIA_API_KEY is required for the routing probe.")

    return ChatNVIDIA(
        model=os.getenv("AIQ_ROUTING_PROBE_MODEL", "nvidia/nvidia/nemotron-3-ultra"),
        base_url=os.getenv("AIQ_ROUTING_PROBE_BASE_URL", "https://inference-api.nvidia.com/v1"),
        temperature=float(os.getenv("AIQ_ROUTING_PROBE_TEMPERATURE", "0")),
        max_tokens=8192,
        chat_template_kwargs={"enable_thinking": True},
    )


@pytest.fixture(scope="module")
def probe_llm_provider() -> LLMProvider:
    llm = _build_probe_llm()
    provider = LLMProvider()
    provider.set_default(llm)
    for role in (LLMRole.ORCHESTRATOR, LLMRole.PLANNER, LLMRole.RESEARCHER, LLMRole.REPORT_WRITER):
        provider.configure(role, llm)
    return provider


@dataclass(frozen=True)
class FirstTurn:
    """What the orchestrator did on its first assistant turn."""

    tool_calls: list[dict[str, Any]]
    text: str

    @property
    def names(self) -> list[str]:
        return [call.get("name", "") for call in self.tool_calls]

    @property
    def subagent_types(self) -> list[str]:
        return [
            (call.get("args") or {}).get("subagent_type", "") for call in self.tool_calls if call.get("name") == "task"
        ]

    @property
    def batch_queries(self) -> list[dict[str, Any]]:
        for call in self.tool_calls:
            if call.get("name") == "run_research_batch":
                return list((call.get("args") or {}).get("queries") or [])
        return []

    @property
    def direct_searches(self) -> set[str]:
        return _SOURCE_TOOL_NAMES & set(self.names)

    def __str__(self) -> str:
        """Distinguish the two failure modes: answered in prose vs called the wrong tool."""
        if not self.tool_calls:
            return f"NO TOOL CALL — answered in prose: {self.text[:120]!r}"
        return " + ".join(
            name + (f"({sub})" if (sub := (call.get("args") or {}).get("subagent_type")) else "")
            for name, call in zip(self.names, self.tool_calls, strict=True)
        )


async def _first_turn(provider: LLMProvider, query: str) -> FirstTurn:
    """Return what the orchestrator emits on its FIRST assistant turn.

    Streams the real graph so SubAgentMiddleware has injected the `task` tool and the subagent
    descriptions — those descriptions are load-bearing routing text, so calling the model directly
    with only the rendered system prompt would probe an incomplete prompt. The stream is abandoned
    as soon as the first AIMessage lands, which is why no subagent ever runs.

    No tool calls means the model answered in prose. That is a routing failure in its own right:
    the prompt states that a plain text reply finishes nothing, and `AutonomousFinalizationMiddleware`
    then has to spend a corrective turn to recover the run.
    """
    agent = AutonomousResearcherAgent(llm_provider=provider, tools=list(_STUB_SOURCE_TOOLS))
    try:
        state = AutonomousResearchAgentState(messages=[HumanMessage(content=query)])
        runnable = agent._build_orchestrator_agent(state, AutonomousFinalReportCommitTracker()).runnable
        async for update in runnable.astream(state, stream_mode="updates"):
            for node_update in (update or {}).values():
                messages = (node_update or {}).get("messages") if isinstance(node_update, dict) else None
                for message in messages or []:
                    if isinstance(message, AIMessage):
                        return FirstTurn(list(message.tool_calls or []), str(message.content or ""))
        return FirstTurn([], "")
    finally:
        agent.finalize(interrupted=True)


# =================================================================================================
# Assertions
# =================================================================================================


@pytest.mark.parametrize("query", NO_RESEARCH)
async def test_requests_with_nothing_to_find_out_do_not_research(probe_llm_provider, query):
    """Criterion 1: a meta/chit-chat request costs zero research calls and finishes in one turn."""
    turn = await _first_turn(probe_llm_provider, query)
    assert turn.names == ["submit_final_report"], f"{query!r} -> {turn}"
    researched = (turn.tool_calls[0].get("args") or {}).get("researched")
    assert researched in (False, "false", "False"), f"{query!r} -> researched={researched!r}"


@pytest.mark.parametrize("query", EASY_SINGLE_AGENT)
async def test_easy_request_goes_to_the_shallow_researcher(probe_llm_provider, query):
    """Criterion 2: a request one agent can finish is handed over whole, in one call.

    The shallow-researcher exit is auto-finalizing, so this first turn is the entire run when it
    routes correctly. Pairing it with any other call would be a routing failure even if the
    delegation itself is right, hence the exact-match on `names`.
    """
    turn = await _first_turn(probe_llm_provider, query)
    assert not turn.direct_searches, f"{query!r} searched directly -> {turn}"
    assert turn.names == ["task"], f"{query!r} -> {turn}"
    assert turn.subagent_types == ["shallow-researcher"], f"{query!r} -> {turn}"


@pytest.mark.parametrize("query", SEVERAL_INDEPENDENT_UNKNOWNS)
async def test_independent_unknowns_go_out_together(probe_llm_provider, query):
    """Independent unknowns fan out in ONE batch; `high` depth stays rare."""
    turn = await _first_turn(probe_llm_provider, query)
    assert not turn.direct_searches, f"{query!r} searched directly -> {turn}"
    assert turn.names.count("run_research_batch") == 1, f"{query!r} -> {turn}"
    queries = turn.batch_queries
    assert 2 <= len(queries) <= 5, f"{query!r} -> {len(queries)} queries"
    high = sum(q.get("depth") == "high" for q in queries)
    assert high <= 1, f"{query!r} -> {high} high-depth of {len(queries)}"


@pytest.mark.parametrize("query", STRUCTURE_FIXED)
async def test_fixed_structure_plans_before_researching(probe_llm_provider, query):
    """Criterion 3, and the D2 regression: the planner must win these routing decisions.

    Zero planner calls across 90 eval trials is what made writer-agent unreachable, since it is
    gated on /shared/plan.json. This is the assertion that would have caught it.
    """
    turn = await _first_turn(probe_llm_provider, query)
    assert "planner-agent" in turn.subagent_types, f"{query!r} -> {turn}"


@pytest.mark.parametrize("query", ENUMERATE_SET)
async def test_multi_condition_enumerations_go_to_the_shallow_researcher(probe_llm_provider, query):
    """N2, positive half: stacked conditions must NOT push an enumeration off the cheap path.

    These are the items the pre-2026-08-31 wording lost. Pooled over 1,259 Fetch-disabled trials,
    shallow scores 0.646 on this shape against the deep paths' 0.605, at roughly a fifth of the
    tokens - the extra ~30 searches buy nothing because the shape is graded, so partial coverage
    already earns partial credit.
    """
    turn = await _first_turn(probe_llm_provider, query)
    assert not turn.direct_searches, f"{query!r} searched directly -> {turn}"
    assert turn.subagent_types == ["shallow-researcher"], f"{query!r} -> {turn}"


@pytest.mark.parametrize("query", SELECT_ONE_WINNER)
async def test_winner_selection_does_not_go_to_the_shallow_researcher(probe_llm_provider, query):
    """N2, negative half: naming one winner needs every candidate priced first.

    This shape is all-or-nothing - 98% of those trials score exactly 0 or exactly 1 - because a
    winner named from partial coverage is wrong, not incomplete, and the shallow exit publishes it
    as the final answer. Measured 0.325 shallow against 0.515 on the deep paths.

    Deliberately asserts only that shallow is declined. Which fan-out path takes it is not
    something the pooled data separates: `planner` and `no-planner-deep` are statistically
    indistinguishable from each other on F1.
    """
    turn = await _first_turn(probe_llm_provider, query)
    assert "shallow-researcher" not in turn.subagent_types, f"{query!r} -> {turn}"


@pytest.mark.parametrize("query", SEVERAL_INDEPENDENT_UNKNOWNS + STRUCTURE_FIXED)
async def test_shallow_researcher_is_not_used_for_work_that_must_be_split(probe_llm_provider, query):
    """The counterweight to the bucket above: cheapest-first must not become cheapest-always.

    A shallow run cannot fan out or honour a fixed section contract, and its report ends the run —
    so choosing it here does not merely cost quality, it forecloses the correct path entirely.
    """
    turn = await _first_turn(probe_llm_provider, query)
    assert "shallow-researcher" not in turn.subagent_types, f"{query!r} -> {turn}"


SOURCE_NAMED = [
    "According to https://www.iea.nl/sites/default/files/2024-11/ICILS_2023_International_Report_0.pdf, "
    "which countries teach computational thinking as a separate subject in primary school?",
    "Using the USDA NASS 2017 Census of Agriculture, which three states produced the most maple syrup?",
    "In World Bank Open Data, what was Kenya's GDP per capita in 2022?",
    "Per table 2.2 of the ICILS 2023 international report, how many countries made CIL compulsory?",
    "What does https://pmc.ncbi.nlm.nih.gov/articles/PMC9506306/ define as a 'no alcohol' product?",
    "According to the FDIC, how many banks failed in 2023?",
]


@pytest.mark.parametrize(
    "query",
    NO_RESEARCH
    + EASY_SINGLE_AGENT
    + SEVERAL_INDEPENDENT_UNKNOWNS
    + STRUCTURE_FIXED
    + ENUMERATE_SET
    + SELECT_ONE_WINNER,
)
async def test_orchestrator_never_opens_with_a_direct_search(probe_llm_provider, query):
    """The 2-call direct budget is for verifying a researcher's result, so turn 1 never uses it.

    Direct orchestrator searches were 480 calls / 5.33 per trial in the baseline, and their raw
    results are what grew the parent context 3.0x and drove the token overrun.
    """
    turn = await _first_turn(probe_llm_provider, query)
    assert not turn.direct_searches, f"{query!r} -> {turn}"


@pytest.mark.parametrize("query", SOURCE_NAMED)
async def test_named_source_requests_never_open_with_a_bare_search(probe_llm_provider, query):
    """Pattern 1: the failure the page-fetch capability exists to remove.

    On the 16 DSQA questions that named their source the agent scored 0.12 fully-correct against a
    reference agent's 0.81, because it searched *around* the named document instead of opening it
    (jobs/2026-08-20__12-58-09/codex_vs_autonomous_analysis.md, section 4).

    The contract is deliberately narrow. Turn 1 may delegate, and it may open the page directly —
    for a request that supplies its own URL a single bounded fetch is often the whole research task
    — but it must not be a bare keyword search against a source the request already identified.
    """
    turn = await _first_turn(probe_llm_provider, query)
    searched = {"web_search_tool", "advanced_web_search_tool"} & set(turn.names)
    assert not searched, f"{query!r} -> {turn}"
