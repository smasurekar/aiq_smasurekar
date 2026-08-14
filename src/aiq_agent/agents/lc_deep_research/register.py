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

"""NAT registration for the LangChain DeepAgents deep-research example.

A third research arm alongside ``adaptive_research_agent`` and ``autonomous_research_agent``, added
so the upstream LangChain deep-research design can be A/B'd against AI-Q's own agents on the same
model, the same harness, and the same dataset.

Unlike its two siblings, this agent is a **reference implementation, not a product surface**. The
config schema is therefore deliberately thin: one model ref (upstream passes a single ``model`` to
``create_deep_agent`` and the sub-agent inherits it), the two upstream delegation limits, and two
execution backstops. There is no tool list, no source router, no citation verification, no loop
guard, and no tier table -- adding any of them would change what this arm measures.

Notably absent: ``tools``. Upstream's search path is its own ``tavily_search``, which calls the
Tavily API directly via ``TavilyClient``. It does **not** route through AI-Q's ``tavily_web_search``
NAT function, so the ``data_source_registry`` block in the config is inert for this agent (it is
kept for UI parity and eval-harness preflight -- see ``configs/config_lc_deep_research_frag.yml``).
"""

import logging

from langchain_core.messages import HumanMessage
from pydantic import ConfigDict
from pydantic import Field

from aiq_agent.common import LLMProvider
from aiq_agent.common import VerboseTraceCallback
from aiq_agent.common import _create_chat_response
from aiq_agent.common import is_verbose
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatResponse
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from .agent import DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS
from .agent import DEFAULT_MAX_RESEARCHER_ITERATIONS
from .agent import DEFAULT_RECURSION_LIMIT
from .agent import LcDeepResearchAgent
from .agent import LcDeepResearchAgentState

logger = logging.getLogger(__name__)


class LcDeepResearchAgentConfig(FunctionBaseConfig, name="lc_deep_research_agent"):
    """Configuration for the LangChain DeepAgents deep-research example agent."""

    model_config = ConfigDict(extra="forbid")

    llm: LLMRef = Field(
        ...,
        description=(
            "LLM for both the orchestrator and the research sub-agent. Upstream passes a single "
            "model to create_deep_agent and the sub-agent inherits it; per-role refs would be an "
            "AI-Q invention with no upstream analogue."
        ),
    )
    verbose: bool = Field(default=True, description="Attach VerboseTraceCallback for step-level tracing.")
    max_concurrent_research_units: int = Field(
        default=DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS,
        ge=1,
        description=(
            "Parallel research sub-agents allowed per delegation round. Rendered into the "
            "delegation prompt and enforced by orchestrator compliance, not by middleware -- "
            "exactly as upstream intends. Default is upstream's value."
        ),
    )
    max_researcher_iterations: int = Field(
        default=DEFAULT_MAX_RESEARCHER_ITERATIONS,
        ge=1,
        description=(
            "Delegation rounds the orchestrator may run before it must stop. Prompt-rendered, not "
            "middleware-enforced. Default is upstream's value."
        ),
    )
    recursion_limit: int = Field(
        default=DEFAULT_RECURSION_LIMIT,
        ge=1,
        description=(
            "LangGraph recursion ceiling for one request. Purely a crash backstop: LangGraph's "
            "default of 25 is too low for a multi-round, multi-sub-agent run and would abort "
            "otherwise-healthy requests."
        ),
    )
    workflow_timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional hard wall-clock deadline for one request. Defaults to None (no deadline), "
            "which is faithful to upstream; the eval harness applies its own per-trial timeout."
        ),
    )


@register_function(config_type=LcDeepResearchAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def lc_deep_research_agent(config: LcDeepResearchAgentConfig, builder: Builder):
    """LangChain DeepAgents deep-research example, wired to a NAT-provided model."""
    model = await builder.get_llm(config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    verbose = is_verbose(config.verbose)
    callbacks = [VerboseTraceCallback()] if verbose else []

    # One model for every role, as upstream does. The provider indirection exists only because it
    # is the shape the async-job runner builds and hands to LcDeepResearchAgent; going through it
    # here too means both entry paths construct the agent identically.
    provider = LLMProvider()
    provider.set_default(model)

    agent = LcDeepResearchAgent(
        llm_provider=provider,
        verbose=verbose,
        callbacks=callbacks,
        config=config,
    )

    async def _run(state: LcDeepResearchAgentState) -> LcDeepResearchAgentState:
        """Run one deep-research request and return the state carrying the final report."""
        return await agent.run(state)

    yield FunctionInfo.from_fn(
        _run,
        description=(
            "LangChain DeepAgents deep-research reference agent (upstream prompts and tools, NAT-provided LLM)."
        ),
    )


########################################################
# LC Deep Research Workflow (Wrapper for Evaluation / Deployment)
########################################################
class LcDeepResearchWorkflowConfig(FunctionBaseConfig, name="lc_deep_research_workflow"):
    """Configuration for the LC deep-research workflow wrapper.

    Accepts a string query and converts it to messages for ``lc_deep_research_agent``. Use this as
    the top-level workflow to run the agent directly with no upstream classifier. The response
    shape matches the adaptive and autonomous workflows, so the eval harnesses run against it
    unchanged.
    """


@register_function(config_type=LcDeepResearchWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def lc_deep_research_workflow(config: LcDeepResearchWorkflowConfig, builder: Builder):
    """Wrapper workflow that accepts string queries for the LC deep-research agent."""
    lc_deep_research_agent_fn = await builder.get_function("lc_deep_research_agent")
    workflow_id = config.name or config.type

    async def _run(query: str) -> ChatResponse:
        """Run LC deep research on a query string."""
        state = LcDeepResearchAgentState(messages=[HumanMessage(content=query)])
        result = await lc_deep_research_agent_fn.ainvoke(state)
        response_content = result.messages[-1].content
        return _create_chat_response(response_content, response_id="research_response", model=workflow_id)

    yield FunctionInfo.from_fn(_run, description="LC deep research workflow for evaluation (accepts string query).")
