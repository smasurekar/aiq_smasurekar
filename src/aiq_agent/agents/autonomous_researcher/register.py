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

"""NAT register function for the autonomous research agent.

A description-driven sibling to ``deep_research_agent`` and ``adaptive_research_agent``. Where
the adaptive agent classifies each request into one of four effort tiers and drives four separate
machines from that label, this agent has no classification step at all: one system prompt, richly
described tools, richly described subagents, and an orchestrator that chooses depth as an ordinary
reasoning step.

The config surface reflects that. Relative to ``AdaptiveResearchAgentConfig`` it **drops** every
tier knob — ``enabled_tiers``, ``enforce_tier_tools``, ``single_loop_single_shot``,
``dynamic_orchestrator_sections``, ``single_shot_search_budget``, ``single_shot_shallow_subagent``,
``shallow_subagent_max_*``, and the already-dead ``single_shot_researcher_llm`` — plus the
source-router knobs (``source_router_llm``, ``enable_source_router``, ``domain_catalog_path``),
whose advisory routing is exactly what rich tool descriptions substitute for.

Adding a capability here means writing a description, not editing a table.
"""

import asyncio
import logging
from typing import TypeVar

from langchain_core.messages import HumanMessage
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSandboxConfig
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSkillsConfig
from aiq_agent.agents.deep_researcher.register import resolve_deep_research_runtime_config
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole
from aiq_agent.common import VerboseTraceCallback
from aiq_agent.common import _create_chat_response
from aiq_agent.common import all_mapped_tools_filtered_out
from aiq_agent.common import filter_tools_by_sources
from aiq_agent.common import is_verbose
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatResponse
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from .agent import DEFAULT_MAX_RESEARCH_CONCURRENCY
from .agent import AutonomousResearcherAgent
from .factory import DEFAULT_SHALLOW_SUBAGENT_MAX_LLM_TURNS
from .factory import DEFAULT_SHALLOW_SUBAGENT_MAX_TOOL_ITERATIONS
from .models import AutonomousRequestTerminationConfig
from .models import AutonomousResearchAgentState
from .models import ResearcherLoopGuardConfig

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")

# Source tools that run their own multi-source research and return a finished, synthesized cited
# answer. Handing one to a research orchestrator invites it to delegate the whole task and pass
# the result through — bypassing run_research_batch, the loop guards, and AI-Q's citation registry
# (their citations never enter get_verified_sources, so the finalizer would strip them). The tier
# design had middleware that could catch a bad delegation; this one deliberately does not, so the
# guard has to be a config decision. Operators can override by setting exclude_tools explicitly.
DEFAULT_EXCLUDED_SYNTHESIZING_TOOLS: tuple[str, ...] = ("you_research", "you_finance_research")


class AutonomousResearchAgentConfig(FunctionBaseConfig, name="autonomous_research_agent"):
    """Configuration for the autonomous research agent."""

    model_config = ConfigDict(extra="forbid")

    orchestrator_llm: LLMRef = Field(..., description="LLM for the orchestrator")
    researcher_llm: LLMRef | None = Field(default=None, description="LLM for the researcher")
    planner_llm: LLMRef | None = Field(default=None, description="LLM for the planner subagent")
    writer_llm: LLMRef | None = Field(default=None, description="LLM for the final writer/synthesis subagent")
    tools: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list,
        description="Explicit tool list. Empty = inherit all from data_source_registry.",
    )
    exclude_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names to exclude. Applied on top of the always-on exclusion of synthesizing "
            "research APIs (you_research, you_finance_research), which return their own cited "
            "answers and would bypass the citation registry."
        ),
    )
    verbose: bool = Field(default=True)
    enable_citation_verification: bool = Field(
        default=True,
        description="Verify generated citations against sources captured from configured tools.",
    )
    researcher_loop_guard: ResearcherLoopGuardConfig = Field(
        default_factory=ResearcherLoopGuardConfig,
        description=(
            "Hard per-researcher limits for source-tool calls, repeated identical requests, and "
            "uninterrupted think loops. Applies to run_research_batch workers, which are seeded "
            "with each query's depth."
        ),
    )
    request_termination: AutonomousRequestTerminationConfig = Field(
        default_factory=AutonomousRequestTerminationConfig,
        description=(
            "Request-wide termination envelope: one flat set of finite budgets for "
            "run_research_batch calls, total delegated queries, repeated queries, and "
            "orchestrator turns, plus a hard workflow deadline and graph recursion ceiling. "
            "Bounds the whole top-level request so it always reaches a terminal state — unlike "
            "the per-researcher loop guard, whose budget resets for every new invocation."
        ),
    )
    skills: DeepResearchSkillsConfig | FunctionRef | None = Field(
        default=None,
        description="Optional inline skills config or function ref to a deep_research_skills config.",
    )
    sandbox: DeepResearchSandboxConfig | FunctionRef | None = Field(
        default=None,
        description="Optional inline sandbox config or function ref to a deep_research_sandbox config.",
    )
    max_research_concurrency: int = Field(
        default=DEFAULT_MAX_RESEARCH_CONCURRENCY,
        ge=1,
        description="Maximum ResearchQuery items accepted and run concurrently per run_research_batch call.",
    )
    max_concurrent_source_tool_calls: int = Field(
        default=8,
        ge=1,
        description="Shared maximum concurrent source-tool calls across researcher workers.",
    )
    max_source_tool_batch_size: int = Field(
        default=8,
        ge=1,
        description="Maximum concrete inputs accepted by batch-capable source tool wrappers.",
    )
    # Deliberately named `shallow_subagent`, not the adaptive arm's `single_shot_shallow_subagent`:
    # there is no effort tier to qualify it with here. Default-on because a request one agent can
    # finish is the common case, and answering it through the full research cycle is pure overhead.
    shallow_subagent: bool = Field(
        default=True,
        description=(
            "Offer the shallow-researcher sub-agent, which answers an easy request end to end and "
            "whose report finishes the run with no further orchestrator turn. Automatically "
            "suppressed for parent-report deltas."
        ),
    )
    shallow_subagent_max_llm_turns: int = Field(
        default=DEFAULT_SHALLOW_SUBAGENT_MAX_LLM_TURNS,
        ge=1,
        description="Maximum LLM turns inside one shallow-researcher sub-run.",
    )
    shallow_subagent_max_tool_iterations: int = Field(
        default=DEFAULT_SHALLOW_SUBAGENT_MAX_TOOL_ITERATIONS,
        ge=1,
        description="Maximum tool-calling iterations inside one shallow-researcher sub-run.",
    )
    # Retrieval narrowing for the sub-run only. `tools` / `exclude_tools` above are global: they
    # decide what the whole agent can reach, so they cannot express "the orchestrator keeps both
    # web tools but the shallow sub-run only gets the wide one". These two do exactly that, and
    # leave the orchestrator's and researcher subagents' tool sets untouched.
    #
    # This matters because the shallow sub-run is a single bounded pass whose report can end the
    # request outright — the evidence it gathers in ~5 calls IS the answer, with no later turn to
    # widen it. When a narrow and a wide retrieval tool are both on offer the model reliably picks
    # the narrow one (measured: 204 of 212 shallow searches went to a 2-result tool over a
    # 5-result one), so pinning the sub-run to the wide tool is worth more here than anywhere else.
    shallow_subagent_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names the shallow-researcher sub-agent may use. Empty (default) inherits the "
            "agent's full tool set. Applied after the request's data_sources filter and after "
            "exclude_tools, and only to the sub-run."
        ),
    )
    shallow_subagent_exclude_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names withheld from the shallow-researcher sub-agent only. Applied after "
            "shallow_subagent_tools. The orchestrator and the other subagents keep them."
        ),
    )

    @field_validator("skills", mode="before")
    @classmethod
    def _parse_inline_skills(cls, value):
        if isinstance(value, dict):
            return DeepResearchSkillsConfig.model_validate(value)
        return value

    @field_validator("sandbox", mode="before")
    @classmethod
    def _parse_inline_sandbox(cls, value):
        if isinstance(value, dict):
            return DeepResearchSandboxConfig.model_validate(value)
        return value


@register_function(config_type=AutonomousResearchAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def autonomous_research_agent(config: AutonomousResearchAgentConfig, builder: Builder):
    """Autonomous research agent: one graph, emergent effort, description-driven routing."""
    skills_config, sandbox_config = resolve_deep_research_runtime_config(config, builder)

    if config.tools:
        tool_refs = config.tools
    else:
        from aiq_agent.common import get_all_tool_refs

        tool_refs = get_all_tool_refs()

    tools = await builder.get_tools(tool_names=tool_refs, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    excluded = {*config.exclude_tools, *DEFAULT_EXCLUDED_SYNTHESIZING_TOOLS}
    dropped = sorted(name for name in excluded if any(getattr(t, "name", "") == name for t in tools))
    tools = [t for t in tools if getattr(t, "name", "") not in excluded]
    if dropped:
        logger.info("Autonomous research excluded %d configured tool(s): %s", len(dropped), ", ".join(dropped))

    # Fail fast on a misspelt shallow tool name. Silently ignoring one would leave the sub-run on
    # the full tool set while the config claims it is pinned - the exact situation these knobs
    # exist to rule out. Checked against the post-exclude_tools names, which is what the sub-run
    # can actually be given.
    configured_names = {getattr(t, "name", "") for t in tools}
    for field_name, requested in (
        ("shallow_subagent_tools", config.shallow_subagent_tools),
        ("shallow_subagent_exclude_tools", config.shallow_subagent_exclude_tools),
    ):
        unknown = sorted(set(requested) - configured_names)
        if unknown:
            raise ValueError(
                f"{field_name} names tool(s) this agent does not have: {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(n for n in configured_names if n)) or '(none)'}"
            )

    from aiq_agent.common import validate_tool_availability

    is_valid, available_count, unavailable = validate_tool_availability(
        tools,
        research_type="autonomous research",
    )
    if not is_valid:
        logger.warning(
            "Startup check: no tools available for autonomous research. "
            "All queries will fail until at least one tool is properly configured.",
        )

    llm = await builder.get_llm(config.orchestrator_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    provider = LLMProvider()
    provider.set_default(llm)
    provider.configure(LLMRole.ORCHESTRATOR, llm)
    if config.researcher_llm:
        researcher_llm = await builder.get_llm(config.researcher_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        provider.configure(LLMRole.RESEARCHER, researcher_llm)
    if config.planner_llm:
        planner_llm = await builder.get_llm(config.planner_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        provider.configure(LLMRole.PLANNER, planner_llm)
    if config.writer_llm:
        writer_llm = await builder.get_llm(config.writer_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        provider.configure(LLMRole.REPORT_WRITER, writer_llm)

    verbose = is_verbose(config.verbose)
    callbacks = [VerboseTraceCallback()] if verbose else []

    def _build_agent(agent_tools, *, job_id: str | None = None) -> AutonomousResearcherAgent:
        """Construct an agent over a tool set. One seam so the two call sites cannot diverge."""
        return AutonomousResearcherAgent(
            llm_provider=provider,
            tools=agent_tools,
            verbose=verbose,
            callbacks=callbacks,
            enable_citation_verification=config.enable_citation_verification,
            researcher_loop_guard=config.researcher_loop_guard,
            request_termination=config.request_termination,
            skills=skills_config,
            sandbox=sandbox_config,
            job_id=job_id,
            max_research_concurrency=config.max_research_concurrency,
            max_concurrent_source_tool_calls=config.max_concurrent_source_tool_calls,
            max_source_tool_batch_size=config.max_source_tool_batch_size,
            shallow_subagent=config.shallow_subagent,
            shallow_subagent_max_llm_turns=config.shallow_subagent_max_llm_turns,
            shallow_subagent_max_tool_iterations=config.shallow_subagent_max_tool_iterations,
            shallow_subagent_tools=config.shallow_subagent_tools,
            shallow_subagent_exclude_tools=config.shallow_subagent_exclude_tools,
        )

    agent = _build_agent(tools)

    async def _run(state: AutonomousResearchAgentState) -> AutonomousResearchAgentState:
        """Run autonomous research with a list of messages or payload."""
        active_agent = agent
        owns_active_agent = False
        interrupted = False
        try:
            data_sources = state.data_sources
            selected_tools = filter_tools_by_sources(tools, data_sources)
            if sandbox_config is not None or (data_sources is not None and selected_tools != tools):
                # Scope the Modal sandbox to the async job_id when one is in NAT context. Falls
                # back to a per-request uuid in DeepAgentsRuntime when None.
                job_id: str | None = None
                try:
                    from nat.builder.context import Context

                    job_id = Context.get().workflow_run_id
                except Exception:  # noqa: BLE001 - Context may be unavailable in sync/eval paths
                    job_id = None
                active_agent = _build_agent(selected_tools, job_id=job_id)
                owns_active_agent = True

            if all_mapped_tools_filtered_out(tools, selected_tools, data_sources):
                logger.warning("Autonomous research received data_sources with no matching tools")

            # Validate tool availability before starting. At least one tool must be available so
            # the agent does not reason about unavailable tools. selected_tools already reflects
            # data_sources filtering.
            from aiq_agent.common import format_user_facing_tool_error
            from aiq_agent.common import validate_tool_availability

            is_valid, _, unavailable_tools = validate_tool_availability(
                selected_tools, research_type="autonomous research"
            )

            if not is_valid:
                error_msg = format_user_facing_tool_error("autonomous research", unavailable_tools)

                from langchain_core.messages import AIMessage

                return AutonomousResearchAgentState(messages=state.messages + [AIMessage(content=error_msg)])

            return await active_agent.run(state)
        except asyncio.CancelledError:
            interrupted = True
            raise
        except Exception:
            logger.exception("Error in autonomous research execution")
            raise
        finally:
            if owns_active_agent:
                await asyncio.to_thread(active_agent.finalize, interrupted=interrupted)

    yield FunctionInfo.from_fn(
        _run,
        description="Autonomous research agent that chooses its own research depth per request.",
    )


########################################################
# Autonomous Research Workflow (Wrapper for Evaluation / Deployment)
########################################################
class AutonomousResearchWorkflowConfig(FunctionBaseConfig, name="autonomous_research_workflow"):
    """Configuration for the autonomous research workflow wrapper.

    Accepts a string query and converts it to messages for the autonomous_research_agent. Use this
    as the top-level workflow to run the agent directly with no upstream classifier. Response
    shape matches the adaptive and deep workflows, so the eval harnesses run against it unchanged.
    """


@register_function(config_type=AutonomousResearchWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def autonomous_research_workflow(config: AutonomousResearchWorkflowConfig, builder: Builder):
    """Wrapper workflow that accepts string queries for the autonomous research agent."""
    autonomous_research_agent_fn = await builder.get_function("autonomous_research_agent")
    workflow_id = config.name or config.type

    async def _run(query: str) -> ChatResponse:
        """Run autonomous research on a query string."""
        state = AutonomousResearchAgentState(messages=[HumanMessage(content=query)])
        result = await autonomous_research_agent_fn.ainvoke(state)
        response_content = result.messages[-1].content
        return _create_chat_response(response_content, response_id="research_response", model=workflow_id)

    yield FunctionInfo.from_fn(_run, description="Autonomous research workflow for evaluation (accepts string query).")
