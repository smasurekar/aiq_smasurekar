# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NAT register function for the adaptive research agent.

One unified research agent (a clone of ``deep_researcher``) that self-selects effort per
request via its orchestrator prompt — collapsing simple queries to a near single-shot inline
path and expanding complex queries to the full planner->research->writer pipeline — with no
upstream classifier and no per-request graph rebuild.
"""

import asyncio
import logging
from typing import Literal
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

from .agent import DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS
from .agent import DEFAULT_MAX_RESEARCH_CONCURRENCY
from .agent import DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE
from .agent import AdaptiveResearcherAgent
from .models import AdaptiveResearchAgentState

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")


class AdaptiveResearchAgentConfig(FunctionBaseConfig, name="adaptive_research_agent"):
    """Configuration for the adaptive research agent."""

    model_config = ConfigDict(extra="forbid")

    orchestrator_llm: LLMRef = Field(..., description="LLM for orchestrator")
    source_router_llm: LLMRef | None = Field(default=None, description="LLM for source-router subagent")
    researcher_llm: LLMRef | None = Field(default=None, description="LLM for researcher")
    planner_llm: LLMRef | None = Field(default=None, description="LLM for planner")
    writer_llm: LLMRef | None = Field(default=None, description="LLM for final writer/synthesis subagent")
    tools: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list,
        description="Explicit tool list. Empty = inherit all from data_source_registry.",
    )
    exclude_tools: list[str] = Field(
        default_factory=list,
        description="Tool names to exclude when inheriting from registry.",
    )
    verbose: bool = Field(default=True)
    domain_catalog_path: str | None = Field(
        default=None,
        description="Optional YAML/JSON domain catalog path for source-router-agent.",
    )
    enable_source_router: bool = Field(
        default=False,
        description=(
            "Enable the advisory source-router-agent. Off by default for the adaptive agent "
            "because the shallow / single-shot effort path skips source routing; the orchestrator "
            "may still choose to route on deep effort when this is enabled."
        ),
    )
    enable_citation_verification: bool = Field(
        default=True,
        description="Verify generated citations against sources captured from configured tools.",
    )
    enabled_tiers: list[Literal["direct", "single_shot", "standard", "deep"]] = Field(
        default_factory=lambda: ["direct", "single_shot", "standard", "deep"],
        min_length=1,
        description=(
            "Effort tiers the adaptive orchestrator may select. Disabled tiers are not described "
            "to the model (Layer-A enforcement) and therefore cannot be chosen. Omit to allow all "
            "four. Presets: [single_shot, deep] (2-tier), [deep] (deep-only), [single_shot] "
            "(shallow-only fast lane). Meta / chit-chat remains a no-research safety path, and "
            "parent-report delta requests always use the citation-safe planned writer pipeline."
        ),
    )
    enforce_tier_tools: bool = Field(
        default=False,
        description=(
            "Layer-B hardening: statically hide heavier tools for disabled top tiers via "
            "ComplexityRouterMiddleware. Off by default; the prompt allow-list is the primary "
            "enforcement. Delegation remains available for mandatory parent-report delta rewrites."
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
        default=DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS,
        ge=1,
        description="Shared maximum concurrent source-tool calls across researcher workers.",
    )
    max_source_tool_batch_size: int = Field(
        default=DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE,
        ge=1,
        description="Maximum concrete inputs accepted by batch-capable source tool wrappers.",
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


@register_function(config_type=AdaptiveResearchAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def adaptive_research_agent(config: AdaptiveResearchAgentConfig, builder: Builder):
    """Adaptive research agent: one graph, model-selected effort per request."""
    skills_config, sandbox_config = resolve_deep_research_runtime_config(config, builder)

    if config.tools:
        tool_refs = config.tools
    else:
        from aiq_agent.common import get_all_tool_refs

        tool_refs = get_all_tool_refs()

    tools = await builder.get_tools(tool_names=tool_refs, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    if config.exclude_tools:
        excluded = set(config.exclude_tools)
        tools = [t for t in tools if getattr(t, "name", "") not in excluded]

    from aiq_agent.common import validate_tool_availability

    is_valid, available_count, unavailable = validate_tool_availability(
        tools,
        research_type="adaptive research",
    )
    if not is_valid:
        logger.warning(
            "Startup check: no tools available for adaptive research. "
            "All queries will fail until at least one tool is properly configured.",
        )

    llm = await builder.get_llm(config.orchestrator_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    provider = LLMProvider()
    provider.set_default(llm)

    provider.configure(LLMRole.ORCHESTRATOR, llm)
    if config.source_router_llm:
        source_router_llm = await builder.get_llm(config.source_router_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        provider.configure(LLMRole.ROUTER, source_router_llm)
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

    agent = AdaptiveResearcherAgent(
        llm_provider=provider,
        tools=tools,
        verbose=verbose,
        callbacks=callbacks,
        domain_catalog_path=config.domain_catalog_path,
        enable_source_router=config.enable_source_router,
        enable_citation_verification=config.enable_citation_verification,
        enabled_tiers=config.enabled_tiers,
        enforce_tier_tools=config.enforce_tier_tools,
        skills=skills_config,
        sandbox=sandbox_config,
        max_research_concurrency=config.max_research_concurrency,
        max_concurrent_source_tool_calls=config.max_concurrent_source_tool_calls,
        max_source_tool_batch_size=config.max_source_tool_batch_size,
    )

    async def _run(state: AdaptiveResearchAgentState) -> AdaptiveResearchAgentState:
        """Run adaptive research with a list of messages or payload."""
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
                active_agent = AdaptiveResearcherAgent(
                    llm_provider=provider,
                    tools=selected_tools,
                    verbose=verbose,
                    callbacks=callbacks,
                    domain_catalog_path=config.domain_catalog_path,
                    enable_source_router=config.enable_source_router,
                    enable_citation_verification=config.enable_citation_verification,
                    enabled_tiers=config.enabled_tiers,
                    enforce_tier_tools=config.enforce_tier_tools,
                    skills=skills_config,
                    sandbox=sandbox_config,
                    job_id=job_id,
                    max_research_concurrency=config.max_research_concurrency,
                    max_concurrent_source_tool_calls=config.max_concurrent_source_tool_calls,
                    max_source_tool_batch_size=config.max_source_tool_batch_size,
                )
                owns_active_agent = True

            if all_mapped_tools_filtered_out(tools, selected_tools, data_sources):
                logger.warning("Adaptive research received data_sources with no matching tools")

            # Validate tool availability before starting. At least one tool must be available so
            # the agent does not reason about unavailable tools. selected_tools already reflects
            # data_sources filtering.
            from aiq_agent.common import format_user_facing_tool_error
            from aiq_agent.common import validate_tool_availability

            is_valid, _, unavailable_tools = validate_tool_availability(
                selected_tools, research_type="adaptive research"
            )

            if not is_valid:
                error_msg = format_user_facing_tool_error("adaptive research", unavailable_tools)

                from langchain_core.messages import AIMessage

                error_state = AdaptiveResearchAgentState(messages=state.messages + [AIMessage(content=error_msg)])
                return error_state

            result = await active_agent.run(state)
            return result
        except asyncio.CancelledError:
            interrupted = True
            raise
        except Exception:
            logger.exception("Error in adaptive research execution")
            raise
        finally:
            if owns_active_agent:
                await asyncio.to_thread(active_agent.finalize, interrupted=interrupted)

    yield FunctionInfo.from_fn(_run, description="Adaptive research agent that self-selects effort per request.")


########################################################
# Adaptive Research Workflow (Wrapper for Evaluation / Deployment)
########################################################
class AdaptiveResearchWorkflowConfig(FunctionBaseConfig, name="adaptive_research_workflow"):
    """Configuration for the adaptive research workflow wrapper.

    Accepts a string query and converts it to messages for the adaptive_research_agent. Use this
    as the top-level workflow to run the unified agent directly with no upstream classifier.
    """

    pass


@register_function(config_type=AdaptiveResearchWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def adaptive_research_workflow(config: AdaptiveResearchWorkflowConfig, builder: Builder):
    """Wrapper workflow that accepts string queries for the adaptive research agent."""
    adaptive_research_agent_fn = await builder.get_function("adaptive_research_agent")
    workflow_id = config.name or config.type

    async def _run(query: str) -> ChatResponse:
        """Run adaptive research on a query string."""
        state = AdaptiveResearchAgentState(messages=[HumanMessage(content=query)])
        result = await adaptive_research_agent_fn.ainvoke(state)
        response_content = result.messages[-1].content
        return _create_chat_response(response_content, response_id="research_response", model=workflow_id)

    yield FunctionInfo.from_fn(_run, description="Adaptive research workflow for evaluation (accepts string query).")
