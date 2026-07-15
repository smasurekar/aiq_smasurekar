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

"""Graph and middleware factory for the adaptive researcher agent.

Almost everything is reused from ``deep_researcher.factory`` — the graph context, tool
groupings, subagent specs, researcher runnable, and the shared middleware stack are imported
verbatim to avoid drift. Only three things diverge for the adaptive POC:

1. The orchestrator holds an extra ``submit_final_report`` tool (the inline finalize signal).
2. The orchestrator prompt is rendered with the enabled effort tiers (Layer-A enforcement).
3. The orchestrator's middleware omits ``SourceRoutingGuardMiddleware`` (which forces
   source-routing before any other tool call and is therefore incompatible with the shallow /
   single-shot path), and optionally appends ``ComplexityRouterMiddleware`` (Layer B, off by
   default).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from deepagents import create_deep_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.tools import BaseTool
from langgraph.store.memory import InMemoryStore

from aiq_agent.agents.deep_researcher.custom_middleware import EmptyContentFixMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import ToolNameSanitizationMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import ToolResultPruningMiddleware
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepAgentsRuntime
from aiq_agent.agents.deep_researcher.factory import FILESYSTEM_TOOL_NAMES
from aiq_agent.agents.deep_researcher.factory import ORCHESTRATOR_AGENT
from aiq_agent.agents.deep_researcher.factory import RESEARCHER_AGENT
from aiq_agent.agents.deep_researcher.factory import DeepResearchGraphContext
from aiq_agent.agents.deep_researcher.factory import DeepResearchMiddlewareSet
from aiq_agent.agents.deep_researcher.factory import DeepResearchToolSet
from aiq_agent.agents.deep_researcher.factory import build_common_middleware
from aiq_agent.agents.deep_researcher.factory import build_deep_research_subagents
from aiq_agent.agents.deep_researcher.factory import build_deep_research_tool_set
from aiq_agent.agents.deep_researcher.factory import build_researcher_runnable
from aiq_agent.agents.deep_researcher.factory import runtime_visibility_middleware
from aiq_agent.agents.deep_researcher.tools.research import build_research_batch_tool
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole

from .custom_middleware import ComplexityRouterMiddleware
from .models import AdaptiveResearchAgentState
from .tiers import enabled_tier_profiles
from .tools.finalize import build_declare_effort_tier_tool
from .tools.finalize import build_submit_final_report_tool

logger = logging.getLogger(__name__)

FINALIZE_TOOL_NAME = "submit_final_report"
DECLARE_TIER_TOOL_NAME = "declare_effort_tier"

# The adaptive tool set is identical to the deep researcher's; alias the builder so agent.py
# reads clearly and future divergence has a single seam.
build_adaptive_research_tool_set = build_deep_research_tool_set


def build_adaptive_orchestrator_middleware(
    *,
    tool_set: DeepResearchToolSet,
    source_registry_middleware: SourceRegistryMiddleware,
    research_batch_tool_name: str = "run_research_batch",
    finalize_tool_name: str = FINALIZE_TOOL_NAME,
) -> list[Any]:
    """Middleware for the adaptive orchestrator.

    Same shape as ``deep_researcher.build_orchestrator_middleware`` with two differences:

    - ``SourceRoutingGuardMiddleware`` is omitted. That guard blocks every orchestrator tool
      call until the source-router writes ``/shared/source_routing.json``; under the adaptive
      design source-routing is advisory/optional (and skipped entirely on shallow paths), so
      the guard would deadlock the single-shot path.
    - ``submit_final_report`` is added to the tool-name sanitizer allowlist.
    """
    valid_tool_names = {tool.name for tool in tool_set.helper_tools}
    valid_tool_names.add(research_batch_tool_name)
    valid_tool_names.add(finalize_tool_name)
    valid_tool_names.add(DECLARE_TIER_TOOL_NAME)
    valid_tool_names.update(FILESYSTEM_TOOL_NAMES)
    return [
        EmptyContentFixMiddleware(),
        ToolNameSanitizationMiddleware(valid_tool_names=sorted(valid_tool_names)),
        ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
        source_registry_middleware,
        ToolResultPruningMiddleware(keep_last_n=10, max_chars=2000),
        ModelRetryMiddleware(max_retries=2, backoff_factor=2.0, initial_delay=1.0),
    ]


def build_adaptive_research_middleware_set(
    *,
    tool_set: DeepResearchToolSet,
    source_registry_middleware: SourceRegistryMiddleware,
    enable_source_router: bool = False,
    artifact_manager: object | None = None,
) -> DeepResearchMiddlewareSet:
    """Build researcher, planner, writer, and (guard-free) orchestrator middleware stacks."""

    def common(extra_valid_tool_names: Sequence[str] = ()) -> list[Any]:
        return build_common_middleware(
            tool_set=tool_set,
            source_registry_middleware=source_registry_middleware,
            artifact_manager=artifact_manager,
            extra_valid_tool_names=extra_valid_tool_names,
        )

    return DeepResearchMiddlewareSet(
        researcher=common(),
        planner=common(),
        writer=common(),
        orchestrator=build_adaptive_orchestrator_middleware(
            tool_set=tool_set,
            source_registry_middleware=source_registry_middleware,
        ),
    )


def build_adaptive_research_graph(
    *,
    llm_provider: LLMProvider,
    state: AdaptiveResearchAgentState,
    prompts: dict[str, str],
    tools: Sequence[BaseTool],
    runtime: DeepAgentsRuntime,
    tool_set: DeepResearchToolSet,
    middleware_set: DeepResearchMiddlewareSet,
    source_registry_middleware: SourceRegistryMiddleware,
    callbacks: list[Any],
    domain_catalog_path: str | None,
    max_research_concurrency: int,
    enabled_tiers: list[str],
    enforce_tier_tools: bool = False,
    enable_source_router: bool = False,
) -> Any:
    """Build the full DeepAgents graph for one adaptive research run.

    Mirrors ``deep_researcher.build_deep_research_graph`` with the adaptive divergences: the
    orchestrator carries ``submit_final_report``; its prompt is rendered with the enabled effort
    tiers; and (only when ``enforce_tier_tools``) ``ComplexityRouterMiddleware`` is appended.
    """
    from aiq_agent.agents.deep_researcher.custom_middleware import ExecuteTimeoutClampMiddleware
    from aiq_agent.agents.deep_researcher.custom_middleware import FilesystemToolCallGuardMiddleware

    cross_cutting_middleware = [
        FilesystemToolCallGuardMiddleware(),
        *runtime_visibility_middleware(runtime),
    ]
    execute_ceiling = runtime.execute_timeout_seconds
    if execute_ceiling:
        cross_cutting_middleware = [
            ExecuteTimeoutClampMiddleware(max_timeout_seconds=execute_ceiling),
            *cross_cutting_middleware,
        ]

    context = DeepResearchGraphContext(
        llm_provider=llm_provider,
        state=state,
        prompts=prompts,
        tools=tools,
        runtime=runtime,
        tool_set=tool_set,
        middleware_set=middleware_set,
        domain_catalog_path=domain_catalog_path,
        current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        max_research_concurrency=max_research_concurrency,
        enable_source_router=enable_source_router,
        backend=runtime.backend,
        visibility_middleware=cross_cutting_middleware,
    )

    researcher_runnable = build_researcher_runnable(
        researcher_model=context.llm_provider.get(LLMRole.RESEARCHER),
        researcher_tools=context.tool_set.researcher_tools,
        system_prompt=context.render_prompt(
            "researcher",
            tools=context.tool_set.tools_info,
            execution_enabled=context.runtime.execution_enabled,
        ),
        researcher_middleware=context.middleware_set.researcher,
        skill_sources=context.skill_sources(RESEARCHER_AGENT),
        backend=context.backend,
        visibility_middleware=context.visibility_middleware,
        filesystem_permissions=context.permissions(RESEARCHER_AGENT),
    )
    research_batch_tool = build_research_batch_tool(
        researcher_runnable=researcher_runnable,
        backend=context.backend,
        callbacks=callbacks,
        max_research_concurrency=max_research_concurrency,
        source_registry_middleware=source_registry_middleware,
    )
    declare_effort_tier_tool = build_declare_effort_tier_tool(backend=context.backend)
    submit_final_report_tool = build_submit_final_report_tool(backend=context.backend)

    orchestrator_tools = [
        *context.tool_set.helper_tools,
        research_batch_tool,
        declare_effort_tier_tool,
        submit_final_report_tool,
    ]

    orchestrator_middleware = context.middleware(context.middleware_set.orchestrator)
    if enforce_tier_tools:
        orchestrator_middleware = [
            *orchestrator_middleware,
            ComplexityRouterMiddleware(
                enabled_tiers=enabled_tiers,
                # Delta rewrites must retain planner/writer delegation even under a shallow-only
                # normal-effort preset so preserved parent citations remain valid.
                allow_delegation=context.parent_report_context_available,
            ),
        ]

    agent = create_deep_agent(
        model=context.llm_provider.get(LLMRole.ORCHESTRATOR),
        tools=orchestrator_tools,
        system_prompt=context.render_prompt(
            "orchestrator",
            clarifier_result=context.state.clarifier_result,
            # Advertise only the tools the orchestrator can actually call. Source tools live on
            # the researcher and are reached via run_research_batch; listing them here would make
            # the orchestrator call them directly, which the runtime rejects.
            tools=[{"name": t.name, "description": t.description} for t in orchestrator_tools],
            # Retrieval tools are NOT callable by the orchestrator (the researcher holds them),
            # but the shallow/standard inline paths must name them in ResearchQuery.preferred_tools,
            # so surface their names/descriptions. Per-request (varies with data_sources), hence
            # rendered below the KV-cache boundary.
            retrieval_tools=context.tool_set.tools_info,
            enable_source_router=context.enable_source_router,
            max_research_concurrency=context.max_research_concurrency,
            execution_enabled=context.runtime.execution_enabled,
            parent_report_context_available=context.parent_report_context_available,
            enabled_tiers=enabled_tiers,
            tier_profiles=enabled_tier_profiles(enabled_tiers),
            triage_hint="",
        ),
        subagents=build_deep_research_subagents(context),
        store=InMemoryStore(),
        middleware=orchestrator_middleware,
        permissions=context.permissions(ORCHESTRATOR_AGENT),
        backend=context.backend,
    )
    return agent.with_config({"recursion_limit": 2000})
