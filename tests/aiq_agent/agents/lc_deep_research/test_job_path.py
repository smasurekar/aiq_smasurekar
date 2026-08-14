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

"""The async-job path: `/jobs` submissions and the UI reach agents through aiq_api, not NAT.

``aiq_api.jobs.runner`` does not resolve the registered NAT function. It looks the agent up in
``aiq_api.registry.AGENT_REGISTRY``, imports the class named there, picks a constructor pattern by
inspecting the signature, discovers a state class by *name*, and calls ``run(state)``. Every one of
those is a naming or signature contract that a rename would silently break -- an agent that works
perfectly under `nat run` can still fail with "Unknown agent type" or an AttributeError here.
These tests pin each link in that chain.
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from aiq_agent.agents.lc_deep_research import agent as lc_agent
from aiq_agent.agents.lc_deep_research.agent import LcDeepResearchAgent
from aiq_agent.agents.lc_deep_research.agent import LcDeepResearchAgentState
from aiq_agent.agents.lc_deep_research.register import LcDeepResearchAgentConfig
from aiq_agent.common import LLMProvider

AGENT_TYPE = "lc_deep_research"


@pytest.fixture
def provider():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    llm_provider = LLMProvider()
    llm_provider.set_default(GenericFakeChatModel(messages=iter([AIMessage(content="ok")])))
    return llm_provider


@pytest.fixture
def stub_graph(monkeypatch):
    """Replace the compiled graph so tests exercise run() without a model call."""
    captured = {}

    class _Graph:
        async def ainvoke(self, graph_input, config=None):
            captured["input"] = graph_input
            captured["config"] = config
            return {"files": {"/final_report.md": "# Report"}, "messages": [AIMessage(content="ack")]}

    monkeypatch.setattr(lc_agent, "build_lc_deep_research_graph", lambda *a, **k: _Graph())
    return captured


class TestRegistryWiring:
    """Without an AGENT_REGISTRY entry the API rejects the agent before any code runs."""

    def test_agent_type_is_registered(self):
        from aiq_api.registry import get_agent_config

        config = get_agent_config(AGENT_TYPE)
        assert config.class_path == "aiq_agent.agents.lc_deep_research.agent.LcDeepResearchAgent"
        assert config.config_name == "lc_deep_research_agent"
        assert config.public is True

    def test_class_path_actually_imports(self):
        """The registry stores a string; a typo or a moved class only fails at job runtime."""
        from aiq_api.jobs.runner import _load_agent_class
        from aiq_api.registry import get_agent_config

        assert _load_agent_class(get_agent_config(AGENT_TYPE).class_path) is LcDeepResearchAgent

    def test_config_name_matches_the_nat_function_name(self):
        """The runner calls builder.get_function_config(config_name) against the loaded YAML."""
        from aiq_api.registry import get_agent_config

        assert get_agent_config(AGENT_TYPE).config_name == LcDeepResearchAgentConfig.static_type()


class TestRunnerContract:
    """Signature- and name-based discovery in aiq_api.jobs.runner."""

    def test_constructor_matches_the_generic_config_pattern(self):
        """Declaring both `config` and `job_id` is what selects the pattern that passes fn_config."""
        from aiq_api.jobs.runner import _CONFIGURABLE_AGENT_KWARGS
        from aiq_api.jobs.runner import _constructor_accepts_explicit_kwargs

        assert _constructor_accepts_explicit_kwargs(LcDeepResearchAgent, _CONFIGURABLE_AGENT_KWARGS)

    def test_runner_constructs_the_agent_with_config_values(self, provider):
        from aiq_api.jobs.runner import _create_agent_instance

        config = LcDeepResearchAgentConfig(llm="some_llm", recursion_limit=42, max_researcher_iterations=2)
        agent = _create_agent_instance(
            agent_cls=LcDeepResearchAgent,
            llm_provider=provider,
            llm=None,
            tools=[],
            fn_config=config,
            verbose=False,
            callbacks=[],
            job_id="job-1",
        )

        assert isinstance(agent, LcDeepResearchAgent)
        assert agent.recursion_limit == 42
        assert agent.max_researcher_iterations == 2
        assert agent.job_id == "job-1"

    def test_state_class_is_discoverable_by_name(self, provider):
        """`_get_agent_state_class` looks for `<AgentName>State` in the agent's own module."""
        from aiq_api.jobs.runner import _get_agent_state_class

        agent = LcDeepResearchAgent(llm_provider=provider, config=None)
        assert _get_agent_state_class(agent) is LcDeepResearchAgentState

    def test_run_takes_a_state_not_a_string(self, provider):
        """The runner branches on the first parameter name; 'input_text'/'query' means string mode."""
        import inspect

        agent = LcDeepResearchAgent(llm_provider=provider, config=None)
        first_param = list(inspect.signature(agent.run).parameters)[0]
        assert first_param not in ("input_text", "query", "input")

    def test_result_is_readable_by_extract_result(self, provider, stub_graph):
        """`_extract_result` reads state.messages[-1].content -- that must be the report."""
        from aiq_api.jobs.runner import _extract_result

        agent = LcDeepResearchAgent(llm_provider=provider, config=None)
        result = asyncio.run(agent.run(LcDeepResearchAgentState(messages=[HumanMessage(content="q")])))

        assert _extract_result(result) == "# Report"

    def test_runner_tolerates_a_config_without_a_tools_field(self):
        """The runner reads fn_config.tools for every agent; this config deliberately has none."""
        config = LcDeepResearchAgentConfig(llm="some_llm")
        assert getattr(config, "tools", None) is None


class TestAgentRun:
    def test_recursion_limit_reaches_the_graph_invocation(self, provider, stub_graph):
        config = LcDeepResearchAgentConfig(llm="some_llm", recursion_limit=57)
        agent = LcDeepResearchAgent(llm_provider=provider, config=config)
        asyncio.run(agent.run(LcDeepResearchAgentState(messages=[HumanMessage(content="q")])))

        assert stub_graph["config"]["recursion_limit"] == 57

    def test_only_the_caller_history_plus_the_report_is_returned(self, provider, stub_graph):
        """The graph's internal message trace stays out of AI-Q state."""
        agent = LcDeepResearchAgent(llm_provider=provider, config=None)
        state = LcDeepResearchAgentState(messages=[HumanMessage(content="q")])
        result = asyncio.run(agent.run(state))

        assert len(result.messages) == 2
        assert result.messages[-1].content == "# Report"
        assert result.files == {"/final_report.md": "# Report"}

    def test_data_sources_are_accepted_and_ignored(self, provider, stub_graph):
        """A UI source toggle cannot reach a tool that calls Tavily directly."""
        agent = LcDeepResearchAgent(llm_provider=provider, config=None)
        state = LcDeepResearchAgentState(messages=[HumanMessage(content="q")], data_sources=["web_search"])
        result = asyncio.run(agent.run(state))

        assert result.messages[-1].content == "# Report"

    def test_nat_tools_are_accepted_and_ignored(self, provider):
        """The job runner passes the config's resolved tools to every agent it builds."""
        agent = LcDeepResearchAgent(llm_provider=provider, tools=["not-a-real-tool"], config=None)
        assert agent.tools == ["not-a-real-tool"]

    def test_workflow_timeout_is_off_by_default(self, provider):
        agent = LcDeepResearchAgent(llm_provider=provider, config=LcDeepResearchAgentConfig(llm="x"))
        assert agent.workflow_timeout_seconds is None
