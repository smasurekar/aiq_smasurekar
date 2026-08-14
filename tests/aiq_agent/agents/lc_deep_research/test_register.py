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

"""The config surface and the shipped workflow config.

The assertions about *absent* knobs are the point: this agent is a reference implementation, and
every AI-Q affordance it grows (tool lists, citation verification, loop guards, tier routing) makes
the arm measure something other than the upstream design.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aiq_agent.agents.lc_deep_research.register import DEFAULT_RECURSION_LIMIT
from aiq_agent.agents.lc_deep_research.register import LcDeepResearchAgentConfig

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "configs/config_lc_deep_research_frag.yml"

# Knobs the sibling agents have that this one must NOT grow.
AIQ_MACHINERY_KNOBS = (
    "tools",
    "exclude_tools",
    "enable_citation_verification",
    "researcher_loop_guard",
    "request_termination",
    "enabled_tiers",
    "source_router_llm",
    "enable_source_router",
    "skills",
    "sandbox",
    "orchestrator_llm",
    "researcher_llm",
    "planner_llm",
    "writer_llm",
)


class TestConfigSurface:
    def test_aiq_machinery_knobs_are_absent(self):
        for knob in AIQ_MACHINERY_KNOBS:
            assert knob not in LcDeepResearchAgentConfig.model_fields, knob

    def test_expected_knobs_are_present(self):
        fields = LcDeepResearchAgentConfig.model_fields
        for kept in (
            "llm",
            "verbose",
            "max_concurrent_research_units",
            "max_researcher_iterations",
            "recursion_limit",
            "workflow_timeout_seconds",
        ):
            assert kept in fields, kept

    def test_defaults_match_upstream(self):
        config = LcDeepResearchAgentConfig(llm="some_llm")
        assert config.max_concurrent_research_units == 3
        assert config.max_researcher_iterations == 3
        # No deadline by default: upstream has none and the eval harness applies its own.
        assert config.workflow_timeout_seconds is None

    def test_recursion_limit_clears_langgraph_default(self):
        """LangGraph's default of 25 aborts healthy multi-round runs with a GraphRecursionError."""
        assert DEFAULT_RECURSION_LIMIT > 25
        assert LcDeepResearchAgentConfig(llm="some_llm").recursion_limit == DEFAULT_RECURSION_LIMIT

    def test_unknown_keys_are_rejected(self):
        with pytest.raises(ValidationError):
            LcDeepResearchAgentConfig(llm="some_llm", enable_citation_verification=True)

    def test_llm_is_required(self):
        with pytest.raises(ValidationError):
            LcDeepResearchAgentConfig()


class TestShippedConfig:
    @pytest.fixture(scope="class")
    def config(self):
        return yaml.safe_load(CONFIG_PATH.read_text())

    def test_workflow_is_the_string_in_wrapper(self, config):
        """The eval harness calls the workflow with a plain string and reads a ChatResponse back."""
        assert config["workflow"]["_type"] == "lc_deep_research_workflow"

    def test_agent_block_validates_against_the_schema(self, config):
        block = dict(config["functions"]["lc_deep_research_agent"])
        block.pop("_type")
        agent_config = LcDeepResearchAgentConfig(**block)
        assert agent_config.max_concurrent_research_units == 3
        assert agent_config.max_researcher_iterations == 3

    def test_uses_nemotron_ultra(self, config):
        llm = config["llms"]["nemotron_super_llm"]
        assert llm["model_name"] == "nvidia/nvidia/nemotron-3-ultra"
        assert config["functions"]["lc_deep_research_agent"]["llm"] == "nemotron_super_llm"

    def test_declares_the_web_search_data_source_for_harness_preflight(self, config):
        """Inert for this agent, but the Harbor eval preflight asserts it with strict_tools: true."""
        source_ids = {source["id"] for source in config["functions"]["data_sources"]["sources"]}
        assert "web_search" in source_ids
        for tool_name in ("web_search_tool", "advanced_web_search_tool"):
            assert tool_name in config["functions"]

    def test_agent_takes_no_tool_list(self, config):
        """Upstream's tavily_search calls Tavily directly; wiring NAT tools here would fork it."""
        assert "tools" not in config["functions"]["lc_deep_research_agent"]
