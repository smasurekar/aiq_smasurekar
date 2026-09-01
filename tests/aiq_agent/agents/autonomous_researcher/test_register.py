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

"""The config surface and the inline finalize tool.

The point of the config assertions is that adding a capability to this agent means writing a
description, not adding a knob — so every tier and source-router knob must be *absent*, not merely
defaulted off.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import ValidationError

from aiq_agent.agents.autonomous_researcher import register as register_module
from aiq_agent.agents.autonomous_researcher.custom_middleware import AutonomousFinalReportCommitTracker
from aiq_agent.agents.autonomous_researcher.register import DEFAULT_EXCLUDED_SYNTHESIZING_TOOLS
from aiq_agent.agents.autonomous_researcher.register import AutonomousResearchAgentConfig
from aiq_agent.agents.autonomous_researcher.tools.finalize import FINAL_REPORT_META_PATH
from aiq_agent.agents.autonomous_researcher.tools.finalize import FINAL_REPORT_PATH
from aiq_agent.agents.autonomous_researcher.tools.finalize import build_submit_final_report_tool

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "configs/config_autonomous_frag.yml"

TIER_KNOBS = (
    "enabled_tiers",
    "enforce_tier_tools",
    "single_loop_single_shot",
    "dynamic_orchestrator_sections",
    "single_shot_search_budget",
    # The tier-qualified spelling stays absent: this agent's knob is `shallow_subagent`, and the
    # sub-agent it enables is reached by description rather than gated on a declared tier.
    "single_shot_shallow_subagent",
    "single_shot_researcher_llm",
)
SOURCE_ROUTER_KNOBS = ("source_router_llm", "enable_source_router", "domain_catalog_path")


class TestConfigSurface:
    def test_tier_knobs_are_absent(self):
        for knob in TIER_KNOBS:
            assert knob not in AutonomousResearchAgentConfig.model_fields, knob

    def test_source_router_knobs_are_absent(self):
        for knob in SOURCE_ROUTER_KNOBS:
            assert knob not in AutonomousResearchAgentConfig.model_fields, knob

    def test_carried_over_knobs_are_kept(self):
        fields = AutonomousResearchAgentConfig.model_fields
        for kept in (
            "orchestrator_llm",
            "researcher_llm",
            "planner_llm",
            "writer_llm",
            "tools",
            "exclude_tools",
            "enable_citation_verification",
            "researcher_loop_guard",
            "request_termination",
            "skills",
            "sandbox",
            "max_research_concurrency",
            "max_concurrent_source_tool_calls",
            "max_source_tool_batch_size",
        ):
            assert kept in fields, kept

    def test_shallow_subagent_knobs(self):
        fields = AutonomousResearchAgentConfig.model_fields
        assert fields["shallow_subagent"].default is True, "default-on: the easy request is the common case"
        assert fields["shallow_subagent_max_llm_turns"].default == 10
        assert fields["shallow_subagent_max_tool_iterations"].default == 5
        assert fields["shallow_subagent_escalate_on_budget_exhaustion"].default is True, (
            "default-on: a truncated report that ends the run scored -0.155 F1 against runs that did not"
        )

    @pytest.mark.parametrize("knob", ["shallow_subagent_max_llm_turns", "shallow_subagent_max_tool_iterations"])
    def test_shallow_loop_bounds_must_be_positive(self, knob):
        with pytest.raises(ValidationError):
            AutonomousResearchAgentConfig(orchestrator_llm="llm", **{knob: 0})

    def test_shallow_knobs_are_forwarded_to_the_agent(self):
        """A config knob that never reaches the constructor is silently inert."""
        source = Path(register_module.__file__).read_text(encoding="utf-8")
        for forwarded in (
            "shallow_subagent=config.shallow_subagent",
            "shallow_subagent_max_llm_turns=config.shallow_subagent_max_llm_turns",
            "shallow_subagent_max_tool_iterations=config.shallow_subagent_max_tool_iterations",
            "shallow_subagent_escalate_on_budget_exhaustion=config.shallow_subagent_escalate_on_budget_exhaustion",
        ):
            assert forwarded in source, forwarded

    def test_unknown_keys_are_rejected(self):
        with pytest.raises(ValidationError):
            AutonomousResearchAgentConfig(orchestrator_llm="llm", enabled_tiers=["deep"])

    def test_synthesizing_research_apis_are_excluded_by_default(self):
        """They return their own cited answers, bypassing the loop guards and citation registry."""
        assert set(DEFAULT_EXCLUDED_SYNTHESIZING_TOOLS) == {"you_research", "you_finance_research"}


class TestShippedConfig:
    @pytest.fixture(scope="class")
    def config(self) -> dict:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_exists_and_wires_the_autonomous_workflow(self, config):
        assert config["workflow"]["_type"] == "autonomous_research_workflow"

    def test_agent_config_validates_against_the_schema(self, config):
        agent = dict(config["functions"]["autonomous_research_agent"])
        agent.pop("_type")
        AutonomousResearchAgentConfig(**agent)

    def test_config_declares_no_tier_or_router_knobs(self, config):
        agent = config["functions"]["autonomous_research_agent"]
        for knob in (*TIER_KNOBS, *SOURCE_ROUTER_KNOBS):
            assert knob not in agent, knob

    def test_config_enables_the_shallow_subagent(self, config):
        assert config["functions"]["autonomous_research_agent"]["shallow_subagent"] is True

    def test_config_fails_the_shallow_subagent_on_budget_exhaustion(self, config):
        agent = config["functions"]["autonomous_research_agent"]
        assert agent["shallow_subagent_escalate_on_budget_exhaustion"] is True


class TestSubmitFinalReport:
    @staticmethod
    def _backend():
        backend = MagicMock()
        backend.upload_files = MagicMock(return_value=[MagicMock(path="p", error=None)])
        return backend

    def test_signature_drops_the_tier_argument(self):
        tool = build_submit_final_report_tool(backend=None, tracker=None)
        assert set(tool.args) == {"markdown", "researched"}

    def test_writes_both_files_and_commits_the_inline_exit(self):
        backend, tracker = self._backend(), AutonomousFinalReportCommitTracker()
        tool = build_submit_final_report_tool(backend=backend, tracker=tracker)

        tool.invoke({"markdown": "# Answer\n\nBody.", "researched": True})

        files = dict(backend.upload_files.call_args.args[0])
        assert files[FINAL_REPORT_PATH] == b"# Answer\n\nBody."
        assert json.loads(files[FINAL_REPORT_META_PATH]) == {"researched": True}
        assert tracker.any_exit_committed({})

    def test_no_tier_metadata_is_recorded(self):
        backend = self._backend()
        tool = build_submit_final_report_tool(backend=backend, tracker=AutonomousFinalReportCommitTracker())
        tool.invoke({"markdown": "# A", "researched": False})
        meta = json.loads(dict(backend.upload_files.call_args.args[0])[FINAL_REPORT_META_PATH])
        assert "tier" not in meta

    def test_empty_markdown_is_rejected(self):
        tool = build_submit_final_report_tool(backend=self._backend(), tracker=None)
        with pytest.raises(ValueError, match="non-empty"):
            tool.func(markdown="   ")

    def test_a_failed_write_does_not_commit_the_exit(self):
        """A digest with no readable report would satisfy the finalization guard falsely."""
        backend = MagicMock()
        backend.upload_files = MagicMock(return_value=[MagicMock(path=FINAL_REPORT_PATH, error="disk full")])
        tracker = AutonomousFinalReportCommitTracker()
        tool = build_submit_final_report_tool(backend=backend, tracker=tracker)
        with pytest.raises(RuntimeError, match="failed to record final report"):
            tool.func(markdown="# A", researched=True)
        assert not tracker.any_exit_committed({})

    def test_return_direct_ends_the_react_loop(self):
        assert build_submit_final_report_tool(backend=None, tracker=None).return_direct is True


class TestResearchDoorFlags:
    """`research_batch_tool` / `researcher_subagent` gate the two delegated-research doors."""

    def test_the_batch_is_the_default_door_and_the_direct_one_is_opt_in(self):
        """Batch-only is the shipped default: the researcher runs behind run_research_batch, and
        `task(researcher-agent)` is a second door onto that same worker that must be asked for.
        """
        fields = AutonomousResearchAgentConfig.model_fields
        assert fields["research_batch_tool"].default is True
        assert fields["researcher_subagent"].default is False

    def test_the_default_config_holds_a_research_path(self):
        """The default must not be the rejected both-off combination."""
        config = AutonomousResearchAgentConfig(orchestrator_llm="llm")
        assert config.research_batch_tool is True
        assert config.researcher_subagent is False

    @pytest.mark.parametrize(
        ("batch", "subagent"),
        [(True, True), (True, False), (False, True)],
    )
    def test_each_single_door_arm_validates(self, batch, subagent):
        config = AutonomousResearchAgentConfig(
            orchestrator_llm="llm",
            research_batch_tool=batch,
            researcher_subagent=subagent,
        )
        assert config.research_batch_tool is batch
        assert config.researcher_subagent is subagent

    def test_both_doors_off_is_rejected(self):
        """Not a stylistic guard: the agent would load, serve, and answer from an exhausted budget.

        With neither door the orchestrator's only retrieval is its own direct source-tool calls,
        capped at ``max_direct_source_calls`` (default 2), and the shallow-researcher is
        first-turn-only and ends the run when it succeeds. That is a silent quality collapse, so it
        has to fail at config load rather than at answer time.
        """
        with pytest.raises(ValidationError, match="cannot both be false"):
            AutonomousResearchAgentConfig(
                orchestrator_llm="llm",
                research_batch_tool=False,
                researcher_subagent=False,
            )

    def test_door_knobs_are_forwarded_to_the_agent(self):
        """A config knob that never reaches the constructor is silently inert."""
        source = Path(register_module.__file__).read_text(encoding="utf-8")
        for forwarded in (
            "research_batch_tool=config.research_batch_tool",
            "researcher_subagent=config.researcher_subagent",
        ):
            assert forwarded in source, forwarded


class TestShippedConfigsAreBatchOnly:
    """Every shipped config must reach the researcher through run_research_batch.

    The direct `task(researcher-agent)` door is an opt-in second route onto the same worker, kept
    for the eval arms. A shipped config that opened it would silently change which arm a default
    deployment runs, and the two are not equal-budget (see the eval-fairness note in
    configs/config_autonomous_frag.yml).
    """

    FRESHQA_PATH = REPO_ROOT / "frontends/benchmarks/freshqa/configs/config_autonomous_frag_freshqa.yml"

    @pytest.mark.parametrize("path", [CONFIG_PATH, FRESHQA_PATH])
    def test_the_batch_door_is_open_the_direct_door_is_not(self, path):
        if not path.exists():
            pytest.skip(f"{path.name} is not present in this checkout")
        agent = dict(yaml.safe_load(path.read_text(encoding="utf-8"))["functions"]["autonomous_research_agent"])
        agent.pop("_type")
        config = AutonomousResearchAgentConfig(**agent)
        assert config.research_batch_tool is True
        assert config.researcher_subagent is False
