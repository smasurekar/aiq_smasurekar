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

"""Tests for graph construction and final-report extraction."""

import pytest
from langchain_core.messages import AIMessage

from aiq_agent.agents.lc_deep_research import agent as lc_agent
from aiq_agent.agents.lc_deep_research.agent import FINAL_REPORT_PATH
from aiq_agent.agents.lc_deep_research.agent import build_lc_deep_research_graph
from aiq_agent.agents.lc_deep_research.agent import extract_final_report


class TestExtractFinalReport:
    """The report lives on the virtual filesystem, not in the closing chat message."""

    @pytest.mark.parametrize(
        "entry",
        [
            "# Report\n\nBody.",
            b"# Report\n\nBody.",
            {"content": "# Report\n\nBody."},
            {"content": b"# Report\n\nBody."},
        ],
        ids=["str", "bytes", "dict-str", "dict-bytes"],
    )
    def test_reads_every_backend_file_shape(self, entry):
        """DeepAgents backends store file contents in any of these shapes depending on backend."""
        result = {"files": {FINAL_REPORT_PATH: entry}, "messages": [AIMessage(content="done")]}
        assert extract_final_report(result) == "# Report\n\nBody."

    def test_report_file_wins_over_the_final_message(self):
        """The closing message is normally an acknowledgement; preferring it would truncate the answer."""
        result = {
            "files": {FINAL_REPORT_PATH: "# Full report"},
            "messages": [AIMessage(content="I have written the report.")],
        }
        assert extract_final_report(result) == "# Full report"

    def test_falls_back_to_the_final_message_when_no_report_file(self):
        """Covers greetings, capability questions, and runs that answered inline."""
        result = {"files": {}, "messages": [AIMessage(content="Hello! I research topics.")]}
        assert extract_final_report(result) == "Hello! I research topics."

    def test_blank_report_file_falls_through_to_the_message(self):
        result = {"files": {FINAL_REPORT_PATH: "   \n  "}, "messages": [AIMessage(content="fallback")]}
        assert extract_final_report(result) == "fallback"

    def test_joins_content_block_messages(self):
        """Some providers return content blocks rather than a flat string."""
        blocks = [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]
        result = {"files": {}, "messages": [AIMessage(content=blocks)]}
        assert extract_final_report(result) == "part one part two"

    def test_raises_when_there_is_nothing_to_return(self):
        """An empty answer must fail here, where the message is actionable, not in the eval runner."""
        with pytest.raises(ValueError, match="neither /final_report.md nor a non-empty final message"):
            extract_final_report({"files": {}, "messages": [AIMessage(content="  ")]})

    def test_raises_on_a_result_with_no_messages_and_no_files(self):
        with pytest.raises(ValueError):
            extract_final_report({})


class TestBuildGraph:
    """Topology assertions -- the graph must stay upstream's single-orchestrator/one-subagent shape."""

    def test_builds_with_a_fake_chat_model(self):
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        graph = build_lc_deep_research_graph(GenericFakeChatModel(messages=iter([AIMessage(content="ok")])))
        assert graph is not None

    def test_researcher_prompt_is_dated(self, monkeypatch):
        """RESEARCHER_INSTRUCTIONS is .format(date=...)-templated; an unrendered field would leak."""
        captured = {}

        def _fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(lc_agent, "create_deep_agent", _fake_create_deep_agent)
        build_lc_deep_research_graph(object(), current_date="2026-01-31")

        (subagent,) = captured["subagents"]
        assert "today's date is 2026-01-31" in subagent["system_prompt"]
        assert "{date}" not in subagent["system_prompt"]

    def test_topology_matches_upstream(self, monkeypatch):
        captured = {}

        def _fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(lc_agent, "create_deep_agent", _fake_create_deep_agent)
        build_lc_deep_research_graph(object())

        assert [t.name for t in captured["tools"]] == ["tavily_search", "think_tool"]
        (subagent,) = captured["subagents"]
        assert subagent["name"] == "research-agent"
        assert [t.name for t in subagent["tools"]] == ["tavily_search", "think_tool"]

    def test_limits_are_rendered_into_the_orchestrator_prompt(self, monkeypatch):
        captured = {}

        def _fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(lc_agent, "create_deep_agent", _fake_create_deep_agent)
        build_lc_deep_research_graph(object(), max_concurrent_research_units=5, max_researcher_iterations=2)

        assert "Use at most 5 parallel sub-agents per iteration" in captured["system_prompt"]
        assert "Stop after 2 delegation rounds" in captured["system_prompt"]
