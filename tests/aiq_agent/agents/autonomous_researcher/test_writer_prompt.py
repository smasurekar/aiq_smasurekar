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

"""The writer's half of the answer-set contract, and the report depth it must not cost.

The writer is the autonomous researcher's long-report exit, so it is where "separate the answer
from the report" is most likely to be misread as "write less". These tests pin both halves: the
answer section exists and is set-disciplined, and the synthesis guidance that produces a full
report is still there underneath it.

The writer prompt had no content tests before the answer contract landed, which is why the
contradiction this file now guards against (an answer-set rule alongside "err on the side of more
useful information") survived unnoticed.
"""

from pathlib import Path

import pytest

_WRITER = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "aiq_agent"
    / "agents"
    / "autonomous_researcher"
    / "prompts"
    / "writer.j2"
)


@pytest.fixture(scope="module")
def writer_prompt() -> str:
    return _WRITER.read_text(encoding="utf-8")


class TestWriterAnswerContract:
    """An `## Answer` section when the request names a discrete target - and only then."""

    def test_answer_section_is_keyed_on_the_declared_answer_type(self, writer_prompt):
        """Shape follows the request, through the `answer_type` the planner already declares.

        `answer_strategy.answer_type` is defined in planner.j2 and read by this prompt already, so
        the contract reuses it instead of introducing a second notion of answer shape.
        """
        assert "`brief_answer` or `table`" in writer_prompt
        assert "`## Answer` section holding" in writer_prompt
        assert "`long_form_report`" in writer_prompt

    def test_long_form_reports_get_no_answer_section_and_no_length_target(self, writer_prompt):
        """The constraint the contract was accepted under: long reports stay long."""
        assert "write the report with no `## Answer` section and no length target" in writer_prompt

    def test_rejected_candidates_are_relocated_not_banned(self, writer_prompt):
        """Explaining why a candidate failed is good research writing; it just moves out of the
        answer section rather than being deleted from the report.
        """
        assert "### Considered and excluded" in writer_prompt
        assert "never inside `## Answer`" in writer_prompt


class TestWriterReportDepthSurvives:
    """The body contract is not the defect and must not be trimmed as collateral."""

    @pytest.mark.parametrize(
        "depth_rule",
        [
            "Cross-synthesize across research files",
            "build a coherent analytical narrative",
            "Point out meaningful conflicts or disagreement",
            "Preserve nuance around mechanisms, causality, trade-offs",
            "Use tables when the evidence has comparable entities",
            "Distinguish cited facts from your synthesis or inference",
        ],
    )
    def test_depth_guidance_is_intact(self, writer_prompt, depth_rule):
        assert depth_rule in writer_prompt, depth_rule

    def test_more_information_guidance_is_scoped_to_the_body_not_deleted(self, writer_prompt):
        """The old line applied to the whole report, which is what contradicted the answer-set
        rule. It is scoped, not removed: the body should still err toward more.
        """
        assert "In the body, err on the side of more useful information rather than less" in writer_prompt
        assert "This does not apply to the `## Answer` section" in writer_prompt
