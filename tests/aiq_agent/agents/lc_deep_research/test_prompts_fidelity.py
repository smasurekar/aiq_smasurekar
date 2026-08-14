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

"""Fidelity tests: the ported prompt surface must stay byte-identical to upstream.

This is the most load-bearing test in the package. ``lc_deep_research`` exists to measure the
upstream LangChain DeepAgents deep-research design on AI-Q's harness, so any edit to its prompts --
however well-intentioned -- silently turns the arm into a measurement of something else. These
digests are the tripwire.

If a test here fails you have three legitimate options, and "update the digest to make it pass" is
only the third one:

1. Revert the prompt edit. Almost always the right answer.
2. If upstream itself changed and you are deliberately re-syncing, re-copy
   ``deepagents/examples/deep_research/research_agent/prompts.py`` wholesale, update the digests,
   and say so in the commit message -- the eval baselines are invalidated.
3. If you are intentionally forking the prompts, this package is the wrong home for that work;
   the adaptive and autonomous agents are where AI-Q prompt engineering belongs.
"""

import hashlib
from pathlib import Path

import pytest

from aiq_agent.agents.lc_deep_research import agent as lc_agent
from aiq_agent.agents.lc_deep_research.research_agent import prompts

# sha256 of the file as copied from
# deepagents/examples/deep_research/research_agent/prompts.py (deepagents repo @ 822f7c9b0),
# normalised to end in exactly one newline. Upstream ships the file with no trailing newline;
# pre-commit's end-of-file-fixer adds one. That byte sits after the closing triple-quote and so
# changes no prompt constant, hence the normalisation rather than an exclusion.
#
# The `pragma: allowlist secret` markers below are for detect-secrets, which classifies any 64-char
# hex string as a high-entropy credential. These are sha256 digests of public prompt text.
UPSTREAM_PROMPTS_FILE_SHA256 = (
    "7ae99c00184f12181646be5d5d2738f19ab2bfbe942b040c0a5412a8b936fc4a"  # pragma: allowlist secret
)

UPSTREAM_CONSTANT_SHA256 = {
    "RESEARCH_WORKFLOW_INSTRUCTIONS": (
        "7fb030f73b6a8dbd44267361bc2ba9e98b3bab4f217087247c111f8e568f8685"  # pragma: allowlist secret
    ),
    "RESEARCHER_INSTRUCTIONS": (
        "33f83ff7cc865ec378b18a78bfdc0c791b0d2ba8f9a6b62ba65ceb1fd72a66e3"  # pragma: allowlist secret
    ),
    "SUBAGENT_DELEGATION_INSTRUCTIONS": (
        "ba5fbea8ef279a56676aa55cdb5d17d8dd7d708d8a3985a5c83a4b311eeece48"  # pragma: allowlist secret
    ),
    "TASK_DESCRIPTION_PREFIX": (
        "3c991f149d704dae54b835295cd74880d1a9530b18bbc00c9b56e8741d3961db"  # pragma: allowlist secret
    ),
}


def test_prompts_module_is_byte_identical_to_upstream():
    """The whole file, not just the constants -- upstream is copied wholesale, comments included."""
    content = Path(prompts.__file__).read_bytes().rstrip(b"\n") + b"\n"
    digest = hashlib.sha256(content).hexdigest()
    assert digest == UPSTREAM_PROMPTS_FILE_SHA256, (
        "research_agent/prompts.py diverged from upstream. See this module's docstring before updating the digest."
    )


@pytest.mark.parametrize("name", sorted(UPSTREAM_CONSTANT_SHA256))
def test_prompt_constant_unchanged(name):
    """Per-constant digests, so a failure names which prompt drifted."""
    digest = hashlib.sha256(getattr(prompts, name).encode("utf-8")).hexdigest()
    assert digest == UPSTREAM_CONSTANT_SHA256[name]


def test_subagent_description_is_upstream_verbatim():
    """The sub-agent description is prompt surface: SubAgentMiddleware renders it into `task`."""
    assert lc_agent.RESEARCH_SUBAGENT_DESCRIPTION == (
        "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time."
    )
    assert lc_agent.RESEARCH_SUBAGENT_NAME == "research-agent"


def test_orchestrator_instructions_match_upstream_assembly():
    """Upstream joins the two orchestrator prompts with a blank line, 80 '=', and a blank line."""
    expected = (
        prompts.RESEARCH_WORKFLOW_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + prompts.SUBAGENT_DELEGATION_INSTRUCTIONS.format(
            max_concurrent_research_units=3,
            max_researcher_iterations=3,
        )
    )
    assert lc_agent.build_orchestrator_instructions() == expected


def test_orchestrator_never_sees_researcher_instructions():
    """RESEARCHER_INSTRUCTIONS is sub-agent-only upstream; leaking it would change orchestrator behaviour."""
    instructions = lc_agent.build_orchestrator_instructions()
    assert "<Hard Limits>" not in instructions
    assert "Tool Call Budgets" not in instructions


def test_upstream_limit_defaults():
    """The delegation limits are accuracy-relevant defaults, not arbitrary numbers."""
    assert lc_agent.DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS == 3
    assert lc_agent.DEFAULT_MAX_RESEARCHER_ITERATIONS == 3
