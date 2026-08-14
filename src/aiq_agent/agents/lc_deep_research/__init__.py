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

"""LangChain DeepAgents deep-research example, packaged as an AI-Q research agent.

``research_agent/`` holds the upstream example's prompts and tools. ``prompts.py`` is byte-identical
to ``deepagents/examples/deep_research/research_agent/prompts.py`` and must stay that way -- the
fidelity test in ``tests/aiq_agent/agents/lc_deep_research/`` enforces it. That is why this package
uses a plain ``prompts.py`` rather than AI-Q's usual Jinja2 ``prompts/*.j2`` convention: the point
of this arm is to measure the upstream design, and a reformatted prompt is a different prompt.
"""

from . import register  # noqa: F401
from .register import lc_deep_research_agent  # noqa: F401

__all__ = [
    "lc_deep_research_agent",
]
