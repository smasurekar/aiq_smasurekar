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

"""Compiled sub-agents registered with the adaptive research orchestrator."""

from .shallow import MAX_SHALLOW_ATTEMPTS
from .shallow import SHALLOW_RESEARCHER_SUBAGENT
from .shallow import ShallowSubagentCapture
from .shallow import build_shallow_researcher_subagent
from .shallow import last_human_text

__all__ = [
    "MAX_SHALLOW_ATTEMPTS",
    "SHALLOW_RESEARCHER_SUBAGENT",
    "ShallowSubagentCapture",
    "build_shallow_researcher_subagent",
    "last_human_text",
]
