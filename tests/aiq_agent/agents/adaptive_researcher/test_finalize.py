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

"""Tests for the submit_final_report finalize tool."""

import json
from unittest.mock import MagicMock

import pytest
from deepagents.backends.protocol import FileUploadResponse

from aiq_agent.agents.adaptive_researcher.tools.finalize import FINAL_REPORT_META_PATH
from aiq_agent.agents.adaptive_researcher.tools.finalize import FINAL_REPORT_PATH
from aiq_agent.agents.adaptive_researcher.tools.finalize import build_submit_final_report_tool


def _fake_backend():
    backend = MagicMock()
    backend.upload_files.side_effect = lambda files: [
        FileUploadResponse(path=path, error=None) for path, _content in files
    ]
    return backend


class TestSubmitFinalReportTool:
    @pytest.mark.asyncio
    async def test_writes_report_and_meta_researched_true(self):
        backend = _fake_backend()
        tool = build_submit_final_report_tool(backend=backend)

        result = await tool.ainvoke({"markdown": "# Answer\n\nBody [1].", "researched": True, "tier": "single_shot"})

        assert result == "Recorded final report."
        backend.upload_files.assert_called_once()
        files = dict(backend.upload_files.call_args.args[0])
        assert files[FINAL_REPORT_PATH].decode("utf-8") == "# Answer\n\nBody [1]."
        assert json.loads(files[FINAL_REPORT_META_PATH].decode("utf-8")) == {
            "researched": True,
            "tier": "single_shot",
        }

    @pytest.mark.asyncio
    async def test_researched_false_round_trips(self):
        backend = _fake_backend()
        tool = build_submit_final_report_tool(backend=backend)

        await tool.ainvoke({"markdown": "Hello! I'm the AI-Q research assistant.", "researched": False, "tier": "meta"})

        files = dict(backend.upload_files.call_args.args[0])
        assert json.loads(files[FINAL_REPORT_META_PATH].decode("utf-8")) == {"researched": False, "tier": "meta"}

    @pytest.mark.asyncio
    async def test_researched_defaults_true(self):
        backend = _fake_backend()
        tool = build_submit_final_report_tool(backend=backend)

        await tool.ainvoke({"markdown": "# Answer\n\nBody."})

        files = dict(backend.upload_files.call_args.args[0])
        assert json.loads(files[FINAL_REPORT_META_PATH].decode("utf-8")) == {"researched": True, "tier": None}

    @pytest.mark.asyncio
    async def test_tier_round_trips(self):
        backend = _fake_backend()
        tool = build_submit_final_report_tool(backend=backend)

        await tool.ainvoke({"markdown": "# Answer\n\nBody [1].", "researched": True, "tier": "standard"})

        files = dict(backend.upload_files.call_args.args[0])
        assert json.loads(files[FINAL_REPORT_META_PATH].decode("utf-8"))["tier"] == "standard"

    @pytest.mark.asyncio
    async def test_tier_defaults_none(self):
        backend = _fake_backend()
        tool = build_submit_final_report_tool(backend=backend)

        await tool.ainvoke({"markdown": "# Answer\n\nBody."})

        files = dict(backend.upload_files.call_args.args[0])
        assert json.loads(files[FINAL_REPORT_META_PATH].decode("utf-8"))["tier"] is None

    @pytest.mark.asyncio
    async def test_empty_markdown_rejected(self):
        backend = _fake_backend()
        tool = build_submit_final_report_tool(backend=backend)

        with pytest.raises(ValueError, match="non-empty"):
            await tool.ainvoke({"markdown": "   ", "researched": True})
        backend.upload_files.assert_not_called()

    @pytest.mark.asyncio
    async def test_backend_upload_error_surfaces(self):
        backend = MagicMock()
        backend.upload_files.side_effect = lambda files: [
            FileUploadResponse(path=path, error="disk full") for path, _content in files
        ]
        tool = build_submit_final_report_tool(backend=backend)

        with pytest.raises(RuntimeError, match="failed to record final report"):
            await tool.ainvoke({"markdown": "# Answer\n\nBody.", "researched": True})

    @pytest.mark.asyncio
    async def test_no_backend_is_noop_but_validates(self):
        tool = build_submit_final_report_tool(backend=None)
        # still validates, still returns marker, just does not persist
        assert await tool.ainvoke({"markdown": "# Answer\n\nBody."}) == "Recorded final report."
        with pytest.raises(ValueError):
            await tool.ainvoke({"markdown": "", "researched": False})
