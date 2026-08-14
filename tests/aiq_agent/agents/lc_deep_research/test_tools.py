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

"""Tests for the ported upstream research tools."""

import logging

import httpx
import pytest

from aiq_agent.agents.lc_deep_research.research_agent import tools


@pytest.fixture(autouse=True)
def _reset_tavily_client(monkeypatch):
    """Keep the lazily-built module-level client from leaking between tests."""
    monkeypatch.setattr(tools, "_tavily_client", None)
    monkeypatch.delenv(tools._TOOL_IO_LOG_ENV, raising=False)


def test_tavily_client_is_not_built_at_import(monkeypatch):
    """The one deviation from upstream: no client construction until the first search.

    AI-Q imports every registered plugin at startup regardless of which agent the active config
    selects. Building TavilyClient at module scope -- as upstream does -- would raise on a missing
    TAVILY_API_KEY and take down configs that never reference this agent.
    """
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert tools._tavily_client is None


def test_get_tavily_client_is_memoized(monkeypatch):
    constructed = []

    class _FakeClient:
        def __init__(self):
            constructed.append(self)

    monkeypatch.setattr(tools, "TavilyClient", _FakeClient)
    first = tools._get_tavily_client()
    second = tools._get_tavily_client()

    assert first is second
    assert len(constructed) == 1


def _stub_search(monkeypatch, results):
    class _FakeClient:
        def search(self, query, max_results, topic):
            self.call = (query, max_results, topic)
            return {"results": results}

    client = _FakeClient()
    monkeypatch.setattr(tools, "_get_tavily_client", lambda: client)
    return client


def test_tavily_search_formats_results_like_upstream(monkeypatch):
    client = _stub_search(monkeypatch, [{"url": "https://example.com/a", "title": "Alpha"}])
    monkeypatch.setattr(tools, "fetch_webpage_content", lambda url: f"body of {url}")

    output = tools.tavily_search.invoke({"query": "what is x"})

    assert "🔍 Found 1 result(s) for 'what is x':" in output
    assert "## Alpha" in output
    assert "**URL:** https://example.com/a" in output
    assert "body of https://example.com/a" in output
    # Upstream's deliberate depth-over-breadth default: one URL per query, full page fetched.
    assert client.call == ("what is x", 1, "general")


def test_tavily_search_handles_no_results(monkeypatch):
    _stub_search(monkeypatch, [])
    output = tools.tavily_search.invoke({"query": "nothing"})
    assert "🔍 Found 0 result(s) for 'nothing':" in output


def test_tool_io_diagnostics_are_opt_in(monkeypatch, caplog):
    _stub_search(monkeypatch, [])

    with caplog.at_level(logging.INFO, logger=tools.__name__):
        tools.tavily_search.invoke({"query": "private research query"})

    assert "private research query" not in caplog.text


def test_tool_io_diagnostics_log_correlated_redacted_preview(monkeypatch, caplog):
    _stub_search(monkeypatch, [{"url": "https://example.com/a", "title": "Alpha"}])
    monkeypatch.setattr(
        tools,
        "fetch_webpage_content",
        lambda url, **kwargs: "page body NVIDIA_API_KEY=nvapi-secretvalue123",
    )
    monkeypatch.setenv(tools._TOOL_IO_LOG_ENV, "1")

    with caplog.at_level(logging.INFO, logger=tools.__name__):
        output = tools.tavily_search.invoke({"query": "actual diagnostic query"})

    assert "actual diagnostic query" in caplog.text
    assert '"event": "search_result"' in caplog.text
    assert '"event": "search_response"' in caplog.text
    assert "page body" in caplog.text
    assert "nvapi-secretvalue123" not in caplog.text
    assert "[REDACTED]" in caplog.text
    # Diagnostics must not modify the response that the model receives.
    assert "nvapi-secretvalue123" in output


def test_search_api_error_is_returned_as_text_not_raised(monkeypatch):
    """A rejected query must not kill the run.

    Regression test for an observed Nemotron Ultra failure: the model grew a query across retries
    until it passed Tavily's 1500-character limit, and upstream's bare .search() call let the
    resulting BadRequestError propagate out of the graph and fail the whole job with no answer.
    """

    class _FailingClient:
        def search(self, query, max_results, topic):
            raise ValueError("Query is too long. Max query length is 1500 characters.")

    monkeypatch.setattr(tools, "_get_tavily_client", lambda: _FailingClient())
    result = tools.tavily_search.invoke({"query": "a" * 2116})

    assert result.startswith("Error searching for '")
    assert "Max query length is 1500 characters." in result


def test_search_error_truncates_the_echoed_query(monkeypatch):
    """Replaying a multi-thousand-character query bloats context and invites the model to repeat it."""

    class _FailingClient:
        def search(self, query, max_results, topic):
            raise ValueError("boom")

    monkeypatch.setattr(tools, "_get_tavily_client", lambda: _FailingClient())
    result = tools.tavily_search.invoke({"query": "x" * 5000})

    assert result.count("x") == 200


def test_fetch_webpage_content_returns_error_string_instead_of_raising(monkeypatch):
    """Upstream swallows fetch failures into a string so one bad URL cannot abort the tool call."""

    def _boom(url, headers, timeout):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(tools.httpx, "get", _boom)
    result = tools.fetch_webpage_content("https://example.com/dead")

    assert result.startswith("Error fetching content from https://example.com/dead:")


def test_fetch_webpage_content_converts_html_to_markdown(monkeypatch):
    class _Response:
        text = "<h1>Title</h1><p>Body</p>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(tools.httpx, "get", lambda url, headers, timeout: _Response())
    result = tools.fetch_webpage_content("https://example.com")

    assert "Title" in result
    assert "<h1>" not in result


def test_think_tool_echoes_the_reflection():
    assert tools.think_tool.invoke({"reflection": "found 3 sources"}) == "Reflection recorded: found 3 sources"


def test_tool_names_are_the_upstream_names():
    """Tool names appear in the prompts; renaming them would desynchronise prompt and schema."""
    assert tools.tavily_search.name == "tavily_search"
    assert tools.think_tool.name == "think_tool"
