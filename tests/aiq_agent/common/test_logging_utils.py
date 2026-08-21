# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for opaque log correlation references."""

import pytest

from aiq_agent.common.logging_utils import DEFAULT_PAYLOAD_MAX_CHARS
from aiq_agent.common.logging_utils import log_content_metadata
from aiq_agent.common.logging_utils import log_identifier_ref
from aiq_agent.common.logging_utils import payload_logging_enabled
from aiq_agent.common.logging_utils import payload_max_chars
from aiq_agent.common.logging_utils import truncate_payload


def test_log_identifier_ref_is_stable_without_exposing_identifier() -> None:
    identifier = "8d312ad2-d097-42b8-93f1-6df4c084d6d4"

    first = log_identifier_ref(identifier)
    second = log_identifier_ref(identifier)

    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 12
    assert identifier not in first
    assert identifier[:8] not in first
    assert log_identifier_ref("different") != first


def test_log_content_metadata_preserves_shape_without_exposing_content() -> None:
    content = "my fake secret is nvapi-vdr-do-not-log"

    metadata = log_content_metadata(content)

    assert metadata == f"chars={len(content)} ref={log_identifier_ref(content)}"
    assert content not in metadata
    assert "nvapi-vdr-do-not-log" not in metadata


def test_log_content_metadata_omits_payload_by_default(monkeypatch) -> None:
    monkeypatch.delenv("AIQ_LOG_PAYLOADS", raising=False)
    content = "my fake secret is nvapi-vdr-do-not-log"

    metadata = log_content_metadata(content)

    assert "payload=" not in metadata
    assert content not in metadata


def test_log_content_metadata_appends_payload_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("AIQ_LOG_PAYLOADS", "1")
    content = '{"confidence": "very high"}'

    metadata = log_content_metadata(content)

    # The metadata prefix survives so digest-based correlation still works with the switch on.
    assert metadata.startswith(f"chars={len(content)} ref={log_identifier_ref(content)} ")
    assert metadata.endswith(f"payload={content}")


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on", " on "])
def test_payload_logging_accepts_the_documented_truthy_spellings(monkeypatch, flag: str) -> None:
    monkeypatch.setenv("AIQ_LOG_PAYLOADS", flag)

    assert payload_logging_enabled() is True


@pytest.mark.parametrize("flag", ["", "0", "false", "no", "off", "maybe"])
def test_payload_logging_stays_off_for_anything_else(monkeypatch, flag: str) -> None:
    monkeypatch.setenv("AIQ_LOG_PAYLOADS", flag)

    assert payload_logging_enabled() is False


def test_truncate_payload_keeps_both_ends_and_reports_the_gap(monkeypatch) -> None:
    monkeypatch.setenv("AIQ_LOG_PAYLOAD_MAX_CHARS", "20")
    text = "HEAD" + ("x" * 100) + "TAIL"

    truncated = truncate_payload(text)

    # Both ends survive: a malformed payload usually goes wrong at the tail.
    assert truncated.startswith("HEAD")
    assert truncated.endswith("TAIL")
    assert f"[{len(text) - 20} chars elided]" in truncated


def test_truncate_payload_treats_zero_as_no_cap(monkeypatch) -> None:
    monkeypatch.setenv("AIQ_LOG_PAYLOAD_MAX_CHARS", "0")
    text = "x" * 100_000

    assert truncate_payload(text) == text


@pytest.mark.parametrize("raw", ["", "not-a-number", "-5"])
def test_payload_max_chars_falls_back_to_the_default(monkeypatch, raw: str) -> None:
    monkeypatch.setenv("AIQ_LOG_PAYLOAD_MAX_CHARS", raw)

    # A negative cap would silently mean "no cap"; clamping keeps the documented default.
    expected = 0 if raw == "-5" else DEFAULT_PAYLOAD_MAX_CHARS
    assert payload_max_chars() == expected
