# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for logging stable references without exposing sensitive content."""

import hashlib
import os

PAYLOAD_ENV_VAR = "AIQ_LOG_PAYLOADS"
PAYLOAD_MAX_CHARS_ENV_VAR = "AIQ_LOG_PAYLOAD_MAX_CHARS"
DEFAULT_PAYLOAD_MAX_CHARS = 20_000

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def payload_logging_enabled() -> bool:
    """Return whether this process opted into writing raw content to the logs.

    Read on every call rather than cached at import: this is a switch flipped for one
    debugging run, and caching it would make it untestable and unresponsive to a change
    made after the module is first imported.
    """
    return os.environ.get(PAYLOAD_ENV_VAR, "").strip().lower() in _TRUTHY


def payload_max_chars() -> int:
    """Return the per-payload character cap; ``0`` means no cap."""
    raw = os.environ.get(PAYLOAD_MAX_CHARS_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_PAYLOAD_MAX_CHARS
    try:
        return max(int(raw), 0)
    except ValueError:
        return DEFAULT_PAYLOAD_MAX_CHARS


def truncate_payload(text: str, limit: int | None = None) -> str:
    """Trim an oversized payload from the middle, keeping both ends.

    Head-only truncation drops the tail, which is where a malformed JSON payload usually
    goes wrong and where the closing structure that identifies a schema violation lives.
    """
    cap = payload_max_chars() if limit is None else limit
    if cap <= 0 or len(text) <= cap:
        return text
    head = cap // 2
    tail = cap - head
    return f"{text[:head]}\n...[{len(text) - cap} chars elided]...\n{text[-tail:]}"


def log_identifier_ref(identifier: str) -> str:
    """Return a stable correlation reference that does not reveal ``identifier``."""
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


def log_content_metadata(content: object) -> str:
    """Return safe, correlatable metadata for content that must not be logged.

    Prompts, model responses, tool payloads, and exception details can contain
    credentials or private customer data. Logging their length and a stable
    digest preserves enough signal to correlate retries without writing the
    content itself to production logs.

    Setting ``AIQ_LOG_PAYLOADS=1`` additionally appends the content. It is off by default
    and is meant for a single diagnostic run: a digest proves that a model re-sent the same
    bytes, but only the payload says *which* bytes were wrong. Sizing is controlled by
    ``AIQ_LOG_PAYLOAD_MAX_CHARS`` (default 20000; ``0`` removes the cap).

    The metadata prefix is emitted either way, so existing log parsing and
    correlation-by-digest keep working with the switch on.
    """
    text = content if isinstance(content, str) else str(content)
    metadata = f"chars={len(text)} ref={log_identifier_ref(text)}"
    if not payload_logging_enabled():
        return metadata
    return f"{metadata} payload={truncate_payload(text)}"
