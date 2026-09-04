# Tavily Extract: empirical security findings

Measured 2026-08-25 against the live `api.tavily.com/extract` endpoint using the AI-Q production
key. Probes: `misc/tavily/security_probe.py`, `misc/tavily/security_probe2.py`, plus a scheme /
attribute control run. Every number below is reproducible with those scripts.

Scope: this tests what the endpoint *does*, not what the contract *says*. Retention and training
terms are a legal question and are not testable here.

## Summary

| Question | Finding |
| :-- | :-- |
| Does Extract return raw web data? | **Yes.** Full page text, verbatim, 14-25x a Search chunk. |
| Does Tavily filter adversarial content on Extract? | **No evidence of any filtering.** |
| Does Extract enable SSRF? | **No, at the vendor boundary.** Private/loopback/metadata/non-HTTP targets and known rebinding hosts are all refused. |
| Does hidden markup carry through? | **Partly.** Comments and `display:none` are stripped; non-rendered `title=` attribute text is not. |
| Does AI-Q ever fetch a URL itself? | **No.** Every path goes through `TavilyExtract`; there is no direct-HTTP fallback. |

## T1 - Extract returns raw page content

`https://en.wikipedia.org/wiki/Prompt_injection`:

- Source HTML: 175,611 chars
- Browser-visible text (direct fetch, scripts/styles/comments stripped): 19,116 chars
- Tavily `raw_content`: **34,151 chars = 179% of the visible text**

It exceeds 100% because Extract emits markdown, so link targets and attribute text are added to
the prose. Response fields are `url`, `title`, `raw_content`, `images` — there is no summary
field, no score, no safety annotation. Verbatim reproduction of sampled page sentences was 6/20;
the misses are whitespace/markdown reflow inside tables and lists, not content rewriting.

**This settles the framing question: Extract is not summarization and not chunking. It is the
page.**

## T2 - Hidden-markup injection carriers: partly closed

Tested on the same page, comparing the source HTML against `raw_content`:

| Carrier | Candidates on page | Present in output |
| :-- | --: | --: |
| `<!-- HTML comments -->` | 5 | **0** |
| `display:none` / `aria-hidden` / `hidden` elements | 1 | **0** |
| `title="..."` attributes (not rendered without hover) | 40 | **23** |
| `alt="..."` attributes | 0 | 0 |

The two classic carriers are stripped. But `title=` text, which a human reader never sees,
reaches the model verbatim. That is a live, if narrow, indirect-injection channel. It was not
possible to test white-on-white text or zero-size fonts without publishing a controlled page.

## T3 - No content filtering on Extract

Three public repositories whose entire subject matter is prompt injection and jailbreaking were
extracted successfully, in full, with no error, no redaction, and no warning:

| URL | Returned | Markers found |
| :-- | --: | :-- |
| `greshake/llm-security` README | 9,086 chars | injected imperative |
| `TakSec/Prompt-Injection-Everywhere` README | 3,517 chars | "ignore previous/above", injected imperative |
| `verazuo/jailbreak_llms` README | 7,031 chars | "do anything now"/DAN, "jailbreak" |

The `TakSec` page returned working injection payloads verbatim, including the literal
"ignore previous instructions" family. The response envelope carries only `request_id`,
`response_time`, `results`, `failed_results` — there is no moderation verdict field to inspect,
which is itself the finding: **Extract exposes no content-safety signal, so we cannot claim one
exists.**

Rachel's "malicious content guardrails" point is not observable at this endpoint. If Tavily runs
safety on Extract, it did not act on the strongest available test material and reports nothing.

## T4 - SSRF surface is closed at the vendor boundary

Every internal target was refused with HTTP 400 `Validation Error: Invalid URL format`:

```
http://169.254.169.254/latest/meta-data/      refused
http://metadata.google.internal/...            refused
http://127.0.0.1:8000/                         refused
http://[::1]:8000/                             refused
http://10.0.0.1/                               refused
file:///etc/passwd                             refused
```

Rebinding-style hosts — public names that resolve into loopback or private space — were also
refused:

```
http://localtest.me/            (resolves to ::1)     refused
http://127.0.0.1.nip.io/                              refused
http://10.0.0.1.nip.io/                               refused
http://spoofed.burpcollaborator.net/                  refused
```

Control run, to rule out the trivial explanation that plain `http://` is simply rejected:

```
http://example.com/                    200 OK, 167 chars
https://example.com/                   200 OK, 167 chars
http://www.iana.org/domains/example    200 OK, 1,018 chars
https://localtest.me/                  400 refused
```

So the refusals are **host-based, not scheme-based**. Caveat worth stating to security: we cannot
distinguish resolution-aware egress control from a named blocklist of well-known SSRF-bypass
domains. `localtest.me`, `nip.io`, and `burpcollaborator.net` are all plausible blocklist entries.
The exposure is materially reduced; it is not proven closed. Only Tavily can confirm the mechanism.

## Code-side observations (AI-Q)

- `sources/web_page_fetch/src/register.py` has **no direct-HTTP fallback**. Failure of
  `TavilyExtract` returns an error string to the agent; it never falls back to fetching the URL
  ourselves. This directly answers Rich Harang's hard rule: AI-Q still never visits a
  model-supplied URL.
- `_validate_url` (`register.py:119`) enforces only scheme and netloc. `http://127.0.0.1/` and
  `http://169.254.169.254/` pass our validation and are stopped solely by Tavily. That is
  currently sufficient because the refusal is on the vendor side, but it means our SSRF posture
  is entirely inherited. A local private-IP/loopback deny-list would make the guarantee ours
  rather than borrowed, and it is a small change.
- `query` is deliberately not forwarded to Extract (`register.py:183`), so there is no
  server-side relevance filtering; windowing is local. Volume reaching the model is bounded by
  `max_chars_per_page` (10,000) and `max_chars_per_call` (24,000), not by anything Tavily does.

## What this means for the review

The honest position is that Extract removes the mitigation Rich Harang credited, and adds no new
one. He accepted Search partly because chunking and reranking reduced the raw-content surface;
T5 measured that surface growing **14-25x** for the same page. SSRF, the concern he was most
absolute about, is the one that measures clean.

That points at his own proposed mitigation set rather than at a claim of existing coverage:

1. A guardrail NIM over fetched content. Endpoint-agnostic, applies identically to Extract, and
   is the mitigation he said would put things in good shape.
2. A source allow/deny list on fetchable URLs. Stronger, but it constrains exactly the pages that
   produced the F1 gains, so the trade needs to be explicit.

Open item for the Tavily channel (`C08H37T4LHZ`): confirm whether any content safety runs on
`/extract`, and whether URL validation resolves DNS or matches a blocklist.
