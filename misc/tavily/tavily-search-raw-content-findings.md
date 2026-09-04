# `include_raw_content` on Tavily Search: empirical security findings

Measured 2026-08-25 against the live `api.tavily.com/search` endpoint using the AI-Q production
key. Probe: `misc/tavily/security_probe3.py`. Companion to
[`tavily-extract-security-findings.md`](tavily-extract-security-findings.md), which covers
`/extract`.

Question: if we skip the fetch tool and instead flip `include_raw_content` on the Search call we
already make, does the search stay as secure as it is today?

**Answer: no. `include_raw_content` is Extract, delivered through Search.** The bytes are
identical, so it inherits the whole Extract risk profile and adds none of Extract's own mitigations
(there aren't any). It does avoid one distinct risk the fetch tool carries: the model never
chooses the URL.

## Summary

| Question | Finding |
| :-- | :-- |
| Is search `raw_content` the same as extract `raw_content`? | **Byte-identical.** Same SHA-1, same length, same page. |
| Does it keep Search's chunking mitigation? | **No.** Chunking is bypassed entirely: 4-16x the chunk volume per call. |
| Any content filtering Extract lacked? | **No.** Full jailbreak corpora returned verbatim through `/search`. |
| Do invisible-markup carriers survive? | **Yes** — `title=` attributes *and* `display:none` text. |
| Does the model gain control over which URL is read? | **No.** The ranker picks; the model supplies only a natural-language query. |
| Is the exposure bounded today? | **No.** `max_content_length` is applied to `content`, not `raw_content`. |

## T6a - Byte identity with Extract

Same page (`en.wikipedia.org/wiki/Prompt_injection`), one search call with
`include_raw_content: true`, one `/extract` call:

```
search  raw_content:   34,151 chars  sha1 20c676260e40bdd6
extract raw_content:   34,151 chars  sha1 20c676260e40bdd6
identical: True
```

This is the finding everything else follows from. There is no separate, safer search-side
rendition — `include_raw_content` calls the same extractor. Every measurement in the Extract
findings doc (no summarization, no redaction, no safety annotation, markdown with link targets
inlined) transfers unchanged.

Note also that `raw_content` is present as a **key in every result even when not requested** — it
is simply `null`. Nothing changes in the response shape, only in whether the field is populated.

## T6b - Volume, and where it is not bounded

Per-call totals across three unrelated queries, 5 results each:

| Query | chunks (`content`) | `raw_content` | multiple |
| :-- | --: | --: | --: |
| prompt injection attacks | 9,471 | 39,257 (`true`) / 135,064 (`"text"`) | 4x / **16x** |
| NVIDIA NeMo guardrails docs | 7,879 | 62,130 | 7x |
| ICILS 2023 report | 6,852 | 39,754 | 5x |

Single results reached 57,287 chars (`github.com/0xk1h0/chatgpt_dan`) and 95,387 chars
(a `next.gr` article, via Extract on the same URL).

The bound that matters is on our side, and it is currently in the wrong place.
`sources/tavily_web_search/src/register.py` applies `_truncate_content` to `doc.get("content")`
only. With `max_content_length: 1000` and `max_results: 5` (`configs/config_cli_default.yml:100`),
today's ceiling is ~5,000 chars of provider text per search. Populating `raw_content` without
extending the same cap to it raises that ceiling by 1-2 orders of magnitude — and
`advanced_web_search_tool` (`config_cli_default.yml:104`) sets no `max_content_length` at all.

Coverage is partial and non-deterministic: 3-5 of 5 results carried `raw_content`, varying by
query and between calls on the same query. Some URLs return empty raw content through both
endpoints. So this is not a reliable-content feature either — it is best-effort extraction.

## T7 - Hidden-markup carriers, and a correction to the Extract doc

Same page and method as Extract T2, on search `raw_content`:

| Carrier | Candidates on page | Present in output |
| :-- | --: | --: |
| `<!-- HTML comments -->` | 5 | **0** |
| `display:none` / `aria-hidden` | 1 | **1** |
| `title="..."` attributes | 40 | **24** |

Surviving samples: `'Type of attack in machine learning'` (display:none),
`'Prompt inyeksiyası – Azerbaijani'` (title attribute).

**Correction to the Extract findings doc.** T2 reported `display:none` candidates as
`1 -> 0 present`, i.e. stripped. That was a probe artifact: round two's regex required a matching
closing tag and text longer than 40 chars, which dropped Wikipedia's 34-char short-description
div. That div is genuinely `style="display:none"` in the source, and its text is genuinely in the
returned content — verified against the raw HTML. So `display:none` text is **not** stripped, and
the T2 claim that "the two classic carriers are stripped" should be narrowed to HTML comments
only. This applies to Extract as much as to Search, since the output is the same bytes.

Net: two of three non-rendered carriers reach the model verbatim.

## T8 - No content filtering on the Search path either

Benign natural-language queries, `include_raw_content: true`, adversarial pages returned in full:

| URL | Returned | Markers found |
| :-- | --: | :-- |
| `swisskyrepo/PayloadsAllTheThings` Prompt Injection README | 15,554 chars | ignore-previous, injected imperative, DAN |
| `0xk1h0/chatgpt_dan` | 57,287 chars | injected imperative, DAN |
| `rubend18/ChatGPT-Jailbreak-Prompts` (HF dataset) | 29,807 chars | injected imperative, DAN |
| `OWASP/www-community` prompt-injection page | 9,034 chars | ignore-previous |
| `TakSec/Prompt-Injection-Everywhere` | 6,530 chars | ignore-previous, injected imperative |

The queries were ordinary research phrasings, not URL requests. The envelope carries
`answer`, `follow_up_questions`, `images`, `query`, `request_id`, `response_time`, `results` — no
moderation verdict field, same as Extract. As before, the finding is the absence of an observable
safety signal, not a proven absence of safety.

## T9 - `true` vs `"markdown"` vs `"text"`

`true` and `"markdown"` produced identical output on every query tested — `true` is an alias for
markdown. `"text"` is a different rendition with different coverage; it was the widest in two of
three queries (5/5 results, up to 16x chunks) and narrower in the third. No variant reduces the
injection surface in a way worth relying on. A first single-query run suggested a large,
consistent gap between the variants; running three queries showed that was result-set drift, not
signal.

## How this compares to the fetch tool

Not a strict improvement, and not a strict regression — the risks trade:

| | `include_raw_content` on Search | `fetch_url_tool` (web_page_fetch) |
| :-- | :-- | :-- |
| Who chooses the URL | Tavily's ranker, from an NL query | **The model**, from prior context |
| Raw page text to the model | Yes, identical bytes | Yes, identical bytes |
| When it fires | **Every search**, unconditionally | Only when the agent decides to fetch |
| Volume ceiling (ours) | **None today** — cap applies to `content` only | `max_chars_per_page` 10,000 / `max_chars_per_call` 24,000 |

The model-supplied-URL concern that dominated the fetch-tool review does not apply here — that
is a real advantage. But it is the concern that measured **clean** at the vendor boundary anyway
(Extract T4). The concern that measured badly — raw adversarial content reaching the model with
no filtering — applies identically, fires on every search rather than selectively, and is
currently unbounded on our side.

So `include_raw_content` is not the conservative option. On aggregate volume of untrusted text
reaching the model, it is the more aggressive of the two.

## What to do if we adopt it

1. **Extend `_truncate_content` to `raw_content`** before anything else. A one-line-shaped change
   in `sources/tavily_web_search/src/register.py`; without it the exposure is unbounded and
   `advanced_web_search_tool` has no cap at all. Render it through `_render_document`'s existing
   `html.escape` path, which is already correct.
2. **Same guardrail NIM** proposed for the fetch tool. It is endpoint-agnostic and covers this
   path identically — this is the mitigation that actually addresses T8.
3. **Make it opt-in per tool instance**, not a global default, so `web_search_tool` (5 results,
   every step) and a deliberate deep-read path can carry different limits.
4. Prefer `"markdown"` over bare `true` in config for explicitness; they are the same thing today
   and the alias is undocumented behavior we should not depend on.

## Open items for the Tavily channel (`C08H37T4LHZ`)

- Confirm `include_raw_content: true` is a documented alias for `"markdown"`.
- Confirm `/search` with `include_raw_content` uses the same extractor as `/extract` (our
  SHA-1 match says yes; worth stated confirmation).
- Why is `raw_content` null for some ranked results? Is that extraction failure, robots policy,
  or a per-plan limit?
- Same question as the Extract doc: does any content safety run on either path?
