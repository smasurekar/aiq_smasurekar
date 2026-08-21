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

"""Tests for deep researcher structured response contracts."""

import json

import pytest
from pydantic import ValidationError

from aiq_agent.agents.adaptive_researcher.models.subagent_contracts import AdaptiveResearchPlan
from aiq_agent.agents.deep_researcher.models import AnswerStrategy
from aiq_agent.agents.deep_researcher.models import Constraint
from aiq_agent.agents.deep_researcher.models import EvidenceJudgment
from aiq_agent.agents.deep_researcher.models import ResearchNotes
from aiq_agent.agents.deep_researcher.models import ResearchPlan
from aiq_agent.agents.deep_researcher.models import ResearchQuery
from aiq_agent.agents.deep_researcher.models import SourceRoutingPlan
from aiq_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_RESEARCH_QUERIES


def _answer_strategy() -> dict:
    return {
        "answer_type": "comparison",
        "title": "CUDA and OpenCL Trade-offs",
        "required_components": [
            {
                "id": "programming_model",
                "name": "Programming model",
                "description": "Compare kernel, memory, and execution models.",
            }
        ],
    }


def _task_analysis() -> dict:
    return {
        "user_intent": "Understand CUDA and OpenCL trade-offs.",
        "explicit_requirements": ["Compare CUDA and OpenCL"],
        "implicit_requirements": ["Cover ecosystem and portability"],
        "out_of_scope": ["General GPU purchasing advice"],
        "language": "English",
    }


def _research_query(**overrides) -> dict:
    payload = {
        "query": "CUDA OpenCL portability",
        "subqueries": [],
        "preferred_tools": ["web_search_tool"],
        "fallback_tools": [],
        "target_components": ["programming_model"],
        "rationale": "Supports the comparison component.",
    }
    payload.update(overrides)
    return payload


def test_research_plan_contract_validates_expected_shape():
    plan = ResearchPlan.model_validate(
        {
            "task_analysis": _task_analysis(),
            "answer_strategy": _answer_strategy(),
            "constraints": [
                {
                    "category": "content",
                    "constraint": "Compare portability, performance, and ecosystem maturity.",
                    "rationale": "These dimensions determine practical adoption.",
                }
            ],
            "queries": [
                {
                    "query": "CUDA OpenCL portability performance ecosystem comparison",
                    "subqueries": ["CUDA OpenCL portability", "CUDA OpenCL benchmark comparison"],
                    "preferred_tools": ["web_search_tool"],
                    "fallback_tools": [],
                    "target_components": ["programming_model"],
                    "rationale": "Supports the comparison component.",
                }
            ],
        }
    )

    assert plan.answer_strategy.required_components[0].id == "programming_model"
    assert plan.constraints[0].category == "content"
    assert plan.queries[0].target_components == ["programming_model"]
    assert plan.queries[0].subqueries == ["CUDA OpenCL portability", "CUDA OpenCL benchmark comparison"]
    assert plan.queries[0].preferred_tools == ["web_search_tool"]
    assert plan.queries[0].fallback_tools == []


def test_research_plan_contract_accepts_prediction_answer_type():
    answer_strategy = _answer_strategy()
    answer_strategy["answer_type"] = "prediction"
    answer_strategy["title"] = "Election Forecast"

    plan = ResearchPlan.model_validate(
        {
            "task_analysis": _task_analysis(),
            "answer_strategy": answer_strategy,
            "constraints": [],
            "queries": [
                {
                    "query": "Example election forecast evidence",
                    "subqueries": [],
                    "preferred_tools": ["polymarket_search_tool"],
                    "fallback_tools": [],
                    "target_components": ["programming_model"],
                    "rationale": "Supports the forecast evidence component.",
                }
            ],
        }
    )

    assert plan.answer_strategy.answer_type == "prediction"
    assert plan.queries[0].preferred_tools == ["polymarket_search_tool"]


def test_research_query_text_boundaries_are_enforced_per_nested_item():
    """Per-item bounds prevent deeply nested plan strings from evading aggregate quotas."""
    accepted = ResearchQuery.model_validate(
        _research_query(
            query="q" * 4096,
            subqueries=["s" * 2048],
        )
    )

    assert len(accepted.query) == 4096
    assert len(accepted.subqueries[0]) == 2048

    with pytest.raises(ValidationError):
        ResearchQuery.model_validate(_research_query(query="q" * 4097))
    with pytest.raises(ValidationError):
        ResearchQuery.model_validate(_research_query(subqueries=["s" * 2049]))


def test_research_plan_query_count_matches_shared_security_ceiling():
    """The structured schema matches the immutable per-job query-count ceiling."""
    base = {
        "task_analysis": _task_analysis(),
        "answer_strategy": _answer_strategy(),
        "constraints": [],
    }

    accepted = ResearchPlan.model_validate(
        {**base, "queries": [_research_query() for _ in range(DEFAULT_MAX_RESEARCH_QUERIES)]}
    )

    assert len(accepted.queries) == DEFAULT_MAX_RESEARCH_QUERIES
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(
            {**base, "queries": [_research_query() for _ in range(DEFAULT_MAX_RESEARCH_QUERIES + 1)]}
        )


def test_reduced_answer_strategy_contract_validates():
    strategy = AnswerStrategy.model_validate(_answer_strategy())

    assert strategy.answer_type == "comparison"
    assert strategy.title == "CUDA and OpenCL Trade-offs"
    assert strategy.required_components[0].id == "programming_model"


def test_constraint_contract_rejects_verification_field():
    with pytest.raises(ValidationError):
        Constraint.model_validate(
            {
                "category": "content",
                "constraint": "Compare portability, performance, and ecosystem maturity.",
                "rationale": "These dimensions determine practical adoption.",
                "verification": "Each dimension appears in the final answer.",
            }
        )


def test_research_notes_contract_validates_expected_shape():
    notes = ResearchNotes.model_validate(
        {
            "query_topic": "CUDA vs OpenCL portability",
            "target_components": ["programming_model"],
            "summary": "CUDA is NVIDIA-specific while OpenCL targets cross-vendor portability.",
            "findings": [
                {
                    "claim": "OpenCL is designed for cross-vendor heterogeneous compute.",
                    "evidence": "The source describes OpenCL as an open standard for heterogeneous platforms.",
                    "source_ids": [1],
                    "confidence": "high",
                    "caveats": ["Portability does not guarantee equal performance across vendors."],
                }
            ],
            "gaps": [
                {
                    "description": "Recent benchmark coverage is sparse.",
                    "impact": "Limits quantitative comparison.",
                    "suggested_follow_up_queries": ["CUDA OpenCL benchmark 2026"],
                }
            ],
            "sources": [
                {
                    "id": 1,
                    "title": "OpenCL Overview",
                    "source_type": "url",
                    "locator": "https://example.test/opencl",
                }
            ],
            "narrative_notes": "OpenCL offers broader portability, while CUDA typically has deeper vendor tooling.",
            "language": "English",
        }
    )

    assert notes.target_components == ["programming_model"]
    assert notes.findings[0].source_ids == [1]
    assert notes.sources[0].source_type == "url"
    assert notes.sources[0].locator == "https://example.test/opencl"
    assert notes.evidence_judgment is None


def test_research_notes_contract_accepts_evidence_judgment():
    notes = ResearchNotes.model_validate(
        {
            "query_topic": "CUDA vs OpenCL portability",
            "target_components": ["programming_model"],
            "summary": "CUDA is NVIDIA-specific while OpenCL targets portability.",
            "findings": [],
            "gaps": [],
            "sources": [],
            "narrative_notes": "OpenCL offers broader portability.",
            "language": "English",
            "evidence_judgment": {
                "relevance_score": 85,
                "confidence": "high",
                "rationale": "Directly supports the programming model component.",
            },
        }
    )

    assert notes.evidence_judgment is not None
    assert notes.evidence_judgment.relevance_score == 85
    assert notes.evidence_judgment.confidence == "high"


def test_evidence_judgment_contract_rejects_invalid_score():
    with pytest.raises(ValidationError):
        EvidenceJudgment.model_validate(
            {
                "relevance_score": 101,
                "confidence": "high",
                "rationale": "Score must stay within the configured range.",
            }
        )


def test_source_routing_plan_contract_validates_expected_shape():
    route = SourceRoutingPlan.model_validate(
        {
            "domain_id": "current_news",
            "domain_name": "Current News",
            "routing_reason": "The user asks for recent developments.",
            "recommendations": [
                {
                    "source_id": "news_search",
                    "tool_names": ["duckduckgo_news_search_tool"],
                    "priority": 1,
                    "rationale": "Best fit for recent news.",
                }
            ],
            "fallback_sources": [
                {
                    "source_id": "web_search",
                    "tool_names": ["web_search_tool"],
                    "priority": 2,
                    "rationale": "Broad web fallback.",
                }
            ],
            "planner_guidance": "Use news_search first, then web_search if coverage is weak.",
        }
    )

    assert route.domain_id == "current_news"
    assert route.recommendations[0].tool_names == ["duckduckgo_news_search_tool"]


def test_subagent_contracts_reject_extra_fields_and_old_plan_shape():
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(
            {
                "task_analysis": _task_analysis(),
                "answer_strategy": _answer_strategy(),
                "constraints": [],
                "queries": [],
                "unexpected": "value",
            }
        )

    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(
            {
                "task_analysis": _task_analysis(),
                "report_title": "Title",
                "report_toc": [],
                "constraints": [],
                "queries": [],
            }
        )

    with pytest.raises(ValidationError):
        ResearchNotes.model_validate(
            {
                "query_topic": "CUDA vs OpenCL portability",
                "target_sections": ["Programming Model Differences"],
                "summary": "Old field should fail.",
                "findings": [],
                "gaps": [],
                "sources": [],
                "narrative_notes": "",
                "language": "English",
            }
        )

    for removed_field, value in (
        ("assembly_instruction", "Synthesize evidence into a comparison."),
        ("selection_mode", "none"),
        ("expected_count", None),
        ("options", []),
    ):
        old_strategy = _answer_strategy()
        old_strategy[removed_field] = value
        with pytest.raises(ValidationError):
            AnswerStrategy.model_validate(old_strategy)


def _research_notes(**overrides) -> dict:
    payload = {
        "query_topic": "CUDA vs OpenCL portability",
        "target_components": ["programming_model"],
        "summary": "CUDA is NVIDIA-specific while OpenCL targets portability.",
        "findings": [],
        "gaps": [],
        "sources": [],
        "narrative_notes": "OpenCL offers broader portability.",
        "language": "English",
    }
    payload.update(overrides)
    return payload


def test_stringified_evidence_judgment_is_decoded():
    """The observed defect: a scalar nested model emitted as a JSON string instead of an object."""
    notes = ResearchNotes.model_validate(
        _research_notes(
            evidence_judgment=json.dumps(
                {
                    "relevance_score": 95,
                    "confidence": "high",
                    "rationale": "Comprehensive coverage of all 38 OECD countries.",
                }
            )
        )
    )

    assert isinstance(notes.evidence_judgment, EvidenceJudgment)
    assert notes.evidence_judgment.relevance_score == 95
    assert notes.evidence_judgment.confidence == "high"


def test_stringified_required_nested_plan_fields_are_decoded():
    """``answer_strategy`` is required and un-unioned, so the fix cannot key off optionality."""
    plan = ResearchPlan.model_validate(
        {
            "task_analysis": json.dumps(_task_analysis()),
            "answer_strategy": json.dumps(_answer_strategy()),
            "constraints": [],
            "queries": [_research_query()],
        }
    )

    assert plan.task_analysis.user_intent == "Understand CUDA and OpenCL trade-offs."
    assert plan.answer_strategy.answer_type == "comparison"
    assert plan.answer_strategy.required_components[0].id == "programming_model"


def test_stringified_nested_model_decoding_is_inherited_by_adaptive_plan():
    """Subclasses that retype fields still get the coercion from ``_StrictContract``."""
    plan = AdaptiveResearchPlan.model_validate(
        {
            "task_analysis": _task_analysis(),
            "answer_strategy": json.dumps(_answer_strategy()),
            "constraints": [],
            "queries": [_research_query(depth="high")],
        }
    )

    assert plan.answer_strategy.title == "CUDA and OpenCL Trade-offs"
    assert plan.queries[0].depth == "high"


def test_decoded_nested_model_still_validates_its_own_contract():
    """Decoding only removes the encoding layer; the payload underneath is validated as usual."""
    with pytest.raises(ValidationError) as excinfo:
        ResearchNotes.model_validate(
            _research_notes(
                evidence_judgment=json.dumps(
                    {"relevance_score": 101, "confidence": "high", "rationale": "Out of range."}
                )
            )
        )

    assert excinfo.value.errors()[0]["loc"] == ("evidence_judgment", "relevance_score")


def test_decoded_nested_model_still_forbids_extra_fields():
    """``extra="forbid"`` is untouched: a decoded object cannot smuggle in invented keys."""
    with pytest.raises(ValidationError) as excinfo:
        ResearchNotes.model_validate(
            _research_notes(
                evidence_judgment=json.dumps(
                    {
                        "relevance_score": 85,
                        "confidence": "high",
                        "rationale": "Useful.",
                        "invented_field": "value",
                    }
                )
            )
        )

    assert excinfo.value.errors()[0]["type"] == "extra_forbidden"


def test_genuine_string_fields_containing_json_are_left_alone():
    """Only nested-model fields are decoded, so prose carrying JSON fragments survives intact."""
    json_prose = 'The page returned {"relevance_score": 95, "confidence": "high"} verbatim.'

    notes = ResearchNotes.model_validate(
        _research_notes(
            summary=json_prose,
            narrative_notes=json_prose,
            evidence_judgment={"relevance_score": 85, "confidence": "high", "rationale": json_prose},
        )
    )

    assert notes.summary == json_prose
    assert notes.narrative_notes == json_prose
    assert notes.evidence_judgment is not None
    assert notes.evidence_judgment.rationale == json_prose


def test_string_field_holding_a_bare_json_object_is_left_alone():
    """A string field whose whole value is valid JSON is still a string, not a dict."""
    encoded_object = json.dumps({"note": "quoted from a fetched page"})

    notes = ResearchNotes.model_validate(_research_notes(narrative_notes=encoded_object))

    assert notes.narrative_notes == encoded_object


def test_stringified_list_of_objects_is_not_decoded():
    """List-valued nested fields are out of scope and keep failing as before."""
    with pytest.raises(ValidationError) as excinfo:
        ResearchNotes.model_validate(
            _research_notes(
                findings=json.dumps(
                    [
                        {
                            "claim": "OpenCL is cross-vendor.",
                            "evidence": "Open standard for heterogeneous platforms.",
                            "source_ids": [1],
                            "confidence": "high",
                            "caveats": [],
                        }
                    ]
                )
            )
        )

    assert excinfo.value.errors()[0]["type"] == "list_type"


@pytest.mark.parametrize(
    "encoded",
    [
        "85",
        '"high"',
        "null",
        "[{'relevance_score': 95}]",
        json.dumps([{"relevance_score": 95, "confidence": "high", "rationale": "A list, not an object."}]),
    ],
)
def test_strings_that_do_not_decode_to_an_object_fall_through_unchanged(encoded):
    """Non-object and malformed values keep pydantic's truthful ``model_type`` error."""
    with pytest.raises(ValidationError) as excinfo:
        ResearchNotes.model_validate(_research_notes(evidence_judgment=encoded))

    error = excinfo.value.errors()[0]
    assert error["loc"] == ("evidence_judgment",)
    assert error["type"] == "model_type"
    assert error["input"] == encoded


def test_malformed_json_is_never_swallowed():
    """A truncated blob -- the one captured value that does not re-parse -- stays a visible failure."""
    truncated = json.dumps(_answer_strategy())[:-40]

    with pytest.raises(ValidationError) as excinfo:
        ResearchPlan.model_validate(
            {
                "task_analysis": _task_analysis(),
                "answer_strategy": truncated,
                "constraints": [],
                "queries": [],
            }
        )

    assert excinfo.value.errors()[0]["type"] == "model_type"


def test_plan_packed_into_answer_strategy_stays_a_visible_failure():
    """The §3.4 relocation case: decoding exposes the misplaced keys instead of hiding them."""
    packed = json.dumps({**_answer_strategy(), "constraints": [], "queries": []})

    with pytest.raises(ValidationError) as excinfo:
        ResearchPlan.model_validate(
            {
                "task_analysis": _task_analysis(),
                "answer_strategy": packed,
                "constraints": [],
                "queries": [],
            }
        )

    error_types = {error["type"] for error in excinfo.value.errors()}
    assert error_types == {"extra_forbidden"}


def test_non_mapping_input_passes_through_the_decoder():
    """The validator must not assume a mapping: a before-validator sees whatever the caller sent."""
    with pytest.raises(ValidationError) as excinfo:
        ResearchNotes.model_validate(["not", "a", "mapping"])

    assert excinfo.value.errors()[0]["type"] == "model_type"


def test_caller_input_dict_is_not_mutated_by_decoding():
    """Decoding copies rather than rewriting the caller's payload, which callers still log."""
    payload = _research_notes(
        evidence_judgment=json.dumps({"relevance_score": 85, "confidence": "high", "rationale": "Useful."})
    )
    original = payload["evidence_judgment"]

    ResearchNotes.model_validate(payload)

    assert payload["evidence_judgment"] == original
