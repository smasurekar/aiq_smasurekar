# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from uuid import UUID

import pytest
from azure.core.exceptions import ServiceRequestError
from knowledge_layer.azure_ai_search.adapter import AzureAISearchIngestor
from knowledge_layer.azure_ai_search.adapter import AzureAISearchRetriever
from knowledge_layer.azure_ai_search.adapter import _build_index_schema
from knowledge_layer.azure_ai_search.adapter import _coerce_page_number
from knowledge_layer.azure_ai_search.adapter import _resolve_filenames
from knowledge_layer.azure_ai_search.adapter import _validate_index_name
from knowledge_layer.register import KnowledgeRetrievalConfig
from knowledge_layer.register import _format_results
from knowledge_layer.register import _setup_backend
from pydantic import SecretStr

from aiq_agent.knowledge import BaseIngestor
from aiq_agent.knowledge import BaseRetriever
from aiq_agent.knowledge import ContentType
from aiq_agent.knowledge import RetrievalResult
from aiq_agent.knowledge.factory import is_ingestor_registered
from aiq_agent.knowledge.factory import is_retriever_registered


def test_backend_registered_and_implements_sdk_contracts():
    assert is_ingestor_registered("azure_ai_search")
    assert is_retriever_registered("azure_ai_search")
    assert issubclass(AzureAISearchIngestor, BaseIngestor)
    assert issubclass(AzureAISearchRetriever, BaseRetriever)


def test_config_requires_endpoint():
    with pytest.raises(ValueError, match="azure_search_endpoint"):
        KnowledgeRetrievalConfig(backend="azure_ai_search")


def test_api_key_auth_requires_secret():
    with pytest.raises(ValueError, match="azure_search_api_key"):
        KnowledgeRetrievalConfig(
            backend="azure_ai_search",
            azure_search_endpoint="https://example.search.windows.net",
            azure_search_auth_mode="api_key",
        )


def test_setup_backend_preserves_secret_types(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_RETRIEVER_BACKEND", "llamaindex")
    monkeypatch.setenv("KNOWLEDGE_INGESTOR_BACKEND", "llamaindex")
    config = KnowledgeRetrievalConfig(
        backend="azure_ai_search",
        azure_search_endpoint="https://example.search.windows.net",
        azure_search_auth_mode="api_key",
        azure_search_api_key="test-search-key",
        embed_api_key="test-embed-key",
    )

    backend, backend_config = _setup_backend(config)

    assert backend == "azure_ai_search"
    assert isinstance(backend_config["api_key"], SecretStr)
    assert isinstance(backend_config["embed_api_key"], SecretStr)
    assert backend_config["embed_dim"] == 4096


@pytest.mark.parametrize(
    ("hit", "expected"),
    [
        ({"@search.reranker_score": 2.0, "@search.score": 0.9}, 0.5),
        ({"@search.reranker_score": 8.0}, 1.0),
        ({"@search.reranker_score": -1.0}, 0.0),
        ({"@search.score": 0.75}, 0.75),
    ],
)
def test_normalize_clamps_and_prefers_reranker_score(hit, expected):
    chunk = AzureAISearchRetriever.__new__(AzureAISearchRetriever).normalize(
        {"id": "chunk-1", "chunk": "text", "file_name": "report.pdf", **hit}
    )

    assert chunk.score == expected


def test_normalize_populates_citation_and_metadata():
    retriever = AzureAISearchRetriever.__new__(AzureAISearchRetriever)
    cited = retriever.normalize(
        {
            "id": "chunk-1",
            "chunk": "text",
            "file_name": "report.pdf",
            "page_number": "3",
            "doc_id": "doc-1",
            "metadata": '{"section":"intro"}',
        }
    )
    fallback = retriever.normalize({"page_number": 0})

    assert cited.display_citation == "report.pdf, p.3"
    assert cited.content_type is ContentType.TEXT
    assert cited.metadata == {"doc_id": "doc-1", "raw": '{"section":"intro"}'}
    assert fallback.content == ""
    assert fallback.file_name == "unknown"
    assert fallback.page_number is None
    assert fallback.display_citation == "unknown"
    UUID(fallback.chunk_id)


def test_shared_formatter_retains_source_and_citation_lines():
    chunk = AzureAISearchRetriever.__new__(AzureAISearchRetriever).normalize(
        {"id": "chunk-1", "chunk": "text", "file_name": "report.pdf", "page_number": 3}
    )
    result = RetrievalResult(query="query", backend="azure_ai_search", chunks=[chunk])

    formatted = _format_results(result, "query")

    assert "Source: report.pdf" in formatted
    assert "Citation: report.pdf, p.3" in formatted


@pytest.mark.asyncio
async def test_retrieve_builds_hybrid_semantic_search_request():
    class FakeEmbedding:
        def get_query_embedding(self, query):
            assert query == "hello"
            return [0.1, 0.2]

    class FakeClient:
        def search(self, **kwargs):
            self.kwargs = kwargs
            return [{"id": "chunk-1", "chunk": "answer", "file_name": "report.pdf"}]

    client = FakeClient()
    retriever = AzureAISearchRetriever.__new__(AzureAISearchRetriever)
    retriever.cfg = SimpleNamespace(use_hybrid=True, use_semantic_ranker=True)
    retriever._embedding = FakeEmbedding()
    retriever._get_client = lambda collection_name: client

    result = await retriever.retrieve("hello", "session-1", top_k=5, filters={"$filter": "doc_id eq 'doc-1'"})

    assert result.success
    assert [chunk.content for chunk in result.chunks] == ["answer"]
    assert client.kwargs["search_text"] == "hello"
    assert client.kwargs["query_type"] == "semantic"
    assert client.kwargs["semantic_configuration_name"] == "default-semantic"
    assert client.kwargs["filter"] == "doc_id eq 'doc-1'"
    assert client.kwargs["top"] == 5
    assert client.kwargs["select"] == ["id", "chunk", "doc_id", "file_name", "page_number", "metadata"]
    vector_query = client.kwargs["vector_queries"][0]
    assert vector_query.vector == [0.1, 0.2]
    assert vector_query.k_nearest_neighbors == 20
    assert vector_query.fields == "embedding"


def test_index_schema_uses_requested_vector_dimensions():
    schema = _build_index_schema("session-1", 4096)
    embedding = next(field for field in schema.fields if field.name == "embedding")

    assert embedding.vector_search_dimensions == 4096
    assert schema.semantic_search.default_configuration_name == "default-semantic"


def test_frontend_collection_names_are_valid():
    _validate_index_name("s_f031d9cf_123")

    with pytest.raises(ValueError, match="Invalid AI Search index name"):
        _validate_index_name("Invalid Name")


def test_filename_and_page_normalization(tmp_path):
    paths = [str(tmp_path / "tmp-one"), str(tmp_path / "tmp-two")]

    assert _resolve_filenames(paths, ["one.pdf", "two.docx"]) == ["one.pdf", "two.docx"]
    assert _resolve_filenames(paths, {paths[0]: "mapped.txt"}) == ["mapped.txt", "tmp-two"]
    assert _coerce_page_number("4") == 4
    assert _coerce_page_number("cover") is None


def test_service_connection_error_is_user_readable():
    message = AzureAISearchIngestor._translate_error(ServiceRequestError("connection refused"))

    assert message == "AI Search service unavailable: connection refused"
