# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import re
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest
from azure.core.exceptions import ResourceNotFoundError
from azure.core.exceptions import ServiceRequestError
from azure.search.documents.indexes.models import SearchIndex
from knowledge_layer.azure_ai_search import adapter as azure_adapter
from knowledge_layer.azure_ai_search.adapter import AzureAISearchIngestor
from knowledge_layer.azure_ai_search.adapter import AzureAISearchRetriever
from knowledge_layer.azure_ai_search.adapter import _build_index_schema
from knowledge_layer.azure_ai_search.adapter import _coerce_page_number
from knowledge_layer.azure_ai_search.adapter import _decode_marker
from knowledge_layer.azure_ai_search.adapter import _encode_marker
from knowledge_layer.azure_ai_search.adapter import _index_name_for_collection
from knowledge_layer.azure_ai_search.adapter import _iter_index_batches
from knowledge_layer.azure_ai_search.adapter import _new_marker
from knowledge_layer.azure_ai_search.adapter import _resolve_filenames
from knowledge_layer.azure_ai_search.adapter import _validate_index_name
from knowledge_layer.azure_ai_search.adapter import _validate_index_schema
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
from aiq_agent.knowledge.schema import FileStatus


class FakeIndexingResult:
    def __init__(self, key: str, succeeded: bool = True, error_message: str | None = None):
        self.key = key
        self.succeeded = succeeded
        self.error_message = error_message


def _literal(filter_text: str | None, field: str, operator: str = "eq") -> str | None:
    if not filter_text:
        return None
    match = re.search(rf"{field} {operator} '((?:''|[^'])*)'", filter_text)
    return match.group(1).replace("''", "'") if match else None


class FakeSearchClient:
    def __init__(self):
        self.documents: dict[str, dict] = {}
        self.upload_batches: list[list[dict]] = []
        self.delete_batches: list[list[dict]] = []
        self.search_filters: list[str | None] = []
        self.fail_upload_ids: set[str] = set()
        self.delete_failures_remaining: dict[str, int] = {}

    def upload_documents(self, documents: list[dict]):
        self.upload_batches.append(documents)
        results = []
        for document in documents:
            key = document["id"]
            succeeded = key not in self.fail_upload_ids
            if succeeded:
                self.documents[key] = dict(document)
            results.append(FakeIndexingResult(key, succeeded, None if succeeded else "rejected"))
        return results

    def delete_documents(self, documents: list[dict]):
        self.delete_batches.append(documents)
        results = []
        for document in documents:
            key = document["id"]
            remaining = self.delete_failures_remaining.get(key, 0)
            succeeded = remaining <= 0
            if remaining > 0:
                self.delete_failures_remaining[key] = remaining - 1
            if succeeded:
                self.documents.pop(key, None)
            results.append(FakeIndexingResult(key, succeeded, None if succeeded else "retry"))
        return results

    def search(self, search_text="*", filter=None, select=None, order_by=None, top=None, **kwargs):
        del search_text, order_by, kwargs
        self.search_filters.append(filter)
        file_id = _literal(filter, "file_id")
        file_name = _literal(filter, "file_name")
        after_id = _literal(filter, "id", "gt")
        documents = sorted(self.documents.values(), key=lambda item: item["id"])
        if file_id is not None:
            documents = [document for document in documents if document.get("file_id") == file_id]
        if file_name is not None:
            documents = [document for document in documents if document.get("file_name") == file_name]
        if after_id is not None:
            documents = [document for document in documents if document["id"] > after_id]
        if top is not None:
            documents = documents[:top]
        if select:
            return [{key: document.get(key) for key in select} for document in documents]
        return [dict(document) for document in documents]

    def get_document_count(self):
        return len(self.documents)


class FakeIndexClient:
    def __init__(self):
        self.indexes: dict[str, SearchIndex] = {}
        self.race_on_create = False
        self.fail_create = False

    def get_index(self, name: str):
        if name not in self.indexes:
            raise ResourceNotFoundError("missing")
        return self.indexes[name]

    def create_index(self, index: SearchIndex):
        if self.race_on_create:
            self.indexes[index.name] = index
            raise RuntimeError("concurrent create")
        if self.fail_create:
            raise RuntimeError("create failed")
        self.indexes[index.name] = index
        return index

    def create_or_update_index(self, index: SearchIndex, **kwargs):
        del kwargs
        self.indexes[index.name] = index
        return index

    def list_indexes(self):
        return list(self.indexes.values())

    def list_index_names(self):
        return list(self.indexes)

    def delete_index(self, name: str):
        if name not in self.indexes:
            raise ResourceNotFoundError("missing")
        del self.indexes[name]


class FakeEmbedding:
    def __init__(self, dimensions: int = 4):
        self.dimensions = dimensions

    def get_query_embedding(self, query):
        del query
        return [0.1] * self.dimensions

    def get_text_embedding_batch(self, texts):
        return [[0.1] * self.dimensions for _ in texts]


class FakeNode:
    def __init__(self, text: str, page: str = "1"):
        self.text = text
        self.metadata = {"page_label": page}

    def get_content(self):
        return self.text


class FakeSplitter:
    def __init__(self, nodes: list[FakeNode]):
        self.nodes = nodes

    def get_nodes_from_documents(self, documents):
        del documents
        return self.nodes


def _config(**overrides):
    config = {
        "endpoint": "https://example.search.windows.net",
        "auth_mode": "api_key",
        "api_key": SecretStr("test-key"),
        "embed_model": "test-embed",
        "embed_dim": 4,
        "embed_base_url": "https://integrate.api.nvidia.com/v1",
        "use_hybrid": True,
        "use_semantic_ranker": True,
        "start_ttl_cleanup": False,
        "index_prefix": "aiq-test",
    }
    config.update(overrides)
    return config


def _ingestor(**overrides):
    ingestor = AzureAISearchIngestor(_config(**overrides))
    ingestor._index_client = FakeIndexClient()
    search_client = FakeSearchClient()
    ingestor._get_search_client = lambda collection_name: search_client
    return ingestor, search_client


def _install_reader(monkeypatch):
    class FakeReader:
        def __init__(self, input_files):
            self.input_files = input_files

        def load_data(self):
            return [object()]

    monkeypatch.setattr("llama_index.core.SimpleDirectoryReader", FakeReader)


def test_backend_registered_and_implements_sdk_contracts():
    assert is_ingestor_registered("azure_ai_search")
    assert is_retriever_registered("azure_ai_search")
    assert issubclass(AzureAISearchIngestor, BaseIngestor)
    assert issubclass(AzureAISearchRetriever, BaseRetriever)


def test_config_requires_endpoint_and_valid_hybrid_semantic_combination():
    with pytest.raises(ValueError, match="azure_search_endpoint"):
        KnowledgeRetrievalConfig(backend="azure_ai_search")
    with pytest.raises(ValueError, match="use_semantic_ranker"):
        KnowledgeRetrievalConfig(
            backend="azure_ai_search",
            azure_search_endpoint="https://example.search.windows.net",
            use_hybrid=False,
            use_semantic_ranker=True,
        )


def test_config_uses_shared_environment_defaults(monkeypatch):
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://env.search.windows.net")
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "env-search-key")
    monkeypatch.setenv("AIQ_AZURE_SEARCH_INDEX_PREFIX", "env-aiq")
    monkeypatch.setenv("AIQ_EMBED_BASE_URL", "https://embed.example.com/v1")
    monkeypatch.setenv("AIQ_EMBED_MODEL", "env-embed")
    monkeypatch.setenv("AIQ_EMBED_DIM", "8")

    config = KnowledgeRetrievalConfig(backend="azure_ai_search")
    backend, backend_config = _setup_backend(config)
    adapter_config = azure_adapter._coerce_config(None)

    assert backend == "azure_ai_search"
    assert backend_config["endpoint"] == "https://env.search.windows.net/"
    assert backend_config["auth_mode"] == "api_key"
    assert backend_config["api_key"].get_secret_value() == "env-search-key"
    assert backend_config["index_prefix"] == "env-aiq"
    assert backend_config["embed_base_url"] == "https://embed.example.com/v1"
    assert backend_config["embed_model"] == "env-embed"
    assert backend_config["embed_dim"] == 8
    assert adapter_config.endpoint == "https://env.search.windows.net"
    assert adapter_config.auth_mode == "api_key"


def test_api_key_auth_requires_secret():
    with pytest.raises(ValueError, match="azure_search_api_key"):
        KnowledgeRetrievalConfig(
            backend="azure_ai_search",
            azure_search_endpoint="https://example.search.windows.net",
            azure_search_auth_mode="api_key",
        )


def test_setup_backend_preserves_secrets_and_prefix(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_RETRIEVER_BACKEND", "llamaindex")
    monkeypatch.setenv("KNOWLEDGE_INGESTOR_BACKEND", "llamaindex")
    config = KnowledgeRetrievalConfig(
        backend="azure_ai_search",
        azure_search_endpoint="https://example.search.windows.net",
        azure_search_auth_mode="api_key",
        azure_search_api_key="test-search-key",
        embed_api_key="test-embed-key",
        azure_search_index_prefix="tenant-aiq",
    )

    backend, backend_config = _setup_backend(config)

    assert backend == "azure_ai_search"
    assert isinstance(backend_config["api_key"], SecretStr)
    assert isinstance(backend_config["embed_api_key"], SecretStr)
    assert backend_config["index_prefix"] == "tenant-aiq"


def test_index_names_are_official_stable_and_collision_safe():
    names = ["Tenant A", "tenant-a", "tenant/a", "TENANT+A"]
    indexes = [_index_name_for_collection(name, "AIQ Prod") for name in names]

    assert len(set(indexes)) == len(names)
    assert indexes[0] == _index_name_for_collection("Tenant A", "AIQ Prod")
    assert all(len(index) <= 128 for index in indexes)
    assert all(re.fullmatch(r"[a-z0-9-]+", index) for index in indexes)
    _validate_index_name(indexes[0])
    with pytest.raises(ValueError, match="Invalid Azure AI Search index name"):
        _validate_index_name("invalid_name")


def test_marker_and_schema_validation_reject_unowned_and_mismatched_indexes():
    cfg = SimpleNamespace(embed_model="test-embed", embed_dim=4, use_semantic_ranker=True)
    marker = _new_marker("docs", cfg, "Docs", {"tenant": "alpha"})
    index = _build_index_schema("aiq-docs-123456789abc", 4, _encode_marker(marker))

    assert _validate_index_schema(index, "docs", cfg)["metadata"] == {"tenant": "alpha"}
    index.description = "unmanaged"
    with pytest.raises(RuntimeError, match="not owned"):
        _validate_index_schema(index, "docs", cfg)
    index.description = _encode_marker(marker)
    embedding = next(field for field in index.fields if field.name == "embedding")
    embedding.vector_search_dimensions = 8
    with pytest.raises(RuntimeError, match="vector profile or dimensions"):
        _validate_index_schema(index, "docs", cfg)


def test_create_collection_handles_race_and_ignores_unmanaged_indexes():
    ingestor, _client = _ingestor()
    ingestor._index_client.race_on_create = True
    created = ingestor.create_collection("docs", description="Documents", metadata={"tenant": "alpha"})
    ingestor._index_client.indexes["unrelated"] = SearchIndex(name="unrelated", fields=[], description="other")

    assert created.name == "docs"
    assert created.metadata["tenant"] == "alpha"
    assert [collection.name for collection in ingestor.list_collections()] == ["docs"]


def test_create_collection_propagates_real_create_failure():
    ingestor, _client = _ingestor()
    ingestor._index_client.fail_create = True

    with pytest.raises(RuntimeError, match="create failed"):
        ingestor.create_collection("docs")


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


def test_normalize_parses_metadata_and_populates_citation():
    retriever = AzureAISearchRetriever.__new__(AzureAISearchRetriever)
    cited = retriever.normalize(
        {
            "id": "chunk-1",
            "chunk": "text",
            "file_id": "file-1",
            "file_name": "report.pdf",
            "page_number": "3",
            "metadata": '{"section":"intro"}',
        }
    )
    fallback = retriever.normalize({"page_number": 0})

    assert cited.display_citation == "report.pdf, p.3"
    assert cited.content_type is ContentType.TEXT
    assert cited.metadata == {"section": "intro", "file_id": "file-1"}
    assert fallback.content == ""
    assert fallback.file_name == "unknown"
    UUID(fallback.chunk_id)


def test_shared_formatter_retains_source_and_citation_lines():
    chunk = AzureAISearchRetriever.__new__(AzureAISearchRetriever).normalize(
        {"id": "chunk-1", "chunk": "text", "file_name": "report.pdf", "page_number": 3}
    )
    formatted = _format_results(RetrievalResult(query="query", backend="azure_ai_search", chunks=[chunk]), "query")

    assert "Source: report.pdf" in formatted
    assert "Citation: report.pdf, p.3" in formatted


@pytest.mark.asyncio
async def test_retrieve_builds_hybrid_semantic_search_request():
    class FakeClient:
        def search(self, **kwargs):
            self.kwargs = kwargs
            return [{"id": "chunk-1", "chunk": "answer", "file_name": "report.pdf"}]

    client = FakeClient()
    retriever = AzureAISearchRetriever.__new__(AzureAISearchRetriever)
    retriever.cfg = SimpleNamespace(use_hybrid=True, use_semantic_ranker=True)
    retriever._embedding = FakeEmbedding(2)
    retriever._get_client = lambda collection_name: client

    result = await retriever.retrieve("hello", "session-1", top_k=5, filters={"$filter": "file_id eq 'file-1'"})

    assert result.success
    assert client.kwargs["search_text"] == "hello"
    assert client.kwargs["query_type"] == "semantic"
    assert client.kwargs["filter"] == "file_id eq 'file-1'"
    assert client.kwargs["select"] == ["id", "chunk", "file_id", "file_name", "page_number", "metadata"]


def test_submit_job_uses_one_canonical_file_id(monkeypatch, tmp_path):
    class NoStartThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(azure_adapter.threading, "Thread", NoStartThread)
    path = tmp_path / "document.txt"
    path.write_text("content", encoding="utf-8")
    ingestor, _client = _ingestor()

    job_id = ingestor.submit_job([str(path)], "docs", {"original_filenames": ["original.txt"]})
    job = ingestor.get_job_status(job_id)
    file_id = job.file_details[0].file_id

    assert file_id in ingestor._files
    assert ingestor._files[file_id].file_name == "original.txt"


def test_batches_respect_action_count_and_payload_size():
    documents = [{"id": str(index), "chunk": "x" * 10} for index in range(5)]
    count_batches = list(_iter_index_batches(documents, "upload", max_actions=2, max_bytes=10_000))
    byte_batches = list(_iter_index_batches(documents, "upload", max_actions=100, max_bytes=120))

    assert [len(batch) for batch in count_batches] == [2, 2, 1]
    assert len(byte_batches) > 1
    with pytest.raises(ValueError, match="16 MiB"):
        list(_iter_index_batches([{"id": "large", "chunk": "x" * 500}], "upload", max_bytes=100))


def test_partial_upload_rolls_back_successful_actions():
    ingestor, client = _ingestor()
    documents = [{"id": f"file-{index}", "chunk": "text"} for index in range(3)]
    client.fail_upload_ids.add("file-1")

    with pytest.raises(RuntimeError, match="rejected upload"):
        ingestor._upload_documents(client, documents)

    assert client.documents == {}
    assert client.delete_batches


def test_delete_retries_per_document_failures():
    ingestor, client = _ingestor()
    client.documents["chunk-1"] = {"id": "chunk-1"}
    client.delete_failures_remaining["chunk-1"] = 1

    ingestor._delete_document_ids(client, ["chunk-1"])

    assert "chunk-1" not in client.documents
    assert len(client.delete_batches) == 2


def test_persistent_delete_failure_is_reported():
    ingestor, client = _ingestor()
    client.documents["chunk-1"] = {"id": "chunk-1"}
    client.delete_failures_remaining["chunk-1"] = 10

    with pytest.raises(RuntimeError, match="rejected delete"):
        ingestor._delete_document_ids(client, ["chunk-1"])

    assert "chunk-1" in client.documents


def test_pagination_is_exhaustive_and_file_ids_are_odata_escaped(monkeypatch):
    monkeypatch.setattr(azure_adapter, "_PAGE_SIZE", 2)
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    for index in range(5):
        client.documents[f"chunk-{index}"] = {
            "id": f"chunk-{index}",
            "file_id": f"file-{index}",
            "file_name": f"file-{index}.txt",
            "file_size": index,
            "uploaded_at": datetime.now(UTC),
            "ingested_at": datetime.now(UTC),
            "metadata": "{}",
        }

    assert len(ingestor.list_files("docs")) == 5
    crafted = "x' or file_id ne ''"
    assert ingestor._delete_file_documents(crafted, "docs") == 0
    assert any("file_id eq 'x'' or file_id ne '''''" in (item or "") for item in client.search_filters)


def test_same_name_replacement_removes_old_generation(monkeypatch, tmp_path):
    _install_reader(monkeypatch)
    path = tmp_path / "document.txt"
    path.write_text("new content", encoding="utf-8")
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    ingestor._embedding = FakeEmbedding()
    ingestor._splitter = FakeSplitter([FakeNode("new")])
    client.documents["old-00000000"] = {
        "id": "old-00000000",
        "chunk": "old",
        "embedding": [0.1] * 4,
        "file_id": "old",
        "file_name": "document.txt",
        "page_number": 1,
        "chunk_index": 0,
        "file_size": 100,
        "uploaded_at": datetime.now(UTC),
        "ingested_at": datetime.now(UTC),
        "metadata": "{}",
    }

    count = ingestor._process_file(
        path=str(path),
        collection_name="docs",
        file_id="new",
        file_name="document.txt",
        file_size=11,
        uploaded_at=datetime.now(UTC),
        metadata={"tenant": "alpha"},
    )

    assert count == 1
    assert set(client.documents) == {"new-00000000"}
    assert client.documents["new-00000000"]["metadata"] == '{"tenant":"alpha"}'


def test_failed_new_generation_preserves_old_generation(monkeypatch, tmp_path):
    _install_reader(monkeypatch)
    path = tmp_path / "document.txt"
    path.write_text("new content", encoding="utf-8")
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    ingestor._embedding = FakeEmbedding()
    ingestor._splitter = FakeSplitter([FakeNode("first"), FakeNode("second")])
    client.documents["old-00000000"] = {
        "id": "old-00000000",
        "file_id": "old",
        "file_name": "document.txt",
    }
    client.fail_upload_ids.add("new-00000001")

    with pytest.raises(RuntimeError, match="rejected upload"):
        ingestor._process_file(
            path=str(path),
            collection_name="docs",
            file_id="new",
            file_name="document.txt",
            file_size=11,
            uploaded_at=datetime.now(UTC),
            metadata={},
        )

    assert set(client.documents) == {"old-00000000"}


def test_replacement_cleanup_failure_retains_complete_new_generation(monkeypatch, tmp_path):
    _install_reader(monkeypatch)
    path = tmp_path / "document.txt"
    path.write_text("new content", encoding="utf-8")
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    ingestor._embedding = FakeEmbedding()
    ingestor._splitter = FakeSplitter([FakeNode("new")])
    client.documents["old-00000000"] = {
        "id": "old-00000000",
        "file_id": "old",
        "file_name": "document.txt",
    }
    client.delete_failures_remaining["old-00000000"] = 10

    with pytest.raises(RuntimeError, match="rejected delete"):
        ingestor._process_file(
            path=str(path),
            collection_name="docs",
            file_id="new",
            file_name="document.txt",
            file_size=11,
            uploaded_at=datetime.now(UTC),
            metadata={},
        )

    assert set(client.documents) == {"old-00000000", "new-00000000"}


def test_metadata_counts_and_status_round_trip():
    ingestor, client = _ingestor()
    ingestor.create_collection("docs", metadata={"tenant": "alpha"})
    now = datetime.now(UTC)
    for index in range(3):
        client.documents[f"file-1-{index:08d}"] = {
            "id": f"file-1-{index:08d}",
            "file_id": "file-1",
            "file_name": "report.pdf",
            "file_size": 42,
            "uploaded_at": now,
            "ingested_at": now,
            "metadata": '{"section":"finance"}',
        }

    files = ingestor.list_files("docs")
    collection = ingestor.get_collection("docs")

    assert files[0].file_id == "file-1"
    assert files[0].chunk_count == 3
    assert files[0].metadata == {"section": "finance"}
    assert ingestor.get_file_status("file-1", "docs") == files[0]
    assert collection.file_count == 1
    assert collection.chunk_count == 3
    assert collection.metadata["tenant"] == "alpha"


def test_failed_uploads_remain_visible():
    ingestor, _client = _ingestor()
    ingestor.create_collection("docs")
    ingestor._files["failed"] = azure_adapter.FileInfo(
        file_id="failed",
        file_name="failed.txt",
        collection_name="docs",
        status=FileStatus.FAILED,
        error_message="bad file",
    )

    files = ingestor.list_files("docs")

    assert [(item.file_id, item.status) for item in files] == [("failed", FileStatus.FAILED)]


def test_ttl_deletes_only_expired_owned_collection_and_clears_summary(monkeypatch):
    ingestor, _client = _ingestor()
    ingestor.create_collection("old")
    ingestor.create_collection("new")
    old_name = ingestor._physical_index_name("old")
    new_name = ingestor._physical_index_name("new")
    old_marker = _decode_marker(ingestor._index_client.indexes[old_name].description)
    new_marker = _decode_marker(ingestor._index_client.indexes[new_name].description)
    old_marker["updated_at"] = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    new_marker["updated_at"] = datetime.now(UTC).isoformat()
    ingestor._index_client.indexes[old_name].description = _encode_marker(old_marker)
    ingestor._index_client.indexes[new_name].description = _encode_marker(new_marker)
    ingestor._index_client.indexes["unrelated"] = SearchIndex(name="unrelated", fields=[], description="other")
    cleared = []
    monkeypatch.setattr(azure_adapter, "clear_collection_summaries", cleared.append)
    ingestor._ttl_hours = 24

    ingestor._cleanup_expired_collections()

    assert old_name not in ingestor._index_client.indexes
    assert new_name in ingestor._index_client.indexes
    assert "unrelated" in ingestor._index_client.indexes
    assert cleared == ["old"]


def test_collection_summary_clears_only_after_confirmed_delete(monkeypatch):
    ingestor, _client = _ingestor()
    ingestor.create_collection("docs")
    cleared = []
    monkeypatch.setattr(azure_adapter, "clear_collection_summaries", cleared.append)

    assert ingestor.delete_collection("docs")
    assert cleared == ["docs"]
    assert not ingestor.delete_collection("docs")
    assert cleared == ["docs"]


def test_file_summary_clears_only_after_confirmed_delete(monkeypatch):
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    client.documents["file-1-00000000"] = {
        "id": "file-1-00000000",
        "file_id": "file-1",
        "file_name": "report.pdf",
        "file_size": 1,
        "uploaded_at": datetime.now(UTC),
        "ingested_at": datetime.now(UTC),
        "metadata": "{}",
    }
    client.delete_failures_remaining["file-1-00000000"] = 10
    cleared = []
    monkeypatch.setattr(azure_adapter, "unregister_summary", lambda collection, file_name: cleared.append(file_name))

    with pytest.raises(RuntimeError, match="rejected delete"):
        ingestor.delete_file("file-1", "docs")

    assert cleared == []


@pytest.mark.asyncio
async def test_health_checks_run_sync_sdk_off_event_loop(monkeypatch):
    ingestor, _client = _ingestor()
    calls = []

    async def fake_to_thread(function, *args):
        calls.append(function)
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    assert await ingestor.health_check()
    assert len(calls) == 1


def test_index_schema_uses_requested_dimensions_and_fields():
    schema = _build_index_schema("aiq-session-123456789abc", 4096)
    fields = {field.name: field for field in schema.fields}

    assert fields["embedding"].vector_search_dimensions == 4096
    assert fields["file_id"].filterable
    assert fields["id"].sortable
    assert schema.semantic_search.default_configuration_name == "default-semantic"


def test_filename_page_normalization_and_error_translation(tmp_path):
    paths = [str(tmp_path / "tmp-one"), str(tmp_path / "tmp-two")]

    assert _resolve_filenames(paths, ["one.pdf", "two.docx"]) == ["one.pdf", "two.docx"]
    assert _resolve_filenames(paths, {paths[0]: "mapped.txt"}) == ["mapped.txt", "tmp-two"]
    assert _coerce_page_number("4") == 4
    assert _coerce_page_number("cover") is None
    assert (
        AzureAISearchIngestor._translate_error(ServiceRequestError("connection refused"))
        == "AI Search service unavailable: connection refused"
    )
