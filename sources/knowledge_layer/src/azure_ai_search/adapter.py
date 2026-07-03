# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from azure.core import MatchConditions
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ClientAuthenticationError
from azure.core.exceptions import HttpResponseError
from azure.core.exceptions import ResourceModifiedError
from azure.core.exceptions import ResourceNotFoundError
from azure.core.exceptions import ServiceRequestError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import HnswAlgorithmConfiguration
from azure.search.documents.indexes.models import HnswParameters
from azure.search.documents.indexes.models import SearchableField
from azure.search.documents.indexes.models import SearchField
from azure.search.documents.indexes.models import SearchFieldDataType
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.indexes.models import SemanticConfiguration
from azure.search.documents.indexes.models import SemanticField
from azure.search.documents.indexes.models import SemanticPrioritizedFields
from azure.search.documents.indexes.models import SemanticSearch
from azure.search.documents.indexes.models import SimpleField
from azure.search.documents.indexes.models import VectorSearch
from azure.search.documents.indexes.models import VectorSearchAlgorithmMetric
from azure.search.documents.indexes.models import VectorSearchProfile
from azure.search.documents.models import VectorizedQuery

from aiq_agent.knowledge import BaseIngestor
from aiq_agent.knowledge import BaseRetriever
from aiq_agent.knowledge import Chunk
from aiq_agent.knowledge import ContentType
from aiq_agent.knowledge import FileProgress
from aiq_agent.knowledge import IngestionJobStatus
from aiq_agent.knowledge import JobState
from aiq_agent.knowledge import RetrievalResult
from aiq_agent.knowledge import clear_collection_summaries
from aiq_agent.knowledge import register_ingestor
from aiq_agent.knowledge import register_retriever
from aiq_agent.knowledge import register_summary
from aiq_agent.knowledge import unregister_summary
from aiq_agent.knowledge.base import CollectionInfo
from aiq_agent.knowledge.base import FileInfo
from aiq_agent.knowledge.base import TTLCleanupMixin
from aiq_agent.knowledge.schema import FileStatus

logger = logging.getLogger(__name__)

_BACKEND_NAME = "azure_ai_search"
_SEMANTIC_CONFIG = "default-semantic"
_SCHEMA_VERSION = 1
_MARKER_PREFIX = "aiq.azure_ai_search:"
_MAX_INDEX_NAME_LENGTH = 128
_MAX_BATCH_ACTIONS = 1000
_MAX_BATCH_BYTES = 16 * 1024 * 1024
_PAGE_SIZE = 1000
_DELETE_ATTEMPTS = 3

COLLECTION_TTL_HOURS = float(os.environ.get("AIQ_COLLECTION_TTL_HOURS", "24"))
TTL_CLEANUP_INTERVAL_SECONDS = int(os.environ.get("AIQ_TTL_CLEANUP_INTERVAL_SECONDS", "3600"))


def _coerce_config(config: dict[str, Any] | None) -> SimpleNamespace:
    """Apply adapter defaults so direct factory usage matches YAML usage."""
    provided = config or {}
    values: dict[str, Any] = {
        "endpoint": os.environ.get("AZURE_SEARCH_ENDPOINT"),
        "api_key": os.environ.get("AZURE_SEARCH_API_KEY"),
        "embed_base_url": os.environ.get("AIQ_EMBED_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        "embed_model": os.environ.get("AIQ_EMBED_MODEL", "nvidia/nv-embed-v1"),
        "embed_dim": int(os.environ.get("AIQ_EMBED_DIM", "4096")),
        "embed_api_key": None,
        "use_hybrid": True,
        "use_semantic_ranker": False,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "summary_max_chars": 1000,
        "collection_name": "default",
        "cleanup_files": True,
        "generate_summary": False,
        "summary_llm": None,
        "index_prefix": os.environ.get("AIQ_AZURE_SEARCH_INDEX_PREFIX", "aiq"),
        "start_ttl_cleanup": True,
    }
    values.update(provided)
    values["auth_mode"] = provided.get(
        "auth_mode",
        "api_key" if values["api_key"] else "managed_identity",
    )
    if not values["endpoint"]:
        raise ValueError("Azure AI Search configuration requires `endpoint`")
    if values["auth_mode"] not in {"managed_identity", "api_key"}:
        raise ValueError("Azure AI Search auth_mode must be 'managed_identity' or 'api_key'")
    if values["chunk_overlap"] >= values["chunk_size"]:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not values["use_hybrid"] and values["use_semantic_ranker"]:
        raise ValueError("use_semantic_ranker=true requires use_hybrid=true")
    return SimpleNamespace(**values)


def _build_search_credential(cfg: SimpleNamespace):
    """Pick the Azure SDK credential without exposing secret values."""
    if cfg.auth_mode == "api_key":
        api_key = _secret_value(cfg.api_key)
        if api_key is None:
            raise ValueError("auth_mode=api_key requires the `api_key` field to be set")
        return AzureKeyCredential(api_key)
    return DefaultAzureCredential()


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return str(value)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def _sanitize_index_part(value: str, fallback: str = "default") -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or fallback


def _index_name_for_collection(collection_name: str, prefix: str = "aiq") -> str:
    """Map a logical collection to an official, collision-safe Azure index name."""
    prefix_part = _sanitize_index_part(prefix, "aiq")[:48].rstrip("-") or "aiq"
    collection_part = _sanitize_index_part(collection_name)
    suffix = uuid.uuid5(uuid.NAMESPACE_URL, f"{prefix}\0{collection_name}").hex[:12]
    available = _MAX_INDEX_NAME_LENGTH - len(prefix_part) - len(suffix) - 2
    collection_part = collection_part[:available].rstrip("-") or "default"
    return f"{prefix_part}-{collection_part}-{suffix}"


def _validate_index_name(name: str) -> None:
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,126}[a-z0-9])?", name) or "--" in name:
        raise ValueError(
            f"Invalid Azure AI Search index name {name!r}. Names must use lowercase letters, digits, and single "
            "hyphens; be 2-128 characters; and start and end with a letter or digit."
        )


def _encode_marker(marker: dict[str, Any]) -> str:
    return _MARKER_PREFIX + json.dumps(marker, separators=(",", ":"), sort_keys=True)


def _decode_marker(description: str | None) -> dict[str, Any] | None:
    if not description or not description.startswith(_MARKER_PREFIX):
        return None
    try:
        marker = json.loads(description[len(_MARKER_PREFIX) :])
    except (TypeError, ValueError):
        return None
    return marker if isinstance(marker, dict) else None


def _new_marker(
    collection_name: str,
    cfg: SimpleNamespace,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = _utc_now().isoformat()
    marker = {
        "backend": _BACKEND_NAME,
        "schema_version": _SCHEMA_VERSION,
        "collection_name": collection_name,
        "description": description,
        "metadata": metadata or {},
        "embedding_model": cfg.embed_model,
        "embedding_dim": cfg.embed_dim,
        "created_at": now,
        "updated_at": now,
    }
    _encode_marker(marker)  # Validate JSON serializability before any service mutation.
    return marker


def _odata_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _and_filter(*filters: str | None) -> str | None:
    values = [f"({value})" for value in filters if value]
    return " and ".join(values) or None


def _parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        logger.warning("Ignoring malformed Azure AI Search metadata JSON")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _iter_index_batches(
    documents: list[dict[str, Any]],
    action: str,
    max_actions: int = _MAX_BATCH_ACTIONS,
    max_bytes: int = _MAX_BATCH_BYTES,
) -> Iterator[list[dict[str, Any]]]:
    """Batch actions below both Azure's count and serialized payload limits."""
    batch: list[dict[str, Any]] = []
    batch_bytes = len(b'{"value":[]}')
    for document in documents:
        payload = {"@search.action": action, **document}
        action_bytes = (
            len(json.dumps(payload, default=_json_default, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            + 1
        )
        if action_bytes + len(b'{"value":[]}') > max_bytes:
            raise ValueError(f"Azure AI Search {action} action exceeds the 16 MiB request limit")
        if batch and (len(batch) >= max_actions or batch_bytes + action_bytes > max_bytes):
            yield batch
            batch = []
            batch_bytes = len(b'{"value":[]}')
        batch.append(document)
        batch_bytes += action_bytes
    if batch:
        yield batch


def _indexing_outcome(results: list[Any], expected: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    expected_keys = [str(document["id"]) for document in expected]
    by_key = {str(getattr(result, "key", "")): result for result in results}
    succeeded: list[str] = []
    failures: list[str] = []
    for key in expected_keys:
        result = by_key.get(key)
        if result is not None and getattr(result, "succeeded", False):
            succeeded.append(key)
        else:
            message = getattr(result, "error_message", None) if result is not None else "missing result"
            failures.append(f"{key}: {message or 'rejected'}")
    return succeeded, failures


def _build_index_schema(name: str, embed_dim: int, description: str | None = None) -> SearchIndex:
    return SearchIndex(
        name=name,
        description=description,
        fields=[
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
                sortable=True,
            ),
            SearchableField(name="chunk", analyzer_name="standard.lucene"),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=embed_dim,
                vector_search_profile_name="hnsw-profile",
            ),
            SimpleField(name="file_id", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchableField(name="file_name", filterable=True, sortable=True),
            SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="file_size", type=SearchFieldDataType.Int64, filterable=True),
            SimpleField(name="uploaded_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="ingested_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="metadata", type=SearchFieldDataType.String),
        ],
        vector_search=VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-default",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                ),
            ],
            profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-default")],
        ),
        semantic_search=SemanticSearch(
            default_configuration_name=_SEMANTIC_CONFIG,
            configurations=[
                SemanticConfiguration(
                    name=_SEMANTIC_CONFIG,
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="file_name"),
                        content_fields=[SemanticField(field_name="chunk")],
                    ),
                ),
            ],
        ),
    )


def _validate_index_schema(index: SearchIndex, collection_name: str, cfg: SimpleNamespace) -> dict[str, Any]:
    marker = _decode_marker(index.description)
    if marker is None:
        raise RuntimeError(f"Azure AI Search index {index.name!r} is not owned by AI-Q")
    if marker.get("backend") != _BACKEND_NAME or marker.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError(f"Azure AI Search index {index.name!r} has an incompatible AI-Q ownership marker")
    if marker.get("collection_name") != collection_name:
        raise RuntimeError(
            f"Azure AI Search index {index.name!r} belongs to collection {marker.get('collection_name')!r}, "
            f"not {collection_name!r}"
        )
    if marker.get("embedding_dim") != cfg.embed_dim or marker.get("embedding_model") != cfg.embed_model:
        raise RuntimeError(
            f"Azure AI Search index {index.name!r} embedding configuration does not match "
            f"{cfg.embed_model!r}/{cfg.embed_dim}"
        )

    fields = {field.name: field for field in index.fields}
    required_types = {
        "id": SearchFieldDataType.String,
        "chunk": SearchFieldDataType.String,
        "embedding": SearchFieldDataType.Collection(SearchFieldDataType.Single),
        "file_id": SearchFieldDataType.String,
        "file_name": SearchFieldDataType.String,
        "page_number": SearchFieldDataType.Int32,
        "chunk_index": SearchFieldDataType.Int32,
        "file_size": SearchFieldDataType.Int64,
        "uploaded_at": SearchFieldDataType.DateTimeOffset,
        "ingested_at": SearchFieldDataType.DateTimeOffset,
        "metadata": SearchFieldDataType.String,
    }
    missing = [name for name in required_types if name not in fields]
    mismatched = [
        name for name, field_type in required_types.items() if name in fields and fields[name].type != field_type
    ]
    if missing or mismatched:
        raise RuntimeError(
            f"Azure AI Search index {index.name!r} schema mismatch: missing={missing}, wrong_type={mismatched}"
        )
    if not fields["id"].key or not fields["id"].filterable or not fields["id"].sortable:
        raise RuntimeError(f"Azure AI Search index {index.name!r} requires id to be key/filterable/sortable")
    if not fields["file_id"].filterable or not fields["file_name"].filterable:
        raise RuntimeError(f"Azure AI Search index {index.name!r} requires filterable file identity fields")
    embedding = fields["embedding"]
    if embedding.vector_search_dimensions != cfg.embed_dim or embedding.vector_search_profile_name != "hnsw-profile":
        raise RuntimeError(f"Azure AI Search index {index.name!r} vector profile or dimensions do not match")
    profile_names = {profile.name for profile in (index.vector_search.profiles if index.vector_search else [])}
    if "hnsw-profile" not in profile_names:
        raise RuntimeError(f"Azure AI Search index {index.name!r} is missing hnsw-profile")
    semantic_names = {
        semantic.name for semantic in (index.semantic_search.configurations if index.semantic_search else [])
    }
    if cfg.use_semantic_ranker and (
        index.semantic_search is None
        or index.semantic_search.default_configuration_name != _SEMANTIC_CONFIG
        or _SEMANTIC_CONFIG not in semantic_names
    ):
        raise RuntimeError(f"Azure AI Search index {index.name!r} semantic configuration does not match")
    return marker


class _AzureIndexMixin:
    cfg: SimpleNamespace
    _credential: Any
    _embedding: Any
    _index_client: SearchIndexClient
    _search_clients: dict[str, SearchClient]

    def _initialize_azure(self, config: dict[str, Any]) -> None:
        self.cfg = _coerce_config(config)
        self._credential = _build_search_credential(self.cfg)
        self._index_client = SearchIndexClient(endpoint=str(self.cfg.endpoint), credential=self._credential)
        self._search_clients = {}
        self._embedding = None

    @property
    def embedding(self):
        if self._embedding is None:
            from llama_index.embeddings.nvidia import NVIDIAEmbedding

            self._embedding = NVIDIAEmbedding(
                model=self.cfg.embed_model,
                base_url=str(self.cfg.embed_base_url),
                api_key=_secret_value(self.cfg.embed_api_key),
            )
        return self._embedding

    def _physical_index_name(self, collection_name: str) -> str:
        name = _index_name_for_collection(collection_name, self.cfg.index_prefix)
        _validate_index_name(name)
        return name

    def _get_search_client(self, collection_name: str) -> SearchClient:
        if collection_name not in self._search_clients:
            self._search_clients[collection_name] = self._index_client.get_search_client(
                self._physical_index_name(collection_name)
            )
        return self._search_clients[collection_name]


@register_retriever(_BACKEND_NAME)
class AzureAISearchRetriever(_AzureIndexMixin, BaseRetriever):
    """Hybrid and semantic-ranked retriever backed by owned Azure indexes."""

    backend_name = _BACKEND_NAME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._initialize_azure(self.config)
        self._validated: set[str] = set()

    def _get_client(self, collection_name: str) -> SearchClient:
        index_name = self._physical_index_name(collection_name)
        if collection_name not in self._validated:
            index = self._index_client.get_index(index_name)
            _validate_index_schema(index, collection_name, self.cfg)
            self._validated.add(collection_name)
        return self._get_search_client(collection_name)

    async def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        return await asyncio.to_thread(self._retrieve_sync, query, collection_name, top_k, filters)

    def _retrieve_sync(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> RetrievalResult:
        try:
            query_vector = self.embedding.get_query_embedding(query)
            client = self._get_client(collection_name)
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=max(top_k * 3, 20),
                fields="embedding",
            )
            search_params: dict[str, Any] = {
                "vector_queries": [vector_query],
                "top": top_k,
                "select": ["id", "chunk", "file_id", "file_name", "page_number", "metadata"],
            }
            if self.cfg.use_hybrid:
                search_params["search_text"] = query
            if self.cfg.use_semantic_ranker:
                search_params["query_type"] = "semantic"
                search_params["semantic_configuration_name"] = _SEMANTIC_CONFIG
            if filters and isinstance(filters, dict) and (odata_filter := filters.get("$filter")):
                search_params["filter"] = odata_filter

            chunks = [self.normalize(hit) for hit in client.search(**search_params)]
            return RetrievalResult(query=query, backend=_BACKEND_NAME, chunks=chunks, success=True)
        except ResourceNotFoundError:
            message = f"AI-Q Azure AI Search collection {collection_name!r} not found"
        except ClientAuthenticationError as error:
            message = f"AI Search authentication failed: {error!s}"
        except ServiceRequestError as error:
            message = f"AI Search service unavailable: {error!s}"
        except HttpResponseError as error:
            message = f"AI Search request failed: {error.status_code} {error.reason or error!s}"
        except Exception as error:  # noqa: BLE001
            message = f"Unexpected error during retrieval: {error!s}"
            logger.exception("Azure AI Search retrieval failed")
        return RetrievalResult(query=query, backend=_BACKEND_NAME, chunks=[], success=False, error_message=message)

    def normalize(self, raw_result: Any) -> Chunk:
        chunk_id = raw_result.get("id") or str(uuid.uuid4())
        content = raw_result.get("chunk") or ""
        file_name = raw_result.get("file_name") or "unknown"
        page_number = _coerce_page_number(raw_result.get("page_number"))
        rerank_score = raw_result.get("@search.reranker_score")
        search_score = raw_result.get("@search.score") or 0.0
        score = float(rerank_score) / 4.0 if rerank_score is not None else float(search_score)
        score = min(max(score, 0.0), 1.0)
        metadata = _parse_metadata(raw_result.get("metadata"))
        if file_id := raw_result.get("file_id"):
            metadata["file_id"] = file_id
        return Chunk(
            chunk_id=str(chunk_id),
            content=str(content),
            score=score,
            file_name=str(file_name),
            page_number=page_number,
            display_citation=f"{file_name}, p.{page_number}" if page_number else str(file_name),
            content_type=ContentType.TEXT,
            metadata=metadata,
        )

    async def health_check(self) -> bool:
        def _check() -> bool:
            list(self._index_client.list_index_names())
            return True

        try:
            return await asyncio.to_thread(_check)
        except Exception:  # noqa: BLE001
            logger.exception("Azure AI Search retriever health_check failed")
            return False


@register_ingestor(_BACKEND_NAME)
class AzureAISearchIngestor(TTLCleanupMixin, _AzureIndexMixin, BaseIngestor):
    """Parse, embed, and persist documents in owned Azure AI Search indexes."""

    backend_name = _BACKEND_NAME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._initialize_azure(self.config)
        self._splitter = None
        self._summary_llm = self.cfg.summary_llm
        self._jobs_lock = threading.RLock()
        self._jobs: dict[str, IngestionJobStatus] = {}
        self._files: dict[str, FileInfo] = {}
        if self.cfg.start_ttl_cleanup:
            self._start_ttl_cleanup_task(COLLECTION_TTL_HOURS, TTL_CLEANUP_INTERVAL_SECONDS)

    @property
    def splitter(self):
        if self._splitter is None:
            from llama_index.core.node_parser import SentenceSplitter

            self._splitter = SentenceSplitter(chunk_size=self.cfg.chunk_size, chunk_overlap=self.cfg.chunk_overlap)
        return self._splitter

    def _get_owned_index(self, collection_name: str) -> tuple[SearchIndex, dict[str, Any]]:
        index = self._index_client.get_index(self._physical_index_name(collection_name))
        return index, _validate_index_schema(index, collection_name, self.cfg)

    def _ensure_index(
        self,
        collection_name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[SearchIndex, dict[str, Any]]:
        index_name = self._physical_index_name(collection_name)
        try:
            return self._get_owned_index(collection_name)
        except ResourceNotFoundError:
            pass

        marker = _new_marker(collection_name, self.cfg, description, metadata)
        schema = _build_index_schema(index_name, self.cfg.embed_dim, _encode_marker(marker))
        try:
            index = self._index_client.create_index(schema)
        except Exception as create_error:  # noqa: BLE001
            try:
                index = self._index_client.get_index(index_name)
            except ResourceNotFoundError:
                raise create_error
        return index, _validate_index_schema(index, collection_name, self.cfg)

    def _update_marker(self, collection_name: str, **updates: Any) -> dict[str, Any]:
        for attempt in range(3):
            index, marker = self._get_owned_index(collection_name)
            marker.update(updates)
            index.description = _encode_marker(marker)
            try:
                self._index_client.create_or_update_index(index, match_condition=MatchConditions.IfNotModified)
                return marker
            except (ResourceModifiedError, HttpResponseError) as error:
                if getattr(error, "status_code", None) != 412 or attempt == 2:
                    raise
        raise RuntimeError(f"Failed to update metadata for collection {collection_name!r}")

    def _update_collection_timestamp(self, collection_name: str) -> None:
        self._update_marker(collection_name, updated_at=_utc_now().isoformat())

    def submit_job(
        self,
        file_paths: list[str],
        collection_name: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        job_config = {**self.config, **(config or {})}
        original_filenames = _resolve_filenames(file_paths, job_config.get("original_filenames"))
        validated = [(path, original_filenames[index]) for index, path in enumerate(file_paths) if Path(path).is_file()]
        file_metadata = job_config.get("metadata") or {}
        _encode_marker({"metadata": file_metadata})

        if not validated:
            with self._jobs_lock:
                self._jobs[job_id] = IngestionJobStatus(
                    job_id=job_id,
                    status=JobState.FAILED,
                    collection_name=collection_name,
                    backend=_BACKEND_NAME,
                    submitted_at=_utc_now(),
                    completed_at=_utc_now(),
                    total_files=len(file_paths),
                    error_message="No valid file paths provided",
                )
            return job_id

        submitted_at = _utc_now()
        file_progress: list[FileProgress] = []
        for path, file_name in validated:
            file_id = str(uuid.uuid4())
            file_progress.append(FileProgress(file_id=file_id, file_name=file_name, status=FileStatus.UPLOADING))
            with self._jobs_lock:
                self._files[file_id] = FileInfo(
                    file_id=file_id,
                    file_name=file_name,
                    collection_name=collection_name,
                    status=FileStatus.UPLOADING,
                    file_size=Path(path).stat().st_size,
                    uploaded_at=submitted_at,
                    metadata={**file_metadata, "job_id": job_id},
                )

        with self._jobs_lock:
            self._jobs[job_id] = IngestionJobStatus(
                job_id=job_id,
                status=JobState.PENDING,
                collection_name=collection_name,
                backend=_BACKEND_NAME,
                submitted_at=submitted_at,
                total_files=len(validated),
                file_details=file_progress,
            )

        threading.Thread(
            target=self._process_job,
            args=(job_id, [path for path, _ in validated], collection_name, job_config),
            daemon=True,
            name=f"aiq-azure-search-ingest-{job_id[:8]}",
        ).start()
        return job_id

    def get_job_status(self, job_id: str) -> IngestionJobStatus:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job.model_copy(deep=True)
        return IngestionJobStatus(
            job_id=job_id,
            status=JobState.FAILED,
            collection_name="",
            backend=_BACKEND_NAME,
            submitted_at=_utc_now(),
            completed_at=_utc_now(),
            error_message="Job not found",
        )

    def _process_job(
        self,
        job_id: str,
        file_paths: list[str],
        collection_name: str,
        config: dict[str, Any],
    ) -> None:
        cleanup = bool(config.get("cleanup_files", self.cfg.cleanup_files))
        self._update_job(job_id, status=JobState.PROCESSING, started_at=_utc_now())
        try:
            self._ensure_index(collection_name)
        except Exception as error:  # noqa: BLE001
            self._fail_job(job_id, f"Failed to ensure collection {collection_name!r}: {error!s}")
            if cleanup:
                self._cleanup_paths(file_paths)
            return

        failed = 0
        for index, path in enumerate(file_paths):
            job = self.get_job_status(job_id)
            detail = job.file_details[index]
            tracked = self._files[detail.file_id]
            self._update_file_progress(job_id, index, status=FileStatus.INGESTING)
            try:
                chunk_count = self._process_file(
                    path=path,
                    collection_name=collection_name,
                    file_id=detail.file_id,
                    file_name=detail.file_name,
                    file_size=tracked.file_size or 0,
                    uploaded_at=tracked.uploaded_at or _utc_now(),
                    metadata={key: value for key, value in tracked.metadata.items() if key != "job_id"},
                )
                self._update_file_progress(
                    job_id,
                    index,
                    status=FileStatus.SUCCESS,
                    progress_percent=100.0,
                    chunks_created=chunk_count,
                )
            except Exception as error:  # noqa: BLE001
                failed += 1
                message = self._translate_error(error)
                self._update_file_progress(job_id, index, status=FileStatus.FAILED, error_message=message)
                logger.exception("Failed to ingest %s", detail.file_name)
            finally:
                if cleanup:
                    self._cleanup_paths([path])
                self._update_job(job_id, processed_files=index + 1)

        if failed == len(file_paths):
            self._fail_job(job_id, f"All {failed} file(s) failed to ingest")
        else:
            self._update_job(job_id, status=JobState.COMPLETED, completed_at=_utc_now())

    def _process_file(
        self,
        *,
        path: str,
        collection_name: str,
        file_id: str,
        file_name: str,
        file_size: int,
        uploaded_at: datetime,
        metadata: dict[str, Any],
    ) -> int:
        from llama_index.core import SimpleDirectoryReader

        old_file_ids = self._find_file_ids_by_name(file_name, collection_name) - {file_id}
        documents = SimpleDirectoryReader(input_files=[path]).load_data()
        if not documents:
            raise ValueError(f"No content extracted from {file_name}")
        nodes = self.splitter.get_nodes_from_documents(documents)
        if not nodes:
            raise ValueError(f"Chunking produced 0 chunks for {file_name}")
        texts = [node.get_content() for node in nodes]
        embeddings = self.embedding.get_text_embedding_batch(texts)
        if any(len(vector) != self.cfg.embed_dim for vector in embeddings):
            raise ValueError(f"Embedding dimensions do not match configured embed_dim={self.cfg.embed_dim}")

        ingested_at = _utc_now()
        encoded_metadata = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        search_documents: list[dict[str, Any]] = []
        for chunk_index, (node, vector) in enumerate(zip(nodes, embeddings, strict=True)):
            search_documents.append(
                {
                    "id": f"{file_id}-{chunk_index:08d}",
                    "chunk": node.get_content(),
                    "embedding": list(vector),
                    "file_id": file_id,
                    "file_name": file_name,
                    "page_number": _coerce_page_number(node.metadata.get("page_label")),
                    "chunk_index": chunk_index,
                    "file_size": file_size,
                    "uploaded_at": uploaded_at,
                    "ingested_at": ingested_at,
                    "metadata": encoded_metadata,
                }
            )

        client = self._get_search_client(collection_name)
        self._upload_documents(client, search_documents)
        self._update_collection_timestamp(collection_name)
        for old_file_id in old_file_ids:
            self._delete_file_documents(old_file_id, collection_name)
            with self._jobs_lock:
                self._files.pop(old_file_id, None)

        summary = self._generate_summary("\n".join(texts), file_name) if self.cfg.generate_summary else None
        if summary:
            register_summary(collection_name, file_name, summary)
        elif old_file_ids:
            unregister_summary(collection_name, file_name)

        with self._jobs_lock:
            tracked = self._files.get(file_id)
            if tracked:
                tracked.status = FileStatus.SUCCESS
                tracked.chunk_count = len(search_documents)
                tracked.ingested_at = ingested_at
                if summary:
                    tracked.metadata["summary"] = summary
        return len(search_documents)

    def _upload_documents(self, client: SearchClient, documents: list[dict[str, Any]]) -> None:
        uploaded_ids: list[str] = []
        try:
            for batch in _iter_index_batches(documents, "upload"):
                results = client.upload_documents(documents=batch)
                succeeded, failures = _indexing_outcome(results, batch)
                uploaded_ids.extend(succeeded)
                if failures:
                    raise RuntimeError(f"Azure AI Search rejected upload actions: {'; '.join(failures)}")
        except Exception as upload_error:  # noqa: BLE001
            if uploaded_ids:
                try:
                    self._delete_document_ids(client, uploaded_ids)
                except Exception as rollback_error:  # noqa: BLE001
                    raise RuntimeError(
                        f"Upload failed ({upload_error}); rollback failed ({rollback_error})"
                    ) from upload_error
            raise

    def _delete_document_ids(self, client: SearchClient, document_ids: list[str]) -> None:
        for batch in _iter_index_batches([{"id": item} for item in document_ids], "delete"):
            pending = batch
            failures: list[str] = []
            for _attempt in range(_DELETE_ATTEMPTS):
                results = client.delete_documents(documents=pending)
                _succeeded, failures = _indexing_outcome(results, pending)
                if not failures:
                    break
                failed_keys = {failure.split(":", 1)[0] for failure in failures}
                pending = [document for document in pending if str(document["id"]) in failed_keys]
            if failures:
                raise RuntimeError(f"Azure AI Search rejected delete actions: {'; '.join(failures)}")

    def _iter_documents(
        self,
        client: SearchClient,
        *,
        filter_text: str | None = None,
        select: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        last_id: str | None = None
        while True:
            cursor_filter = f"id gt {_odata_literal(last_id)}" if last_id is not None else None
            page = list(
                client.search(
                    search_text="*",
                    filter=_and_filter(filter_text, cursor_filter),
                    select=select,
                    order_by=["id asc"],
                    top=_PAGE_SIZE,
                )
            )
            if not page:
                return
            yield from page
            next_id = str(page[-1].get("id") or "")
            if len(page) < _PAGE_SIZE:
                return
            if not next_id or next_id == last_id:
                raise RuntimeError("Azure AI Search pagination did not advance")
            last_id = next_id

    def _find_file_ids_by_name(self, file_name: str, collection_name: str) -> set[str]:
        client = self._get_search_client(collection_name)
        filter_text = f"file_name eq {_odata_literal(file_name)}"
        return {
            str(hit["file_id"])
            for hit in self._iter_documents(client, filter_text=filter_text, select=["id", "file_id"])
            if hit.get("file_id")
        }

    def _delete_file_documents(self, file_id: str, collection_name: str) -> int:
        client = self._get_search_client(collection_name)
        filter_text = f"file_id eq {_odata_literal(file_id)}"
        ids = [
            str(hit["id"])
            for hit in self._iter_documents(client, filter_text=filter_text, select=["id"])
            if hit.get("id")
        ]
        if ids:
            self._delete_document_ids(client, ids)
        return len(ids)

    def _generate_summary(self, text: str, file_name: str) -> str | None:
        if self._summary_llm is None:
            return None
        snippet = text[: self.cfg.summary_max_chars]
        prompt = f"Summarise the following document ({file_name}) in one sentence (max 30 words):\n\n{snippet}"
        try:
            response = self._summary_llm.invoke(prompt)
            summary = getattr(response, "content", None) or str(response)
            return summary.strip() or None
        except Exception:  # noqa: BLE001
            logger.exception("Summary generation failed for %s", file_name)
            return None

    def _update_job(self, job_id: str, **fields: Any) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                self._jobs[job_id] = job.model_copy(update=fields)

    def _fail_job(self, job_id: str, error_message: str) -> None:
        self._update_job(job_id, status=JobState.FAILED, error_message=error_message, completed_at=_utc_now())

    def _update_file_progress(self, job_id: str, index: int, **fields: Any) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None or index >= len(job.file_details):
                return
            details = list(job.file_details)
            details[index] = details[index].model_copy(update=fields)
            tracked = self._files.get(details[index].file_id)
            if tracked:
                tracked.status = details[index].status
                tracked.error_message = details[index].error_message
                tracked.chunk_count = details[index].chunks_created
                if tracked.status == FileStatus.SUCCESS:
                    tracked.ingested_at = _utc_now()
            self._jobs[job_id] = job.model_copy(update={"file_details": details})

    @staticmethod
    def _cleanup_paths(paths: list[str]) -> None:
        for path in paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    @staticmethod
    def _translate_error(error: Exception) -> str:
        if isinstance(error, ResourceNotFoundError):
            return f"AI Search index not found: {error!s}"
        if isinstance(error, ClientAuthenticationError):
            return f"AI Search authentication failed: {error!s}"
        if isinstance(error, ServiceRequestError):
            return f"AI Search service unavailable: {error!s}"
        if isinstance(error, HttpResponseError):
            return f"AI Search request failed ({error.status_code}): {error.reason or error!s}"
        return str(error)

    def create_collection(
        self,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CollectionInfo:
        index, marker = self._ensure_index(name, description, metadata)
        if description is not None or metadata:
            marker = self._update_marker(
                name,
                description=description if description is not None else marker.get("description"),
                metadata={**(marker.get("metadata") or {}), **(metadata or {})},
                updated_at=_utc_now().isoformat(),
            )
        return self._collection_info(name, index, marker)

    def delete_collection(self, name: str) -> bool:
        try:
            index, _marker = self._get_owned_index(name)
        except ResourceNotFoundError:
            return False
        self._index_client.delete_index(index.name)
        try:
            self._index_client.get_index(index.name)
        except ResourceNotFoundError:
            pass
        else:
            return False
        self._search_clients.pop(name, None)
        with self._jobs_lock:
            self._files = {file_id: info for file_id, info in self._files.items() if info.collection_name != name}
        clear_collection_summaries(name)
        return True

    def list_collections(self) -> list[CollectionInfo]:
        collections: list[CollectionInfo] = []
        for index in self._index_client.list_indexes():
            marker = _decode_marker(index.description)
            if not marker or marker.get("backend") != _BACKEND_NAME or marker.get("schema_version") != _SCHEMA_VERSION:
                continue
            collection_name = marker.get("collection_name")
            if not isinstance(collection_name, str):
                continue
            try:
                _validate_index_schema(index, collection_name, self.cfg)
                collections.append(self._collection_info(collection_name, index, marker))
            except Exception:  # noqa: BLE001
                logger.exception("Skipping invalid AI-Q Azure AI Search index %s", index.name)
        return collections

    def get_collection(self, name: str) -> CollectionInfo | None:
        try:
            index, marker = self._get_owned_index(name)
        except ResourceNotFoundError:
            return None
        return self._collection_info(name, index, marker)

    def _collection_info(self, name: str, index: SearchIndex, marker: dict[str, Any]) -> CollectionInfo:
        client = self._get_search_client(name)
        files = self.list_files(name)
        return CollectionInfo(
            name=name,
            description=marker.get("description"),
            file_count=len(files),
            chunk_count=client.get_document_count(),
            backend=_BACKEND_NAME,
            metadata={
                **(marker.get("metadata") or {}),
                "index_name": index.name,
                "embedding_model": marker.get("embedding_model"),
                "embedding_dim": marker.get("embedding_dim"),
            },
            created_at=_parse_timestamp(marker.get("created_at")),
            updated_at=_parse_timestamp(marker.get("updated_at")),
        )

    def upload_file(
        self,
        file_path: str,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> FileInfo:
        path = Path(file_path)
        if not path.is_file():
            return FileInfo(
                file_id=str(uuid.uuid4()),
                file_name=path.name,
                collection_name=collection_name,
                status=FileStatus.FAILED,
                error_message=f"File not found: {file_path}",
            )
        job_id = self.submit_job(
            [file_path],
            collection_name,
            {"original_filenames": [path.name], "metadata": metadata or {}},
        )
        with self._jobs_lock:
            file_id = self._jobs[job_id].file_details[0].file_id
            info = self._files[file_id].model_copy(deep=True)
            info.metadata["job_id"] = job_id
            return info

    def delete_file(self, file_id: str, collection_name: str) -> bool:
        info = self.get_file_status(file_id, collection_name)
        if info is None:
            return False
        deleted = self._delete_file_documents(file_id, collection_name)
        if deleted == 0 and info.status not in {FileStatus.FAILED, FileStatus.UPLOADING}:
            return False
        with self._jobs_lock:
            self._files.pop(file_id, None)
        if deleted:
            remaining_same_name = any(
                item.file_name == info.file_name and item.file_id != file_id
                for item in self.list_files(collection_name)
            )
            if not remaining_same_name:
                unregister_summary(collection_name, info.file_name)
            self._update_collection_timestamp(collection_name)
        return True

    def list_files(self, collection_name: str) -> list[FileInfo]:
        try:
            self._get_owned_index(collection_name)
            client = self._get_search_client(collection_name)
            hits = self._iter_documents(
                client,
                select=[
                    "id",
                    "file_id",
                    "file_name",
                    "file_size",
                    "uploaded_at",
                    "ingested_at",
                    "metadata",
                ],
            )
            aggregated: dict[str, dict[str, Any]] = {}
            for hit in hits:
                file_id = str(hit.get("file_id") or "")
                if not file_id:
                    continue
                entry = aggregated.setdefault(
                    file_id,
                    {
                        "file_name": hit.get("file_name") or "unknown",
                        "file_size": hit.get("file_size"),
                        "uploaded_at": _parse_timestamp(hit.get("uploaded_at")),
                        "ingested_at": _parse_timestamp(hit.get("ingested_at")),
                        "metadata": _parse_metadata(hit.get("metadata")),
                        "chunk_count": 0,
                    },
                )
                entry["chunk_count"] += 1
        except ResourceNotFoundError:
            return []
        except Exception:  # noqa: BLE001
            logger.exception("Failed to list files for %r", collection_name)
            return []

        files = {
            file_id: FileInfo(
                file_id=file_id,
                file_name=entry["file_name"],
                collection_name=collection_name,
                status=FileStatus.SUCCESS,
                file_size=entry["file_size"],
                chunk_count=entry["chunk_count"],
                uploaded_at=entry["uploaded_at"],
                ingested_at=entry["ingested_at"],
                metadata=entry["metadata"],
            )
            for file_id, entry in aggregated.items()
        }
        with self._jobs_lock:
            for file_id, tracked in self._files.items():
                if tracked.collection_name == collection_name and (
                    tracked.status != FileStatus.SUCCESS or file_id not in files
                ):
                    files[file_id] = tracked.model_copy(deep=True)
        return sorted(files.values(), key=lambda item: (item.file_name, item.file_id))

    def get_file_status(self, file_id: str, collection_name: str) -> FileInfo | None:
        with self._jobs_lock:
            tracked = self._files.get(file_id)
            if tracked and tracked.collection_name == collection_name and tracked.status != FileStatus.SUCCESS:
                return tracked.model_copy(deep=True)
        return next((file_info for file_info in self.list_files(collection_name) if file_info.file_id == file_id), None)

    async def health_check(self) -> bool:
        def _check() -> bool:
            list(self._index_client.list_index_names())
            return True

        try:
            return await asyncio.to_thread(_check)
        except Exception:  # noqa: BLE001
            logger.exception("Azure AI Search ingestor health_check failed")
            return False


def _resolve_filenames(file_paths: list[str], raw: Any) -> list[str]:
    if isinstance(raw, dict):
        return [raw.get(path) or Path(path).name for path in file_paths]
    if isinstance(raw, list):
        return [
            raw[index] if index < len(raw) and raw[index] else Path(path).name for index, path in enumerate(file_paths)
        ]
    return [Path(path).name for path in file_paths]


def _coerce_page_number(page_label: Any) -> int | None:
    if page_label is None:
        return None
    try:
        page_number = int(str(page_label))
        return page_number if page_number > 0 else None
    except (TypeError, ValueError):
        return None
