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

"""NAT function for knowledge retrieval.

This function provides direct library access to the knowledge layer,
allowing agents to search ingested documents without an external API server.

The retriever is instantiated once and reused for all queries.
"""

import logging
import os
from typing import Literal

from pydantic import Field
from pydantic import HttpUrl
from pydantic import SecretStr
from pydantic import model_validator

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


# Type-safe backend selection - Pydantic validates at config load time
BackendType = Literal["llamaindex", "foundational_rag", "azure_ai_search"]


class KnowledgeRetrievalConfig(FunctionBaseConfig, name="knowledge_retrieval"):
    """Configuration for knowledge retrieval function."""

    backend: BackendType = Field(default="llamaindex", description="Knowledge backend to use")
    collection_name: str = Field(default="default", description="Name of the collection/index to search")
    top_k: int = Field(default=5, description="Number of results to return")
    # Summarization options (applies to all backends)
    generate_summary: bool = Field(
        default=False, description="Generate one-sentence summary for each ingested document"
    )
    summary_model: str | None = Field(
        default=None,
        description="Required when generate_summary=true: LLM reference from llms: section",
    )
    summary_db: str = Field(
        default="sqlite+aiosqlite:///./summaries.db",
        description="Database URL for document summaries (SQLite or PostgreSQL)",
    )
    # LlamaIndex-specific options
    chroma_dir: str = Field(
        default="/tmp/chroma_data", description="Directory for ChromaDB persistence (LlamaIndex only)"
    )
    # Foundational RAG (hosted RAG Blueprint) options
    rag_url: str = Field(default="http://localhost:8081/v1", description="RAG query server URL (foundational_rag only)")
    ingest_url: str = Field(
        default="http://localhost:8082/v1", description="RAG ingestion server URL (foundational_rag only)"
    )
    timeout: int = Field(default=120, description="Request timeout in seconds (foundational_rag only)")
    verify_ssl: bool = Field(
        default=True, description="Verify SSL certificates (foundational_rag only). Set false for self-signed certs."
    )
    # Azure AI Search options
    azure_search_endpoint: HttpUrl | None = Field(
        default=None,
        description="Azure AI Search service URL (azure_ai_search only)",
    )
    azure_search_api_key: SecretStr | None = Field(
        default=None,
        description="Optional Azure AI Search admin key; managed identity is used when omitted",
    )
    azure_search_auth_mode: Literal["managed_identity", "api_key"] = Field(
        default="managed_identity",
        description="Authentication mode for Azure AI Search",
    )
    embed_endpoint: HttpUrl = Field(
        default=HttpUrl("https://integrate.api.nvidia.com/v1"),
        description="OpenAI-compatible embedding endpoint (azure_ai_search only)",
    )
    embed_model: str = Field(
        default="nvidia/nv-embed-v1",
        description="Embedding model name (azure_ai_search only)",
    )
    embed_dim: int = Field(
        default=4096,
        gt=0,
        description="Embedding dimensions; must match existing Azure AI Search indexes",
    )
    embed_api_key: SecretStr | None = Field(
        default=None,
        description="Optional embedding API key; NVIDIA_API_KEY is used when omitted",
    )
    use_hybrid: bool = Field(default=True, description="Combine vector and keyword search (azure_ai_search only)")
    use_semantic_ranker: bool = Field(default=True, description="Enable Azure semantic ranking (azure_ai_search only)")
    chunk_size: int = Field(default=512, gt=0, description="Tokens per chunk (azure_ai_search only)")
    chunk_overlap: int = Field(default=64, ge=0, description="Token overlap between chunks (azure_ai_search only)")
    summary_max_chars: int = Field(
        default=1000,
        gt=0,
        description="Maximum document characters sent to summary model (azure_ai_search only)",
    )

    @model_validator(mode="after")
    def validate_backend_config(self):
        """Validate and warn about unused backend-specific config options."""
        backend = self.backend.lower()

        # Validate summary configuration
        if self.generate_summary and not self.summary_model:
            raise ValueError(
                "generate_summary=true requires summary_model to be set. "
                "Configure summary_model to reference an LLM from the llms: section."
            )

        if backend == "llamaindex":
            # LlamaIndex uses chroma_dir, warn if RAG-specific options are set
            if self.rag_url != "http://localhost:8081/v1":
                logger.warning("rag_url is ignored for llamaindex backend")
            if self.ingest_url != "http://localhost:8082/v1":
                logger.warning("ingest_url is ignored for llamaindex backend")

        elif backend == "foundational_rag":
            # Foundational RAG uses rag_url/ingest_url, warn if others are set
            if self.chroma_dir != "/tmp/chroma_data":
                logger.warning("chroma_dir is ignored for foundational_rag backend")
            if not self.verify_ssl:
                logger.warning("SSL verification disabled for foundational_rag. Use only in trusted environments.")

        elif backend == "azure_ai_search":
            if self.azure_search_endpoint is None:
                raise ValueError("azure_ai_search requires azure_search_endpoint")
            if self.azure_search_auth_mode == "api_key" and self.azure_search_api_key is None:
                raise ValueError("azure_search_auth_mode=api_key requires azure_search_api_key")
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")

        return self


def _setup_backend(config: KnowledgeRetrievalConfig, summary_llm_obj=None) -> tuple[str, dict]:
    """
    Import the backend adapter and build its configuration.

    Importing the adapter module triggers the @register_retriever/@register_ingestor
    decorators, which register the adapter classes with the factory.

    Args:
        config: Knowledge retrieval configuration
        summary_llm_obj: Optional resolved LLM object for summarization

    Returns:
        Tuple of (backend_name, backend_config_dict)
    """
    backend = config.backend.lower()

    # Summary config: LLM object if resolved, else adapters use default NVIDIA model
    summary_config = {
        "generate_summary": config.generate_summary,
        "summary_llm": summary_llm_obj,
    }

    if backend == "llamaindex":
        import knowledge_layer.llamaindex.adapter  # noqa: F401

        os.environ.setdefault("AIQ_CHROMA_DIR", config.chroma_dir)
        backend_config = {
            "persist_dir": config.chroma_dir,
            **summary_config,
        }

    elif backend == "foundational_rag":
        import knowledge_layer.foundational_rag.adapter  # noqa: F401

        backend_config = {
            "rag_url": config.rag_url,
            "ingest_url": config.ingest_url,
            "timeout": config.timeout,
            "verify_ssl": config.verify_ssl,
            **summary_config,
        }

    elif backend == "azure_ai_search":
        import knowledge_layer.azure_ai_search.adapter  # noqa: F401

        backend_config = {
            "endpoint": str(config.azure_search_endpoint),
            "api_key": config.azure_search_api_key,
            "auth_mode": config.azure_search_auth_mode,
            "embed_endpoint": str(config.embed_endpoint),
            "embed_model": config.embed_model,
            "embed_dim": config.embed_dim,
            "embed_api_key": config.embed_api_key,
            "use_hybrid": config.use_hybrid,
            "use_semantic_ranker": config.use_semantic_ranker,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "summary_max_chars": config.summary_max_chars,
            "collection_name": config.collection_name,
            "cleanup_files": True,
            **summary_config,
        }

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'llamaindex', 'foundational_rag', or 'azure_ai_search'.")

    os.environ["KNOWLEDGE_RETRIEVER_BACKEND"] = backend
    os.environ["KNOWLEDGE_INGESTOR_BACKEND"] = backend

    return backend, backend_config


def _get_retriever(config: KnowledgeRetrievalConfig):
    """Get the retriever singleton from the factory."""
    from aiq_agent.knowledge.factory import get_retriever

    backend, backend_config = _setup_backend(config)
    retriever = get_retriever(backend, backend_config)
    logger.info("Initialized %s retriever", backend)
    return retriever


def _initialize_ingestor(config: KnowledgeRetrievalConfig, summary_llm_obj=None):
    """
    Initialize and activate the ingestor for the Knowledge API.

    Called during function registration to:
    1. Create the ingestor singleton via the factory
    2. Set it as the active ingestor for API routes to use

    Args:
        config: Knowledge retrieval configuration
        summary_llm_obj: Optional resolved LLM object for summarization
    """
    from aiq_agent.knowledge.factory import get_ingestor
    from aiq_agent.knowledge.factory import set_active_ingestor

    backend, backend_config = _setup_backend(config, summary_llm_obj)
    ingestor = get_ingestor(backend, backend_config)
    set_active_ingestor(ingestor)
    logger.info("Activated %s ingestor for Knowledge API", backend)
    return ingestor


def _format_results(retrieval_result, query: str) -> str:
    """
    Format retrieval results for LLM consumption.

    Returns a structured string that provides context for the agent.
    The format includes explicit citation fields so the LLM knows exactly
    what to use in its References section.
    """
    # Check for retrieval errors and surface them to the agent
    if not retrieval_result.success:
        error_msg = retrieval_result.error_message or "Unknown error"
        return f"Knowledge retrieval failed: {error_msg}\n\nQuery: '{query}'"

    if not retrieval_result.chunks:
        return f"No relevant documents found for query: '{query}'"

    lines = [f"Found {len(retrieval_result.chunks)} relevant document(s):\n"]

    for i, chunk in enumerate(retrieval_result.chunks, 1):
        # Build citation string: "filename, p.X" or just "filename"
        if chunk.page_number and chunk.page_number > 0:
            citation = f"{chunk.file_name}, p.{chunk.page_number}"
        else:
            citation = chunk.file_name

        # Header with source info
        lines.append(f"--- Result {i} ---")
        lines.append(f"Source: {chunk.file_name}")
        if chunk.page_number and chunk.page_number > 0:
            lines.append(f"Page: {chunk.page_number}")
        lines.append(f"Citation: {citation}")
        lines.append(f"Content Type: {chunk.content_type.value}")
        lines.append(f"Relevance Score: {chunk.score:.2f}")
        lines.append("")

        # Content (truncate if very long)
        content = chunk.content
        if len(content) > 1500:
            content = content[:1500] + "... [truncated]"
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


@register_function(config_type=KnowledgeRetrievalConfig)
async def knowledge_retrieval(config: KnowledgeRetrievalConfig, _builder: Builder):
    """
    Knowledge retrieval function for searching ingested documents.

    This function provides semantic search over documents that have been
    previously ingested into the knowledge layer. It supports multiple
    backends (LlamaIndex, Foundational RAG) and returns formatted results
    suitable for LLM consumption.

    The retriever and ingestor are initialized once when the function is
    created and reused for all subsequent queries. The ingestor singleton
    is also made available to the Knowledge API routes via the factory.
    """
    # Resolve summary LLM if specified (enterprise approach)
    summary_llm_obj = None
    if config.summary_model and config.generate_summary:
        from nat.builder.framework_enum import LLMFrameworkEnum

        summary_llm_obj = await _builder.get_llm(config.summary_model, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        logger.info("Resolved summary model: %s", config.summary_model)

    # Initialize summary DB with configured URL
    from aiq_agent.knowledge.factory import configure_summary_db

    configure_summary_db(config.summary_db)

    retriever = _get_retriever(config)

    _initialize_ingestor(config, summary_llm_obj)

    collection = config.collection_name
    top_k = config.top_k

    logger.info(
        "Knowledge retrieval initialized: backend=%s, collection=%s, top_k=%d", config.backend, collection, top_k
    )

    async def search(query: str) -> str:
        """Search for documents relevant to the query.

        Args:
            query (str): Natural language query describing what information you need.

        Returns:
            str: Formatted string containing relevant document excerpts with citations.
        """
        # Determine collection: prefer session context (UI) over config default
        try:
            ctx = Context.get()
            session_collection = ctx.conversation_id if ctx else None
            target_collection = session_collection or collection
        except Exception:
            target_collection = collection

        logger.info(f"Knowledge search: query='{query[:100]}...' collection={target_collection}")

        try:
            # Call the retriever
            result = await retriever.retrieve(
                query=query,
                collection_name=target_collection,
                top_k=top_k,
            )

            # Format for LLM
            formatted = _format_results(result, query)
            logger.info(f"Knowledge search returned {len(result.chunks)} chunks")
            # Debug: Log what we're returning to the LLM
            logger.debug(f"Formatted result for LLM:\n{formatted[:500]}...")
            return formatted

        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return f"Error searching knowledge base: {str(e)}"

    # Yield the function info for NAT registration
    yield FunctionInfo.from_fn(
        search,
        description=(
            "Search the knowledge base for relevant documents. "
            "Use this to find information from ingested PDFs, documents, and other files. "
            f"Returns up to {top_k} relevant excerpts with citations."
        ),
    )
