<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Azure AI Search backend

This backend stores AI-Q document chunks and vectors in Azure AI Search. It
uses the shared `knowledge_retrieval` NAT function, Knowledge API, session
collection routing, summary store, and citation formatter.

## Install

```bash
uv pip install -e "sources/knowledge_layer[azure_ai_search]"
```

## Configure

Set service and model credentials in the environment:

```bash
export AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net
export NVIDIA_API_KEY=<embedding-api-key>
# Optional: setting this selects API-key auth instead of DefaultAzureCredential.
export AZURE_SEARCH_API_KEY=<search-admin-key>
```

```yaml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    backend: azure_ai_search
    collection_name: ${COLLECTION_NAME:-aiq_default}
    use_hybrid: true
    use_semantic_ranker: false
    top_k: 5
    chunk_size: 512
    chunk_overlap: 64

    generate_summary: true
    summary_model: summary_llm
    summary_db: ${AIQ_SUMMARY_DB:-sqlite+aiosqlite:///./summaries.db}
```

Explicit YAML values still override the environment-backed defaults. Azure
Search uses `AZURE_SEARCH_ENDPOINT` and optional `AZURE_SEARCH_API_KEY`.
Embedding configuration shares `AIQ_EMBED_BASE_URL`, `AIQ_EMBED_MODEL`, and
`NVIDIA_API_KEY` with the LlamaIndex backend; Azure additionally accepts
`AIQ_EMBED_DIM` and `AIQ_AZURE_SEARCH_INDEX_PREFIX`.

Semantic ranking is disabled by default because support depends on the Azure AI
Search service. Set `use_semantic_ranker: true` only when semantic ranking is
enabled; it also requires `use_hybrid: true`.

When `AZURE_SEARCH_API_KEY` is absent, the adapter uses
`DefaultAzureCredential`. Set `AZURE_CLIENT_ID` when a user-assigned identity
should be selected. The identity needs permission to create and delete indexes
and to read, write, and delete index documents.

The adapter parses PDF, DOCX, TXT, and Markdown uploads with LlamaIndex,
creates one namespaced Azure AI Search index per AI-Q collection, and performs
vector or hybrid retrieval with optional semantic ranking. Logical collection
names are sanitized and combined with a stable hash, preventing collisions and
protecting unrelated indexes in a shared service. Only indexes containing the
AI-Q ownership/schema marker are visible or mutable through this backend.

Upload responses return canonical UUID file IDs used by job progress, list,
status, and delete operations. Re-uploading the same filename writes and
verifies a new generation before deleting old chunks. Upload and delete
requests stay below Azure's 1,000-action and 16 MiB limits, and every
per-document result is checked.

Collections use the shared Knowledge Layer TTL settings:
`AIQ_COLLECTION_TTL_HOURS` defaults to 24 hours and
`AIQ_TTL_CLEANUP_INTERVAL_SECONDS` defaults to 3600 seconds. Successful file
and collection deletion also clears corresponding summary records.

`embed_dim` must match both the embedding model output and any existing index.
Changing from a 2048-dimensional model to `nvidia/nv-embed-v1` at 4096
dimensions requires deleting and re-ingesting the old collection. The adapter
validates ownership, fields, vector profile, dimensions, and semantic
configuration before use; it does not alter an incompatible schema. Indexes
created by the earlier un-namespaced implementation are deliberately ignored,
so re-ingest those collections.

For direct API tests, use the same collection or conversation context used for
upload. A standalone chat request without that context falls back to the
configured `collection_name`.
