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

```yaml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    backend: azure_ai_search
    collection_name: ${COLLECTION_NAME:-aiq_default}

    azure_search_endpoint: https://<service>.search.windows.net
    azure_search_auth_mode: managed_identity
    # For key authentication instead:
    # azure_search_auth_mode: api_key
    # azure_search_api_key: ${AZURE_SEARCH_API_KEY}

    embed_endpoint: https://integrate.api.nvidia.com/v1
    embed_model: nvidia/nv-embed-v1
    embed_dim: 4096
    # NVIDIAEmbedding reads NVIDIA_API_KEY when this field is omitted.
    # embed_api_key: ${NVIDIA_API_KEY}

    use_hybrid: true
    use_semantic_ranker: true
    top_k: 5
    chunk_size: 512
    chunk_overlap: 64

    generate_summary: true
    summary_model: summary_llm
    summary_db: ${AIQ_SUMMARY_DB:-sqlite+aiosqlite:///./summaries.db}
```

Managed identity uses `DefaultAzureCredential`. Set `AZURE_CLIENT_ID` when a
user-assigned identity should be selected. The identity needs permission to
create and delete indexes and to read, write, and delete index documents.

The adapter parses PDF, DOCX, TXT, and Markdown uploads with LlamaIndex,
creates one Azure AI Search index per AI-Q collection, and performs vector or
hybrid retrieval with optional semantic ranking. AI-Q frontend conversations
use their conversation ID as the collection name, keeping uploads and
WebSocket retrieval in the same index.

`embed_dim` must match both the embedding model output and any existing index.
Changing from a 2048-dimensional model to `nvidia/nv-embed-v1` at 4096
dimensions requires deleting and re-ingesting the old collection. The adapter
does not alter an existing index schema.

For direct API tests, use the same collection or conversation context used for
upload. A standalone chat request without that context falls back to the
configured `collection_name`.
