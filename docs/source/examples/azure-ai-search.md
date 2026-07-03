<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Example: Azure AI Search knowledge layer

Use Azure AI Search as the document store while retaining AI-Q's existing
upload API, per-conversation collection routing, document summaries, and
citations. This example assumes the Azure AI Search service and embedding
endpoint already exist; it does not deploy Azure infrastructure.

Install the backend dependency:

```bash
uv pip install -e "sources/knowledge_layer[azure_ai_search]"
```

Set the environment used by the adapter:

```bash
export AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net
export NVIDIA_API_KEY=<embedding-api-key>
# Optional; omit to use DefaultAzureCredential.
export AZURE_SEARCH_API_KEY=<search-admin-key>
```

## Grant managed identity access

When `AZURE_SEARCH_API_KEY` is absent, enable role-based access on the Azure AI
Search service and grant the workload identity both of these built-in roles:

| Role | Used for |
|------|----------|
| `Search Service Contributor` | Create, inspect, update, and delete AI-Q collection indexes. |
| `Search Index Data Contributor` | Upload, query, and delete index documents. |

Assign the roles at the search-service scope because AI-Q creates one index per
logical collection. The principal ID is the object ID of the system-assigned or
user-assigned managed identity running AI-Q.


Replace the `knowledge_search` block in a web configuration such as
`configs/config_web_default_llamaindex.yml`:

```yaml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    backend: azure_ai_search
    collection_name: ${COLLECTION_NAME:-aiq_default}
    top_k: 5
    use_hybrid: true
    use_semantic_ranker: false

    generate_summary: true
    summary_model: summary_llm
    summary_db: ${AIQ_SUMMARY_DB:-sqlite+aiosqlite:///./summaries.db}
```

Explicit YAML options override environment defaults. `AZURE_SEARCH_API_KEY`
selects API-key authentication when present; otherwise the adapter uses
`DefaultAzureCredential`. Set `AZURE_CLIENT_ID` to select a user-assigned
identity. Embeddings share `AIQ_EMBED_BASE_URL`, `AIQ_EMBED_MODEL`, and
`NVIDIA_API_KEY` with the LlamaIndex backend. Azure-specific optional settings
are `AIQ_EMBED_DIM` and `AIQ_AZURE_SEARCH_INDEX_PREFIX`.

Semantic ranking is opt-in because availability depends on the Azure AI Search
service. Set `use_semantic_ranker: true` only when semantic ranking is enabled;
it also requires `use_hybrid: true`.

Existing indexes must use the configured `embed_dim`. Delete and re-ingest a
collection when changing embedding dimensions. Frontend WebSocket queries use
the conversation ID as the collection; direct API tests must supply equivalent
context or query the configured fallback collection.

The backend only lists or mutates indexes carrying its AI-Q ownership marker.
Logical collection names map to collision-safe physical names under
`azure_search_index_prefix`; un-namespaced indexes from earlier versions are
ignored and must be re-ingested. File IDs returned by upload are authoritative
for status and delete operations. Same-name uploads replace the prior file only
after the new generation has been fully indexed.
