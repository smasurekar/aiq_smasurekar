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

Replace the `knowledge_search` block in a web configuration such as
`configs/config_web_default_llamaindex.yml`:

```yaml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    backend: azure_ai_search
    collection_name: ${COLLECTION_NAME:-aiq_default}
    top_k: 5

    azure_search_endpoint: https://<service>.search.windows.net
    azure_search_auth_mode: managed_identity

    embed_endpoint: https://integrate.api.nvidia.com/v1
    embed_model: nvidia/nv-embed-v1
    embed_dim: 4096
    use_hybrid: true
    use_semantic_ranker: true

    generate_summary: true
    summary_model: summary_llm
    summary_db: ${AIQ_SUMMARY_DB:-sqlite+aiosqlite:///./summaries.db}
```

For API-key authentication, set `azure_search_auth_mode: api_key` and add
`azure_search_api_key: ${AZURE_SEARCH_API_KEY}`. Managed identity uses
`DefaultAzureCredential`; set `AZURE_CLIENT_ID` to select a user-assigned
identity. If `embed_api_key` is omitted, the NVIDIA embedding client reads
`NVIDIA_API_KEY`.

Existing indexes must use the configured `embed_dim`. Delete and re-ingest a
collection when changing embedding dimensions. Frontend WebSocket queries use
the conversation ID as the collection; direct API tests must supply equivalent
context or query the configured fallback collection.
