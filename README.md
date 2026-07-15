# EduRAG

EduRAG is an education question-answering service with a FastAPI backend and a
browser-based chat workspace. It can run against its configured production
services or in an explicit, in-memory demo mode.

## Secure local setup

The tracked [`config.example.yaml`](config.example.yaml) is the safe template.
Create your local configuration and supply credentials only through the
environment:

```bash
cp config.example.yaml config.yaml
export EDURAG_MYSQL_PASSWORD='replace-me'
export EDURAG_REDIS_PASSWORD='replace-me'
export EDURAG_ADMIN_TOKEN='replace-with-a-long-random-token'
conda activate edurag
python -m pytest -q
python main.py --mock
```

`config.yaml` is local deployment state and must never be committed. It is
ignored by Git; keep database passwords and the administrator token out of YAML,
source control, logs, and shell history. Use a long, randomly generated value
for `EDURAG_ADMIN_TOKEN` and inject it through your deployment secret manager in
production.

## Runtime modes

- **Production (default):** `python main.py` initializes every configured
  backend. MySQL, Redis, Milvus, the LLM endpoint, and the configured local
  models/data must be available. If a required backend cannot initialize, the
  API reports that it is unavailable rather than silently serving demo data.
- **Demo/mock:** `python main.py --mock` sets `EDURAG_API_MOCK=true` and starts
  an in-memory example system. It needs no MySQL, Redis, Milvus, or LLM service
  and is for local demonstrations only; its answers are clearly identified as
  mock responses.

You may also set `EDURAG_API_MOCK=true` explicitly when launching through a
process manager. Other values, including `1`, do not enable mock mode.

## FAQ administration

FAQ reads are public, but FAQ create, update, and delete requests require the
administrator Bearer token. Send the same token configured in
`EDURAG_ADMIN_TOKEN`; do not expose it to the browser or commit it.

```bash
curl -X POST http://127.0.0.1:8001/api/faq \
  -H "Authorization: Bearer $EDURAG_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is RAG?","answer":"Retrieval-augmented generation."}'
```

Missing or invalid tokens are rejected with HTTP 401. Production write access
should be made only by trusted server-side tooling over TLS.

## Verification

Run the complete automated suite from the project root:

```bash
conda run -n edurag python -m pytest -q
```
