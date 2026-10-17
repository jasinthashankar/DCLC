# DCLC — Deduplication and Content-Level Compression

DCLC is a research prototype for studying storage reduction across related file versions. The implementation will be built in verifiable stages, with baseline behavior measured separately from any learned resemblance model.

## Current stage

**Stage 1: API and exact-deduplication baseline.** This initial implementation provides a small FastAPI service to compare two uploaded files using deterministic fixed-size chunking and SHA-256 chunk identities. It is a baseline for evaluation, not yet the paper's complete content-defined or learned DCLC method.

## Planned stages

1. Establish a reproducible baseline and automated tests.
2. Integrate and validate the research chunking / resemblance pipeline.
3. Evaluate storage savings, retrieval quality, and exact reconstruction.
4. Add a React frontend after the API contract is stable.

## Run locally

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn src.dclc.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Tests

```powershell
python -m pytest -q
```

## Research integrity

Reported metrics must come from executed experiments. The baseline must not be described as a learned model, and a result is not considered validated until the test output and evaluation configuration are recorded.