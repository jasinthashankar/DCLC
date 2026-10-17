"""HTTP API for the DCLC fixed-size deduplication baseline."""

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.dclc.core import analyze_pair

app = FastAPI(
    title="DCLC Research API",
    version="0.1.0",
    description=(
        "Baseline API for exact fixed-size chunk reuse. This is not yet the full "
        "content-defined chunking, learned resemblance, or delta-encoding method."
    ),
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


async def _read_limited(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Each uploaded file must be at most {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB.",
        )
    return data


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "dclc-api", "version": app.version}


@app.post("/api/v1/analyze", tags=["analysis"])
async def analyze_files(
    source_file: UploadFile = File(..., description="Original or reference file"),
    target_file: UploadFile = File(..., description="New or target version"),
    chunk_size: int = Query(default=4096, ge=1, le=1_048_576),
) -> dict[str, object]:
    """Compare two files using the fixed-size exact-chunk baseline."""
    source = await _read_limited(source_file)
    target = await _read_limited(target_file)
    result = analyze_pair(source, target, chunk_size=chunk_size)
    return {
        "method": "fixed_size_sha256_baseline",
        "research_status": "baseline_only",
        "notice": (
            "These are exact fixed-size chunk reuse metrics. They do not measure "
            "learned resemblance detection or delta-encoding performance."
        ),
        "result": result.to_dict(),
    }
