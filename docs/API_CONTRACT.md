# DCLC API contract — baseline v0.1

## `GET /health`

Returns a small health response when the API process is running.

## `POST /api/v1/analyze`

Multipart form fields:

- `source_file`: original/reference file (required)
- `target_file`: newer/target file (required)
- `chunk_size`: fixed-size chunk length in bytes, from 1 through 1,048,576; default `4096`

Each file is limited to 10 MiB in this starter implementation.

The response identifies its method as `fixed_size_sha256_baseline` and labels the research status `baseline_only`. `matched_target_bytes` is the total length of target chunk occurrences whose SHA-256 and length also occur in the source, limited by occurrence counts. `target_byte_savings_ratio` is `matched_target_bytes / target_size_bytes`, or zero for an empty target.

## Important limitations

- Chunks are fixed-size and depend on byte alignment; insertions can shift later boundaries.
- A matching chunk is an estimate of reusable bytes, not a complete storage-engine implementation.
- No content-defined chunking, learned embeddings, resemblance search, delta encoding, persistence, or multi-user storage is implemented yet.
- Do not describe baseline metrics as results from the proposed research method.
