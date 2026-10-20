"""Version-Aware Contrastive Learning Pairing & Feasibility Study (Phase 3.5).

Analyzes chunk alignment, boundary shifts, content overlap, and pairing rules
for multi-version deduplication and resemblance detection in DCLC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class PairCategory(str, Enum):
    """Pair classification categories for version-aware contrastive learning."""
    EXACT_DUPLICATE = "exact_duplicate"
    VERSION_RESEMBLANCE = "version_resemblance"
    BOUNDARY_SHIFT_OVERLAP = "boundary_shift_overlap"
    UNRELATED = "unrelated"


@dataclass
class ChunkPairAudit:
    """Detailed audit of a pair of chunks across file versions."""
    chunk_idx: int
    v1_global_id: int
    v2_global_id: int
    v1_byte_offset: int
    v1_byte_length: int
    v2_byte_offset: int
    v2_byte_length: int
    sha256_match: bool
    line_jaccard: float
    category: PairCategory
    description: str


def compute_line_jaccard(bytes1: bytes, bytes2: bytes) -> float:
    """Compute line-level Jaccard similarity between two byte chunks."""
    lines1 = set(bytes1.splitlines())
    lines2 = set(bytes2.splitlines())
    if not (lines1 or lines2):
        return 0.0
    return float(len(lines1 & lines2) / len(lines1 | lines2))


def audit_version_chunk_alignment(
    raw_v1: bytes,
    raw_v2: bytes,
    v1_chunk_records: list[dict[str, Any]],
    v2_chunk_records: list[dict[str, Any]],
    resemblance_jaccard_threshold: float = 0.30,
) -> list[ChunkPairAudit]:
    """Audit aligned chunk pairs between two file versions.

    Args:
        raw_v1: Byte content of Version 1.
        raw_v2: Byte content of Version 2.
        v1_chunk_records: List of manifest chunk dicts for Version 1.
        v2_chunk_records: List of manifest chunk dicts for Version 2.
        resemblance_jaccard_threshold: Minimum Jaccard to classify non-identical chunks as resemblance.

    Returns:
        List of ChunkPairAudit objects for all aligned positions.
    """
    num_pairs = min(len(v1_chunk_records), len(v2_chunk_records))
    audits = []

    for i in range(num_pairs):
        c1 = v1_chunk_records[i]
        c2 = v2_chunk_records[i]

        b1 = raw_v1[c1["byte_offset"] : c1["byte_offset"] + c1["byte_length"]]
        b2 = raw_v2[c2["byte_offset"] : c2["byte_offset"] + c2["byte_length"]]

        sha_match = bool(c1["sha256"] == c2["sha256"])
        jaccard = compute_line_jaccard(b1, b2)

        if sha_match:
            category = PairCategory.EXACT_DUPLICATE
            desc = "Bit-for-bit identical chunk across versions."
        elif jaccard >= resemblance_jaccard_threshold:
            category = PairCategory.VERSION_RESEMBLANCE
            desc = f"Aligned version modification ({jaccard*100:.1f}% line overlap)."
        elif jaccard > 0.0:
            category = PairCategory.BOUNDARY_SHIFT_OVERLAP
            desc = f"Weak overlap ({jaccard*100:.1f}%) due to boundary displacement."
        else:
            category = PairCategory.UNRELATED
            desc = "Disjoint content."

        audits.append(
            ChunkPairAudit(
                chunk_idx=i,
                v1_global_id=c1["global_chunk_id"],
                v2_global_id=c2["global_chunk_id"],
                v1_byte_offset=c1["byte_offset"],
                v1_byte_length=c1["byte_length"],
                v2_byte_offset=c2["byte_offset"],
                v2_byte_length=c2["byte_length"],
                sha256_match=sha_match,
                line_jaccard=jaccard,
                category=category,
                description=desc,
            )
        )

    return audits


def assess_feasibility_of_train_test_split(
    resemblance_pairs: list[Any],
    min_required_pairs: int = 20,
) -> dict[str, Any]:
    """Assess whether the dataset has enough trustworthy edited pairs for a valid train/test split.

    Prevents data leakage and meaningless evaluations on under-determined datasets.

    Args:
        resemblance_pairs: List of audited resemblance pairs.
        min_required_pairs: Minimum threshold of distinct pairs required for statistical validity.

    Returns:
        Feasibility assessment report.
    """
    count = len(resemblance_pairs)
    can_split = bool(count >= min_required_pairs)

    if not can_split:
        recommendations = [
            f"Current dataset contains only {count} trustworthy edited pairs (insufficient for train/test split).",
            "Splitting 3 pairs into train (e.g. 2 pairs) and test (1 pair) for an 800k-parameter model causes severe overfitting or data leakage.",
            "STOP training before proceeding to prevent publishing misleading evaluation numbers.",
            "Acquisition Plan A: Generate a synthetic multi-version corpus of 200+ files with controlled token mutation rates (5%, 10%, 25%, 50%).",
            "Acquisition Plan B: Ingest genuine revision histories from standard research benchmarks (Linux Kernel, GCC git commits, BSD) as specified in paper Section 4.1.",
        ]
    else:
        recommendations = [
            f"Dataset contains {count} pairs, meeting minimum required threshold ({min_required_pairs}).",
            "Proceed with stratified train/val/test split across disjoint file versions to prevent data leakage.",
        ]

    return {
        "can_split_safely": can_split,
        "trustworthy_resemblance_pairs_count": count,
        "min_required_pairs": min_required_pairs,
        "recommendations": recommendations,
    }


def run_version_aware_feasibility_study(
    raw_dir: Path | str = "data/raw",
    manifest_path: Path | str = "data/processed/dataset_manifest.json",
) -> dict[str, Any]:
    """Execute complete feasibility audit of version-aware contrastive pairing.

    Args:
        raw_dir: Directory containing raw sample files.
        manifest_path: Path to dataset manifest JSON.

    Returns:
        Structured feasibility study report.
    """
    raw_path = Path(raw_dir)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    chunks = manifest["chunks"]

    v1_chunks = [c for c in chunks if c["file_id"] == 1]
    v2_chunks = [c for c in chunks if c["file_id"] == 2]

    raw_v1 = (raw_path / "sample_doc_v1.txt").read_bytes()
    raw_v2 = (raw_path / "sample_doc_v2.txt").read_bytes()

    audits = audit_version_chunk_alignment(
        raw_v1=raw_v1,
        raw_v2=raw_v2,
        v1_chunk_records=v1_chunks,
        v2_chunk_records=v2_chunks,
    )

    exact_duplicates = [a for a in audits if a.category == PairCategory.EXACT_DUPLICATE]
    resemblance_pairs = [a for a in audits if a.category == PairCategory.VERSION_RESEMBLANCE]
    boundary_shifts = [a for a in audits if a.category == PairCategory.BOUNDARY_SHIFT_OVERLAP]

    feasibility = assess_feasibility_of_train_test_split(resemblance_pairs, min_required_pairs=20)

    report = {
        "alignment_audits": [
            {
                "chunk_idx": a.chunk_idx,
                "v1_global_id": a.v1_global_id,
                "v2_global_id": a.v2_global_id,
                "v1_offset_len": f"{a.v1_byte_offset} ({a.v1_byte_length})",
                "v2_offset_len": f"{a.v2_byte_offset} ({a.v2_byte_length})",
                "sha256_match": a.sha256_match,
                "line_jaccard": a.line_jaccard,
                "category": a.category.value,
                "description": a.description,
            }
            for a in audits
        ],
        "category_counts": {
            "exact_duplicates": len(exact_duplicates),
            "version_resemblance": len(resemblance_pairs),
            "boundary_shift_overlap": len(boundary_shifts),
            "total_aligned_positions": len(audits),
        },
        "feasibility": feasibility,
    }

    return report


def main() -> None:
    """CLI runner for Version-Aware Feasibility Study."""
    print("=" * 80)
    print(" DCLC Phase 3.5: Version-Aware Contrastive Learning Feasibility Study")
    print("=" * 80)

    report = run_version_aware_feasibility_study()

    counts = report["category_counts"]
    print(f"\nTotal Aligned Chunk Positions between v1 and v2: {counts['total_aligned_positions']}")
    print(f"  - Exact Duplicate Chunks (100% byte match):     {counts['exact_duplicates']}")
    print(f"  - Version-Resemblance Chunks (Jaccard >= 0.30):  {counts['version_resemblance']}")
    print(f"  - Boundary-Shift Chunks (Partial overlap):      {counts['boundary_shift_overlap']}")

    print("\n" + "-" * 80)
    print(" DETAILED AUDIT OF MODIFIED CHUNKS")
    print("-" * 80)
    for a in report["alignment_audits"]:
        if a["category"] != PairCategory.EXACT_DUPLICATE.value:
            print(f" Chunk {a['chunk_idx']:2d}: v1 global {a['v1_global_id']:2d} vs v2 global {a['v2_global_id']:2d}")
            print(f"   v1 Range: {a['v1_offset_len']} | v2 Range: {a['v2_offset_len']}")
            print(f"   Line Jaccard: {a['line_jaccard']:.4f} | Category: {a['category']}")
            print(f"   Analysis: {a['description']}\n")

    print("-" * 80)
    print(" FEASIBILITY & DATA LEAKAGE ASSESSMENT")
    print("-" * 80)
    feas = report["feasibility"]
    print(f"Can split safely without data leakage: {feas['can_split_safely']}")
    print(f"Trustworthy edited pairs count:        {feas['trustworthy_resemblance_pairs_count']} (Required: >= {feas['min_required_pairs']})")
    print("\nRecommendations & Action Plan:")
    for rec in feas["recommendations"]:
        print(f"  * {rec}")
    print("=" * 80)


if __name__ == "__main__":
    main()
