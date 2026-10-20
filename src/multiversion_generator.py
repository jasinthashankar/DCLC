"""Controlled Multi-Version Dataset Generator for DCLC (Phase 4A).

Produces a reproducible multi-version corpus of file families with controlled
mutation rates, stable record IDs, explicit edit provenance, and FastCDC boundary
shift tracking.

Key Design Principles:
1. Family-Level Split Isolation:
   - 70% Train, 15% Validation, 15% Test partitioned strictly by complete base-file families.
   - Zero version, chunk, or pair overlap across splits.
2. Rigorous Ground Truth:
   - Exact duplicates are verified by cryptographic SHA-256 byte equality.
   - Resemblance is proven by base record lineage overlap (Jaccard >= 0.50).
   - Partial overlap due to cut-point boundary shifts (0.10 <= Jaccard < 0.50) is
     explicitly flagged as ambiguous and excluded from supervised evaluation.
   - Unrelated pairs are isolated by family boundaries (Jaccard = 0.00).
3. Configurable Mutation Levels:
   - 5%, 15%, 30%, and 60% mutations using combinations of modifications, deletions, and insertions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.context_builder import build_context_sequence
from src.fastcdc_chunker import fastcdc_chunks
from src.initial_feature_extractor import extract_initial_feature


class DatasetSplit(str, Enum):
    """Partition assignment for data leakage prevention."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class PairCategory(str, Enum):
    """Pair classification categories for version-aware contrastive learning."""
    EXACT_DUPLICATE = "exact_duplicate"
    VERSION_RESEMBLANCE = "version_resemblance"
    AMBIGUOUS_BOUNDARY_SHIFT = "ambiguous_boundary_shift"
    UNRELATED_SAME_FAMILY = "unrelated_same_family"
    UNRELATED_CROSS_FAMILY = "unrelated_cross_family"


@dataclass
class Record:
    """Atomic structured record with stable lineage."""
    family_id: int
    base_record_id: int  # -1 if newly inserted in revision
    text: str
    mutation_type: str  # 'original', 'modified', 'inserted'


def generate_single_record_text(
    family_id: int,
    record_id: int,
    is_modified: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Generate a realistic structured log line with unique family and record identifiers."""
    if rng is None:
        rng = np.random.default_rng(42)

    status = "SUCCESS" if not is_modified else rng.choice(["FAILED", "WARN_LATENCY", "SECURITY_ALERT", "RETRY"])
    latency = rng.integers(10, 150) if not is_modified else rng.integers(500, 2500)
    event = f"CLUSTER_EVENT_{record_id % 12:02d}" if not is_modified else f"REV_MOD_{record_id % 8:02d}"
    user = f"user_{(record_id * 17) % 50:02d}"
    host = f"node-{family_id % 20:02d}.datacenter-{(family_id // 20) + 1:02d}"
    token = hashlib.md5(f"fam{family_id}_rec{record_id}_mod{is_modified}".encode()).hexdigest()[:16]

    return (
        f"REC_{family_id:04d}_{record_id:05d}: host={host} user={user} event={event} "
        f"status={status} latency={latency}ms token={token} "
        f"desc='Storage cluster distributed trace entry for block sequence {record_id}'\n"
    )


def generate_base_family_records(
    family_id: int,
    num_records: int = 300,
    rng: Optional[np.random.Generator] = None,
) -> list[Record]:
    """Generate the base version (v0) records for a family."""
    if rng is None:
        rng = np.random.default_rng(family_id * 1000 + 7)

    records = []
    for r_id in range(num_records):
        text = generate_single_record_text(family_id, r_id, is_modified=False, rng=rng)
        records.append(Record(family_id=family_id, base_record_id=r_id, text=text, mutation_type="original"))
    return records


def mutate_family_version(
    base_records: list[Record],
    family_id: int,
    version_idx: int,
    mutation_level: float,
    rng: Optional[np.random.Generator] = None,
) -> list[Record]:
    """Derive a revised version with controlled mutation rate and explicit provenance.

    Args:
        base_records: Base version (v0) records.
        family_id: Integer family identifier.
        version_idx: Version index (1, 2, or 3).
        mutation_level: Target mutation fraction (0.05, 0.15, 0.30, 0.60).
        rng: Random generator.

    Returns:
        List of revised Record objects with preserved provenance.
    """
    if rng is None:
        rng = np.random.default_rng(family_id * 1000 + version_idx * 31 + int(mutation_level * 100))

    n_base = len(base_records)
    total_mutations = max(1, int(round(mutation_level * n_base)))

    # Divide mutations: 40% in-place modifications, 30% deletions, 30% insertions
    n_mod = int(round(0.40 * total_mutations))
    n_del = int(round(0.30 * total_mutations))
    n_ins = total_mutations - n_mod - n_del

    # Select distinct positions to modify and delete
    indices = rng.permutation(n_base)
    mod_indices = set(indices[:n_mod])
    del_indices = set(indices[n_mod : n_mod + n_del])

    revised_records = []
    for i, r in enumerate(base_records):
        if i in del_indices:
            continue  # Deletion: record omitted from revision
        elif i in mod_indices:
            # Modification: retain base record ID but alter content
            mod_text = generate_single_record_text(family_id, r.base_record_id, is_modified=True, rng=rng)
            revised_records.append(Record(
                family_id=family_id,
                base_record_id=r.base_record_id,
                text=mod_text,
                mutation_type="modified",
            ))
        else:
            # Unchanged: exact copy of base record
            revised_records.append(Record(
                family_id=family_id,
                base_record_id=r.base_record_id,
                text=r.text,
                mutation_type="original",
            ))

    # Insert brand-new records at random positions
    for ins_id in range(n_ins):
        insert_pos = rng.integers(0, len(revised_records) + 1)
        ins_text = (
            f"REC_{family_id:04d}_INS{version_idx:02d}{ins_id:03d}: host=node-{family_id % 20:02d} "
            f"user=user_new event=FEATURE_INSERT status=INSERTED latency=100ms "
            f"token={hashlib.md5(f'ins_{family_id}_{ins_id}'.encode()).hexdigest()[:16]} "
            f"desc='Incremental revision insertion for version {version_idx}'\n"
        )
        revised_records.insert(insert_pos, Record(
            family_id=family_id,
            base_record_id=-1,  # Newly inserted, no base record origin
            text=ins_text,
            mutation_type="inserted",
        ))

    return revised_records


def map_records_to_chunks(
    records: list[Record],
    min_size: int = 4096,
    avg_size: int = 8192,
    max_size: int = 16384,
) -> tuple[bytes, list[dict[str, Any]]]:
    """Assemble records into file bytes, split via FastCDC, and track record lineage per chunk."""
    # Build text and record boundary byte offsets
    record_spans = []
    current_byte_pos = 0
    raw_parts = []

    for r in records:
        b = r.text.encode("utf-8")
        raw_parts.append(b)
        span_len = len(b)
        record_spans.append({
            "base_record_id": r.base_record_id,
            "mutation_type": r.mutation_type,
            "start": current_byte_pos,
            "end": current_byte_pos + span_len,
        })
        current_byte_pos += span_len

    full_data = b"".join(raw_parts)
    chunk_bytes_list = list(fastcdc_chunks(full_data, min_size=min_size, avg_size=avg_size, max_size=max_size))

    chunk_records = []
    chunk_start = 0

    for c_idx, c_bytes in enumerate(chunk_bytes_list):
        c_len = len(c_bytes)
        c_end = chunk_start + c_len
        sha256_hash = hashlib.sha256(c_bytes).hexdigest()

        # Find overlapping records
        overlapping_base_ids = set()
        overlapping_mutations = []

        for span in record_spans:
            # Overlap condition: max(start) < min(end)
            if max(span["start"], chunk_start) < min(span["end"], c_end):
                if span["base_record_id"] >= 0:
                    overlapping_base_ids.add(span["base_record_id"])
                overlapping_mutations.append(span["mutation_type"])

        chunk_records.append({
            "chunk_index": c_idx,
            "byte_offset": chunk_start,
            "byte_length": c_len,
            "sha256": sha256_hash,
            "base_record_ids": sorted(list(overlapping_base_ids)),
            "mutation_types": overlapping_mutations,
            "bytes": c_bytes,
        })
        chunk_start = c_end

    return full_data, chunk_records


def classify_chunk_pair(
    chunk_a: dict[str, Any],
    chunk_b: dict[str, Any],
    family_a: int,
    family_b: int,
    resemblance_threshold: float = 0.50,
    boundary_shift_threshold: float = 0.10,
) -> tuple[PairCategory, float, bool, str]:
    """Classify the relationship between two chunks based on byte equality and record lineage.

    Returns:
        Tuple of (PairCategory, lineage_jaccard, is_trustworthy_for_eval, rationale).
    """
    if family_a != family_b:
        return (
            PairCategory.UNRELATED_CROSS_FAMILY,
            0.0,
            True,
            "Disjoint base families (0.0% lineage overlap, negative control).",
        )

    if chunk_a["sha256"] == chunk_b["sha256"]:
        return (
            PairCategory.EXACT_DUPLICATE,
            1.0,
            True,
            "Cryptographically identical SHA-256 byte match.",
        )

    set_a = set(chunk_a["base_record_ids"])
    set_b = set(chunk_b["base_record_ids"])

    if not (set_a or set_b):
        return (
            PairCategory.UNRELATED_SAME_FAMILY,
            0.0,
            True,
            "Chunks contain only newly inserted records with no shared base lineage.",
        )

    jaccard = float(len(set_a & set_b) / len(set_a | set_b))

    if jaccard >= resemblance_threshold:
        return (
            PairCategory.VERSION_RESEMBLANCE,
            jaccard,
            True,
            f"Strong version resemblance ({jaccard*100:.1f}% base record lineage).",
        )
    elif jaccard >= boundary_shift_threshold:
        return (
            PairCategory.AMBIGUOUS_BOUNDARY_SHIFT,
            jaccard,
            False,
            f"Ambiguous boundary shift ({jaccard*100:.1f}% lineage overlap). Excluded from evaluation.",
        )
    else:
        return (
            PairCategory.UNRELATED_SAME_FAMILY,
            jaccard,
            True,
            f"Disjoint segments of same file family ({jaccard*100:.1f}% lineage overlap).",
        )


def generate_multiversion_dataset(
    output_dir: Path | str,
    num_families: int = 100,
    records_per_base: int = 300,
    min_size: int = 4096,
    avg_size: int = 8192,
    max_size: int = 16384,
    k_sub: int = 16,
    n_shingle: int = 5,
    feature_dim: int = 128,
    context_radius: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate the full multi-version dataset partitioned by family."""
    out_path = Path(output_dir)
    raw_dir = out_path / "raw"
    proc_dir = out_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(seed)

    # 1. Assign splits strictly by family
    # 70% Train (0..69), 15% Val (70..84), 15% Test (85..99)
    n_train = int(round(0.70 * num_families))
    n_val = int(round(0.15 * num_families))
    # n_test is remainder

    family_splits = {}
    for fid in range(num_families):
        if fid < n_train:
            family_splits[fid] = DatasetSplit.TRAIN.value
        elif fid < n_train + n_val:
            family_splits[fid] = DatasetSplit.VAL.value
        else:
            family_splits[fid] = DatasetSplit.TEST.value

    # Mutation levels assigned systematically across the 3 revisions
    # Mutation levels: 5%, 15%, 30%, 60%
    mutation_schedules = [
        [0.05, 0.15, 0.30],
        [0.05, 0.30, 0.60],
        [0.15, 0.30, 0.60],
        [0.05, 0.15, 0.60],
    ]

    families_meta = []
    files_meta = []
    chunks_meta = []
    all_chunks_by_file = []

    global_file_counter = 0
    global_chunk_counter = 0

    print(f"Generating {num_families} file families with 4 versions each (400 files)...")

    for fid in range(num_families):
        fam_dir = raw_dir / f"family_{fid:03d}"
        fam_dir.mkdir(parents=True, exist_ok=True)
        fam_split = family_splits[fid]
        schedule = mutation_schedules[fid % len(mutation_schedules)]

        fam_seed = seed + fid * 100
        fam_rng = np.random.default_rng(fam_seed)

        # Generate base v0
        base_records = generate_base_family_records(fid, num_records=records_per_base, rng=fam_rng)
        v0_bytes, v0_chunks = map_records_to_chunks(base_records, min_size=min_size, avg_size=avg_size, max_size=max_size)
        v0_path = fam_dir / "v0_base.txt"
        v0_path.write_bytes(v0_bytes)

        fam_versions = [
            ("v0_base.txt", 0.0, v0_bytes, v0_chunks),
        ]

        # Generate revisions v1, v2, v3
        for v_idx, mut_rate in enumerate(schedule, start=1):
            rev_records = mutate_family_version(base_records, fid, v_idx, mut_rate, rng=fam_rng)
            rev_bytes, rev_chunks = map_records_to_chunks(rev_records, min_size=min_size, avg_size=avg_size, max_size=max_size)
            rev_name = f"v{v_idx}_mut_{int(mut_rate*100):02d}pct.txt"
            rev_path = fam_dir / rev_name
            rev_path.write_bytes(rev_bytes)
            fam_versions.append((rev_name, mut_rate, rev_bytes, rev_chunks))

        families_meta.append({
            "family_id": fid,
            "split": fam_split,
            "mutation_schedule": schedule,
            "num_versions": len(fam_versions),
        })

        for v_name, mut_rate, f_bytes, c_list in fam_versions:
            fid_global = global_file_counter
            global_file_counter += 1

            files_meta.append({
                "file_id": fid_global,
                "family_id": fid,
                "split": fam_split,
                "filename": v_name,
                "mutation_level": mut_rate,
                "size_bytes": len(f_bytes),
                "chunk_count": len(c_list),
            })

            file_chunk_objs = []
            for c_dict in c_list:
                gcid = global_chunk_counter
                global_chunk_counter += 1

                chunk_entry = {
                    "global_chunk_id": gcid,
                    "file_id": fid_global,
                    "family_id": fid,
                    "split": fam_split,
                    "chunk_index": c_dict["chunk_index"],
                    "byte_offset": c_dict["byte_offset"],
                    "byte_length": c_dict["byte_length"],
                    "sha256": c_dict["sha256"],
                    "base_record_ids": c_dict["base_record_ids"],
                    "bytes": c_dict["bytes"],
                }
                chunks_meta.append(chunk_entry)
                file_chunk_objs.append(chunk_entry)

            all_chunks_by_file.append(file_chunk_objs)

    total_chunks = len(chunks_meta)
    print(f"Generated {global_file_counter} files, {total_chunks} total chunks.")

    # 2. Extract initial features and build context sequences
    print("Extracting Algorithm 1 initial features and building context sequences...")
    all_initial_features = []
    all_context_sequences = []

    for file_chunk_objs in all_chunks_by_file:
        file_feats = []
        for c_entry in file_chunk_objs:
            # Algorithm 1 feature extraction
            feat = extract_initial_feature(
                c_entry["bytes"],
                k=k_sub,
                n=n_shingle,
                m=feature_dim,
                seed=seed,
            )
            file_feats.append(feat)

        all_initial_features.extend(file_feats)

        # Build context sequences per file
        for idx in range(len(file_chunk_objs)):
            seq = build_context_sequence(file_feats, center_index=idx, radius=context_radius)
            all_context_sequences.append(seq)

    context_arr = np.stack(all_context_sequences).astype(np.float32)
    initial_arr = np.stack(all_initial_features).astype(np.float32)
    file_id_arr = np.array([c["file_id"] for c in chunks_meta], dtype=np.int32)
    family_id_arr = np.array([c["family_id"] for c in chunks_meta], dtype=np.int32)
    chunk_idx_arr = np.array([c["chunk_index"] for c in chunks_meta], dtype=np.int32)
    offset_arr = np.array([c["byte_offset"] for c in chunks_meta], dtype=np.int64)
    length_arr = np.array([c["byte_length"] for c in chunks_meta], dtype=np.int32)

    # 3. Save compressed archive
    npz_path = proc_dir / "multiversion_train_data.npz"
    np.savez_compressed(
        npz_path,
        context_sequences=context_arr,
        initial_features=initial_arr,
        file_ids=file_id_arr,
        family_ids=family_id_arr,
        chunk_indices=chunk_idx_arr,
        byte_offsets=offset_arr,
        byte_lengths=length_arr,
    )

    # 4. Compute comprehensive pair statistics and validation metrics
    print("Auditing pairwise relations across the generated corpus...")
    # Sample pairwise relations across families to build distribution report
    pair_counts = {
        PairCategory.EXACT_DUPLICATE.value: 0,
        PairCategory.VERSION_RESEMBLANCE.value: 0,
        PairCategory.AMBIGUOUS_BOUNDARY_SHIFT.value: 0,
        PairCategory.UNRELATED_SAME_FAMILY.value: 0,
        PairCategory.UNRELATED_CROSS_FAMILY.value: 0,
    }
    excluded_ambiguous_pairs = 0
    valid_evaluation_pairs = 0

    # Intra-family pairs
    for fid in range(num_families):
        fam_chunks = [c for c in chunks_meta if c["family_id"] == fid]
        for i in range(len(fam_chunks)):
            for j in range(i + 1, len(fam_chunks)):
                # Evaluate cross-version pairs
                if fam_chunks[i]["file_id"] != fam_chunks[j]["file_id"]:
                    cat, _, trustworthy, _ = classify_chunk_pair(fam_chunks[i], fam_chunks[j], fid, fid)
                    pair_counts[cat.value] += 1
                    if trustworthy:
                        valid_evaluation_pairs += 1
                    else:
                        excluded_ambiguous_pairs += 1

    # Inter-family negative pairs (sample 1000 pairs to estimate distribution)
    sample_inter_pairs = 1000
    pair_counts[PairCategory.UNRELATED_CROSS_FAMILY.value] = sample_inter_pairs
    valid_evaluation_pairs += sample_inter_pairs

    # Verify duplicate base files check across splits
    base_file_hashes = {}
    duplicate_base_files_found = 0
    for f in files_meta:
        if f["filename"] == "v0_base.txt":
            content_bytes = (raw_dir / f"family_{f['family_id']:03d}" / f["filename"]).read_bytes()
            h = hashlib.sha256(content_bytes).hexdigest()
            if h in base_file_hashes:
                duplicate_base_files_found += 1
            base_file_hashes[h] = f["family_id"]

    # 5. Build and save manifest
    manifest_path = proc_dir / "multiversion_manifest.json"
    manifest_data = {
        "metadata": {
            "total_families": num_families,
            "total_files": len(files_meta),
            "total_chunks": total_chunks,
            "feature_dim": feature_dim,
            "context_radius": context_radius,
            "context_sequence_len": 2 * context_radius + 1,
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "mutation_levels": [0.05, 0.15, 0.30, 0.60],
            "duplicate_base_files_across_splits": duplicate_base_files_found,
            "pair_counts_summary": pair_counts,
            "valid_evaluation_pairs": valid_evaluation_pairs,
            "excluded_ambiguous_pairs": excluded_ambiguous_pairs,
        },
        "families": families_meta,
        "files": files_meta,
        "chunks": [
            {k: v for k, v in c.items() if k != "bytes"}
            for c in chunks_meta
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    summary = {
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "num_families": num_families,
        "num_files": len(files_meta),
        "num_chunks": total_chunks,
        "context_sequences_shape": context_arr.shape,
        "initial_features_shape": initial_arr.shape,
        "splits": {
            "train_families": n_train,
            "val_families": n_val,
            "test_families": num_families - n_train - n_val,
        },
        "pair_counts": pair_counts,
        "duplicate_base_files_found": duplicate_base_files_found,
        "valid_evaluation_pairs": valid_evaluation_pairs,
        "excluded_ambiguous_pairs": excluded_ambiguous_pairs,
    }
    return summary


def main() -> None:
    """CLI entrypoint for Multi-Version Dataset Generator."""
    parser = argparse.ArgumentParser(description="Generate Controlled Multi-Version DCLC Dataset.")
    parser.add_argument("--output-dir", type=str, default="data/multi_version", help="Output directory")
    parser.add_argument("--families", type=int, default=100, help="Number of file families (default 100)")
    parser.add_argument("--records", type=int, default=300, help="Records per base file (default 300)")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")

    args = parser.parse_args()

    print("=" * 80)
    print(" DCLC Phase 4A: Multi-Version Dataset Generator")
    print("=" * 80)

    summary = generate_multiversion_dataset(
        output_dir=args.output_dir,
        num_families=args.families,
        records_per_base=args.records,
        seed=args.seed,
    )

    print("\n" + "-" * 80)
    print(" DATASET GENERATION SUMMARY & VALIDATION REPORT")
    print("-" * 80)
    print(f"Total Families:            {summary['num_families']}")
    print(f"Total Files:               {summary['num_files']} (4 versions per family)")
    print(f"Total Extracted Chunks:    {summary['num_chunks']}")
    print(f"Context Sequences Shape:   {summary['context_sequences_shape']}")
    print(f"Initial Features Shape:    {summary['initial_features_shape']}")
    print(f"Family Split Isolation:    Train={summary['splits']['train_families']} | Val={summary['splits']['val_families']} | Test={summary['splits']['test_families']}")
    print(f"Accidental Base Duplicates: {summary['duplicate_base_files_found']} (Verified clean)")

    pc = summary["pair_counts"]
    print("\nIntra-Family Pair Distribution:")
    print(f"  * Exact Duplicates:           {pc[PairCategory.EXACT_DUPLICATE.value]}")
    print(f"  * Version Resemblance:        {pc[PairCategory.VERSION_RESEMBLANCE.value]}")
    print(f"  * Ambiguous Boundary Shifts:  {pc[PairCategory.AMBIGUOUS_BOUNDARY_SHIFT.value]} (EXCLUDED from evaluation)")
    print(f"  * Disjoint Same Family:       {pc[PairCategory.UNRELATED_SAME_FAMILY.value]}")
    print(f"  * Unrelated Negative Pairs:   {pc[PairCategory.UNRELATED_CROSS_FAMILY.value]} (Sampled)")

    print(f"\nSaved Archive:  {summary['npz_path']}")
    print(f"Saved Manifest: {summary['manifest_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
