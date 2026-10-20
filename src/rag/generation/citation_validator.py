"""
src/rag/generation/citation_validator.py
-----------------------------------------
DCLC-RAG Phase 6D — Application-Level Citation Extraction and Validation.

Guarantees:
1. Every citation tag in the generated response (e.g., [chunk_1]) is parsed.
2. Verified against the exact set of FormattedEvidenceChunks provided in the prompt.
3. Citations referencing non-existent or un-retrieved tags are flagged as INVALID.
4. Valid citations are enriched with application-verified provenance:
   doc_id, version_id, chunk_id, filename, chunk_index, byte_offset, byte_length.
5. Never invents metadata or guesses source positions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from src.rag.generation.prompt_template import FormattedEvidenceChunk


@dataclass
class VerifiedCitation:
    """Enriched, application-verified citation metadata."""
    tag: str                  # e.g. "[chunk_1]"
    chunk_id: str             # SHA-256 fingerprint
    doc_id: str               # Document identifier
    version_id: str           # Version identifier
    filename: str             # Source filename
    chunk_index: int          # Index in version
    byte_offset: int          # Starting byte offset in original document
    byte_length: int          # Length in bytes
    snippet: str              # Leading snippet for UI display
    is_valid: bool = True     # True if found in prompt context


@dataclass
class CitationValidationReport:
    """Report on all citations extracted from a generated answer."""
    raw_citations_found: List[str]
    valid_citations: List[VerifiedCitation]
    invalid_citations: List[str]
    has_valid_citations: bool
    citation_precision: float   # len(valid) / len(raw)


_CITATION_TAG_PATTERN = re.compile(r"\[(chunk_\d+)\]", re.IGNORECASE)


def validate_and_enrich_citations(
    generated_text: str,
    evidence_chunks: List[FormattedEvidenceChunk],
    target_version_id: Optional[str] = None,
) -> CitationValidationReport:
    """
    Extract citation tags from LLM response text, validate against evidence chunks,
    and enrich with verified source coordinates.

    Parameters
    ----------
    generated_text    : Raw text output from LLM.
    evidence_chunks   : Chunks included in the prompt context.
    target_version_id : Target version scope, if any.
    """
    raw_matches = _CITATION_TAG_PATTERN.findall(generated_text)
    raw_tags = [f"[{m.lower()}]" for m in raw_matches]

    # Map tag -> chunk
    chunk_by_tag: Dict[str, FormattedEvidenceChunk] = {
        c.tag.lower(): c for c in evidence_chunks
    }

    valid_citations: List[VerifiedCitation] = []
    invalid_citations: List[str] = []
    seen_valid_tags: Set[str] = set()

    for tag in raw_tags:
        if tag in chunk_by_tag:
            c = chunk_by_tag[tag]
            # Verify version scope constraint if target version requested
            if target_version_id is not None and c.version_id != target_version_id:
                invalid_citations.append(f"{tag} (version mismatch: {c.version_id} != {target_version_id})")
                continue

            if tag not in seen_valid_tags:
                seen_valid_tags.add(tag)
                snippet = c.text_content.strip().replace("\n", " ")[:120]
                valid_citations.append(
                    VerifiedCitation(
                        tag=c.tag,
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        version_id=c.version_id,
                        filename=c.filename,
                        chunk_index=c.chunk_index,
                        byte_offset=c.byte_offset,
                        byte_length=c.byte_length,
                        snippet=snippet,
                        is_valid=True,
                    )
                )
        else:
            invalid_citations.append(tag)

    total_citations = len(raw_tags)
    precision = (len(valid_citations) / total_citations) if total_citations > 0 else 1.0

    return CitationValidationReport(
        raw_citations_found=raw_tags,
        valid_citations=valid_citations,
        invalid_citations=invalid_citations,
        has_valid_citations=len(valid_citations) > 0,
        citation_precision=precision,
    )
