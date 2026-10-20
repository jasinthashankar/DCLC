"""
src/rag/document_parser.py
--------------------------
DCLC-RAG Phase 6B — Document Parser.

Reads a file from disk and returns a ParsedDocument containing:
  - raw_bytes  : the exact bytes on disk (passed to FastCDC)
  - text_content: UTF-8 decoded text (with latin-1 fallback) for future RAG retrieval
  - encoding_detected: the codec that successfully decoded the file

Supported extensions: .txt, .md, .py
Raises UnsupportedFileError for any other extension.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


class UnsupportedFileError(Exception):
    """Raised when the file extension is not supported by the document parser."""


# Extensions that the parser handles in this phase.
_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".py"})

# Encodings tried in order.
_ENCODING_PREFERENCE: tuple[str, ...] = ("utf-8", "latin-1")


@dataclass
class ParsedDocument:
    """
    Holds the parsed content of a single document file.

    Attributes
    ----------
    filename       : Original filename (basename only).
    extension      : Lowercase extension including the leading dot, e.g. ".txt".
    raw_bytes      : Exact bytes read from disk. Used for FastCDC chunking and SHA-256.
    text_content   : Human-readable text decoded from raw_bytes.
    encoding_detected: The codec used to successfully decode the file, e.g. "utf-8".
    content_sha256 : SHA-256 hex digest of raw_bytes; computed automatically.
    """

    filename: str
    extension: str
    raw_bytes: bytes
    text_content: str
    encoding_detected: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "content_sha256", hashlib.sha256(self.raw_bytes).hexdigest()
        )


def parse_document(path: "Path | str") -> ParsedDocument:
    """
    Parse a document file and return a ParsedDocument.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path to the file to parse.

    Returns
    -------
    ParsedDocument

    Raises
    ------
    UnsupportedFileError
        If the file extension is not in {.txt, .md, .py}.
    UnsupportedFileError
        If the file cannot be decoded as text after all encoding fallbacks.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    extension = path.suffix.lower()

    if extension not in _SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Extension '{extension}' is not supported. "
            f"Supported extensions: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    raw_bytes: bytes = path.read_bytes()

    # Empty file: return empty text with utf-8 as detected encoding.
    if not raw_bytes:
        return ParsedDocument(
            filename=path.name,
            extension=extension,
            raw_bytes=raw_bytes,
            text_content="",
            encoding_detected="utf-8",
        )

    # Try each encoding in preference order.
    for codec in _ENCODING_PREFERENCE:
        try:
            text_content = raw_bytes.decode(codec)
            return ParsedDocument(
                filename=path.name,
                extension=extension,
                raw_bytes=raw_bytes,
                text_content=text_content,
                encoding_detected=codec,
            )
        except (UnicodeDecodeError, LookupError):
            continue

    raise UnsupportedFileError(
        f"File '{path.name}' could not be decoded as text using any supported "
        f"encoding: {list(_ENCODING_PREFERENCE)}. "
        "Only plain-text files are supported in Phase 6B."
    )
