# Augenblick — abctokz
"""Data manifest: tracks corpus files and their metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from abctokz.utils.hashing import sha256_file
from abctokz.utils.io import load_json, save_json


@dataclass
class CorpusEntry:
    """Metadata for a single corpus file.

    Attributes:
        path: Absolute or relative path to the file.
        language: Language tag (e.g. ``"hi"``, ``"en"``).
        n_lines: Number of non-empty lines.
        checksum: SHA-256 digest.
    """

    path: str
    language: str
    n_lines: int = 0
    checksum: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dict."""
        return {
            "path": self.path,
            "language": self.language,
            "n_lines": self.n_lines,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusEntry":
        """Deserialise from dict."""
        return cls(
            path=data["path"],
            language=data.get("language", ""),
            n_lines=data.get("n_lines", 0),
            checksum=data.get("checksum", ""),
        )


@dataclass
class DataManifest:
    """Collection of :class:`CorpusEntry` objects.

    Attributes:
        entries: List of corpus file entries.
    """

    entries: list[CorpusEntry] = field(default_factory=list)

    def add(self, path: str | Path, language: str = "") -> CorpusEntry:
        """Add a corpus file, computing its checksum and line count.

        Args:
            path: Path to the corpus file.
            language: Language tag.

        Returns:
            The created :class:`CorpusEntry`.
        """
        p = Path(path)
        n_lines = sum(1 for ln in p.open(encoding="utf-8") if ln.strip())
        checksum = sha256_file(p)
        entry = CorpusEntry(path=str(p), language=language, n_lines=n_lines, checksum=checksum)
        self.entries.append(entry)
        return entry

    def save(self, path: str | Path) -> None:
        """Save manifest to a JSON file.

        Args:
            path: Destination JSON path.
        """
        save_json({"entries": [e.to_dict() for e in self.entries]}, path)

    @classmethod
    def load(cls, path: str | Path) -> "DataManifest":
        """Load manifest from a JSON file.

        Args:
            path: Source JSON path.

        Returns:
            Loaded :class:`DataManifest`.
        """
        data = load_json(path)
        entries = [CorpusEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(entries=entries)

    @property
    def total_lines(self) -> int:
        """Total number of lines across all corpus entries."""
        return sum(e.n_lines for e in self.entries)

    @property
    def languages(self) -> list[str]:
        """Unique language tags present in the manifest."""
        return sorted({e.language for e in self.entries if e.language})
