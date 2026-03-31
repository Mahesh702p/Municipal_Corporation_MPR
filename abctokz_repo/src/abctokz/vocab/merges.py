# Augenblick — abctokz
"""BPE merge rules storage and serialization."""

from __future__ import annotations

from abctokz.types import MergePair, MergeRules


class MergeTable:
    """Stores and indexes BPE merge rules for fast lookup.

    Merge rules are stored in ranked order; lower rank = higher priority
    (applied first). The table supports O(1) rank lookup by pair.

    Args:
        rules: Ordered list of ``((a, b), merged)`` tuples representing the
            merge rules learned during BPE training.

    Example::

        table = MergeTable([(("h", "e"), "he"), (("he", "l"), "hel")])
        assert table.get_rank(("h", "e")) == 0
        assert table.merge_result(("h", "e")) == "he"
    """

    def __init__(self, rules: MergeRules) -> None:
        self._rules: MergeRules = list(rules)
        # pair -> (rank, merged_token)
        self._index: dict[MergePair, tuple[int, str]] = {
            pair: (rank, merged) for rank, (pair, merged) in enumerate(rules)
        }

    @property
    def rules(self) -> MergeRules:
        """The ordered merge rules."""
        return list(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    def get_rank(self, pair: MergePair) -> int | None:
        """Return the rank of *pair*, or ``None`` if not a learned merge.

        Args:
            pair: ``(left, right)`` token pair.

        Returns:
            Integer rank (0 = highest priority), or ``None``.
        """
        entry = self._index.get(pair)
        return None if entry is None else entry[0]

    def merge_result(self, pair: MergePair) -> str | None:
        """Return the merged token string for *pair*, or ``None``.

        Args:
            pair: ``(left, right)`` token pair.

        Returns:
            Merged token string, or ``None`` if *pair* is not a learned merge.
        """
        entry = self._index.get(pair)
        return None if entry is None else entry[1]

    def __contains__(self, pair: MergePair) -> bool:
        return pair in self._index

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_list(self) -> list[list[str]]:
        """Serialise to a list of ``[a, b, merged]`` triples."""
        return [[a, b, merged] for (a, b), merged in self._rules]

    @classmethod
    def from_list(cls, data: list[list[str]]) -> "MergeTable":
        """Deserialise from a list of ``[a, b, merged]`` triples.

        Args:
            data: Serialised merge rules.

        Returns:
            New :class:`MergeTable`.
        """
        rules: MergeRules = [((row[0], row[1]), row[2]) for row in data]
        return cls(rules)

    def to_text(self) -> str:
        """Serialise to a text format (one merge per line: ``a b merged``).

        The first line is a version/header comment.

        Returns:
            Multi-line string.
        """
        lines = ["#version: abctokz-merges-1.0"]
        for (a, b), merged in self._rules:
            lines.append(f"{a} {b} {merged}")
        return "\n".join(lines)

    @classmethod
    def from_text(cls, text: str) -> "MergeTable":
        """Deserialise from the text format.

        Args:
            text: Multi-line string (lines starting with ``#`` are comments).

        Returns:
            New :class:`MergeTable`.
        """
        rules: MergeRules = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ", 2)
            if len(parts) < 3:
                continue
            rules.append(((parts[0], parts[1]), parts[2]))
        return cls(rules)
