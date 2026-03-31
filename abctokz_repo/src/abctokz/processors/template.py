# Augenblick — abctokz
"""Template post-processor for flexible special token insertion."""

from __future__ import annotations

from abctokz.processors.base import PostProcessor
from abctokz.types import Encoding


class TemplatePostProcessor(PostProcessor):
    """Construct an encoding according to a token template.

    A template is a list of items, where each item is either:
    - A special-token spec ``{"special": token_str, "id": token_id}``
    - A sequence reference ``{"sequence": "A"}`` or ``{"sequence": "B"}``

    Sequence ``"A"`` refers to the primary encoding; ``"B"`` refers to the
    optional pair encoding.

    Args:
        single: Template for a single sequence.
        pair: Template for a sequence pair. Required if pair encodings are used.

    Example::

        pp = TemplatePostProcessor(
            single=[{"special": "<s>", "id": 1}, {"sequence": "A"}, {"special": "</s>", "id": 2}],
        )
        enc = Encoding(ids=[5, 6], tokens=["hello", "world"], offsets=[(0,5),(6,11)],
                       special_tokens_mask=[0,0], attention_mask=[1,1])
        result = pp.process(enc)
        assert result.ids == [1, 5, 6, 2]
    """

    def __init__(
        self,
        single: list[dict[str, object]],
        pair: list[dict[str, object]] | None = None,
    ) -> None:
        self._single = single
        self._pair = pair

    def process(self, encoding: Encoding, pair: Encoding | None = None) -> Encoding:
        """Build an encoding from the template.

        Args:
            encoding: Primary sequence encoding.
            pair: Optional pair encoding.

        Returns:
            Template-expanded :class:`~abctokz.types.Encoding`.
        """
        template = self._pair if (pair is not None and self._pair) else self._single

        ids: list[int] = []
        tokens: list[str] = []
        offsets: list[tuple[int, int]] = []
        special_mask: list[int] = []
        attn_mask: list[int] = []

        for item in template:
            if "special" in item:
                ids.append(int(item["id"]))  # type: ignore[arg-type]
                tokens.append(str(item["special"]))
                offsets.append((0, 0))
                special_mask.append(1)
                attn_mask.append(1)
            elif "sequence" in item:
                seq = pair if item["sequence"] == "B" and pair is not None else encoding
                ids.extend(seq.ids)
                tokens.extend(seq.tokens)
                offsets.extend(seq.offsets if seq.offsets else [(0, 0)] * len(seq.ids))
                special_mask.extend(seq.special_tokens_mask if seq.special_tokens_mask else [0] * len(seq.ids))
                attn_mask.extend(seq.attention_mask if seq.attention_mask else [1] * len(seq.ids))

        return Encoding(
            ids=ids,
            tokens=tokens,
            offsets=offsets,
            special_tokens_mask=special_mask,
            attention_mask=attn_mask,
        )
