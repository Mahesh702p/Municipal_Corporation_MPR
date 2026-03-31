# Augenblick — abctokz
"""Special-token post-processor: prepends BOS and/or appends EOS."""

from __future__ import annotations

from abctokz.processors.base import PostProcessor
from abctokz.types import Encoding


class SpecialTokensPostProcessor(PostProcessor):
    """Add BOS and/or EOS tokens to an encoding.

    Args:
        bos_token: BOS token string. If ``None``, no BOS is added.
        bos_id: BOS token ID.
        eos_token: EOS token string. If ``None``, no EOS is added.
        eos_id: EOS token ID.

    Example::

        pp = SpecialTokensPostProcessor(bos_token="<s>", bos_id=1, eos_token="</s>", eos_id=2)
        enc = Encoding(ids=[5, 6], tokens=["hello", "world"])
        result = pp.process(enc)
        assert result.ids == [1, 5, 6, 2]
        assert result.tokens == ["<s>", "hello", "world", "</s>"]
    """

    def __init__(
        self,
        bos_token: str | None = None,
        bos_id: int = 1,
        eos_token: str | None = None,
        eos_id: int = 2,
    ) -> None:
        self._bos_token = bos_token
        self._bos_id = bos_id
        self._eos_token = eos_token
        self._eos_id = eos_id

    def process(self, encoding: Encoding, pair: Encoding | None = None) -> Encoding:
        """Prepend BOS and/or append EOS tokens to *encoding*.

        Args:
            encoding: Input encoding.
            pair: Unused (single-sequence processor).

        Returns:
            New encoding with special tokens added.
        """
        ids = list(encoding.ids)
        tokens = list(encoding.tokens)
        offsets = list(encoding.offsets)
        special_mask = list(encoding.special_tokens_mask)
        attn_mask = list(encoding.attention_mask)

        if self._bos_token is not None:
            ids = [self._bos_id] + ids
            tokens = [self._bos_token] + tokens
            offsets = [(0, 0)] + offsets
            special_mask = [1] + special_mask
            attn_mask = [1] + attn_mask

        if self._eos_token is not None:
            ids = ids + [self._eos_id]
            tokens = tokens + [self._eos_token]
            offsets = offsets + [(0, 0)]
            special_mask = special_mask + [1]
            attn_mask = attn_mask + [1]

        return Encoding(
            ids=ids,
            tokens=tokens,
            offsets=offsets,
            special_tokens_mask=special_mask,
            attention_mask=attn_mask,
        )
