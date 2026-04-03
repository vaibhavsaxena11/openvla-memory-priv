"""
proprio_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous proprioceptive states.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ProprioTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_proprio: float = -1,
        max_proprio: float = 1,
        token_begin_idx: int = None,
    ) -> None:
        """
        Discretizes continuous proprioceptive states into N bins per dimension and maps to specified vocabulary tokens.

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_proprio: Minimum proprio value (for clipping, setting lower bound on bin interval).
        :param max_proprio: Maximum proprio value (for clipping, setting upper bound on bin interval).
        :param token_begin_idx: Starting index in vocabulary for proprio tokens. token_end_idx is inferred as token_begin_idx + bins.
        """
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_proprio = min_proprio
        self.max_proprio = max_proprio

        # Set token range - if not provided, use tokens before action tokens
        if token_begin_idx is not None:
            self.token_begin_idx = token_begin_idx
            self.token_end_idx = token_begin_idx + self.n_bins
            # Assert that token_end_idx is within vocabulary bounds
            assert (
                self.token_end_idx <= self.tokenizer.vocab_size
            ), f"token_end_idx ({self.token_end_idx}) exceeds vocabulary size ({self.tokenizer.vocab_size})"
        else:
            # Default: use tokens before the last (n_bins + 1) tokens (which are typically reserved for actions)
            self.token_end_idx = int(self.tokenizer.vocab_size - (self.n_bins + 1))
            self.token_begin_idx = self.token_end_idx - self.n_bins

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_proprio, max_proprio, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    def __call__(self, proprio: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin proprioceptive states to the specified vocabulary token range."""
        proprio = np.clip(proprio, a_min=float(self.min_proprio), a_max=float(self.max_proprio))
        discretized_proprio = np.digitize(proprio, self.bins)

        # Map discretized values to token IDs in the specified range
        # digitize returns values in [1, n_bins], we map to [token_begin_idx, token_begin_idx + n_bins - 1]
        token_ids = self.token_begin_idx + discretized_proprio - 1
        token_ids = np.clip(token_ids, a_min=self.token_begin_idx, a_max=self.token_end_idx - 1)

        # Handle single element vs. batch
        if len(token_ids.shape) == 1:
            return self.tokenizer.decode(token_ids.tolist())
        else:
            return self.tokenizer.batch_decode(token_ids.tolist())

    def decode_token_ids_to_proprio(self, proprio_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous proprioceptive states for discrete proprio token IDs.

        NOTE =>> Because of the way the proprio values are discretized w.r.t. the bins (and not the bin centers),
                 the digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self.bins has 256 values. Then self.bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We map token IDs back to these indices, then subtract 1 so that they
                    are between [0, 255]. There is still one index (i==255) that would cause an out-of-bounds error
                    if used to index into self.bin_centers. Therefore, if i==255, we subtract 1 from it so that it
                    just becomes the index of the last bin center. We implement this simply via clipping between
                    [0, 255 - 1].
        """
        # Map token IDs back to discretized indices
        discretized_proprio = proprio_token_ids - self.token_begin_idx + 1

        # Clip and adjust for bin centers indexing
        discretized_proprio = np.clip(discretized_proprio - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_proprio]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
