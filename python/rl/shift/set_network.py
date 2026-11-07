from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SetScorer(nn.Module):
    """
    Scores every candidate row while letting each row look at the others.

    One attention layer whose logits are biased by minus the travel time
    between the two candidates, so a row attends mostly to the orders near it.
    The score of a row comes from its own embedding and what it attended to.
    """

    uses_sets = True

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int = 128,
    ) -> None:
        super().__init__()

        self.hidden_dimension = hidden_dimension

        self.embed = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
        )
        self.query = nn.Linear(hidden_dimension, hidden_dimension)
        self.key = nn.Linear(hidden_dimension, hidden_dimension)
        self.value = nn.Linear(hidden_dimension, hidden_dimension)

        # softplus(distance_scale) is how strongly distance suppresses attention.
        self.distance_scale = nn.Parameter(torch.tensor(1.0))

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """
        features [B, W, F], valid [B, W] (False for padding), pairs [B, W, W].
        Returns scores [B, W]; padded rows are meaningless.
        """

        hidden = self.embed(features)

        logits = self.query(hidden) @ self.key(hidden).transpose(1, 2)
        logits = logits / math.sqrt(self.hidden_dimension)
        logits = logits - F.softplus(self.distance_scale) * pairs
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))

        context = torch.softmax(logits, dim=-1) @ self.value(hidden)

        return self.head(torch.cat([hidden, context], dim=-1)).squeeze(-1)
