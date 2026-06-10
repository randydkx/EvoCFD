import torch
import torch.nn as nn
from torch import Tensor
from typing import List

class FourierEmbeddings(nn.Module):
    def __init__(
        self, period: float, order: int
    ) -> None:
        super().__init__()
        self.frequencies = torch.arange(1, order + 1, dtype=torch.float32) / period
        self.frequencies = nn.Parameter(self.frequencies)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 1
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return x


class TemporalEmbeddings(nn.Module):
    def __init__(
        self,
        # t_mean: float,
        # t_std: float,
        order: List[int],
        trend: bool,
        d_embedding: int,
        periods: List[float] = None,
    ) -> None:
        super().__init__()
        self.order = order
        self.trend = trend
        self.periodicity = sum(order) > 0
        self.out_dim = (d_embedding if self.periodicity else 0) + (1 if self.trend else 0)
        self.periods = periods or [31557600.0, 2629800.0, 604800.0, 86400.0]
        
        # if self.trend:
        #     self.t_mean = t_mean
        #     self.t_std = t_std
        
        if self.periodicity:
            assert len(order) == 4, "The length of orders must be 4, corresponding to (year, month, week, day)"
            assert len(self.periods) == 4, "The length of periods must match the temporal order length"
            self.embeddings = nn.ModuleList([                                   # period priors
                FourierEmbeddings(self.periods[0], order[0]) if order[0] else None,
                FourierEmbeddings(self.periods[1], order[1]) if order[1] else None,
                FourierEmbeddings(self.periods[2], order[2]) if order[2] else None,
                FourierEmbeddings(self.periods[3], order[3]) if order[3] else None,
            ])
            self.embeddings = nn.ModuleList([embedding for embedding in self.embeddings if embedding is not None])
            self.linear = nn.Linear(2 * sum(order), d_embedding)
            self.relu = nn.ReLU()

    def forward(self, x, x_trend):
        # x_trend = (x[..., None] - self.t_mean) / self.t_std if self.trend else None
        x_trend = x_trend[..., None]
        if self.periodicity:
            x = torch.cat([module(x) for module in self.embeddings], dim=-1)
            x = self.linear(x)
            x = self.relu(x)
            x = torch.cat([x, x_trend], dim=-1) if x_trend is not None else x
        else:
            assert x_trend is not None, "forwards() will not be called if self.out_dim == 0"
            x = x_trend
        return x