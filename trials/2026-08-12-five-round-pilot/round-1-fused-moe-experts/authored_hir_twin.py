"""Executable runtime twin used to validate ``authored_hir.py``."""

import torch
import torch.nn.functional as F

import authored_hir as sem
from tilefoundry.runtime import runtime_func, runtime_module


@runtime_module(sem.RoutedExpertsCheck)
class RoutedExpertsTwin:
    @runtime_func
    def routed_experts(
        self,
        hidden,
        topk_weights,
        topk_ids,
        w_gate_up,
        w_down,
    ):
        flat_ids = topk_ids.reshape(-1).to(torch.int64)
        selected_in = w_gate_up.index_select(0, flat_ids).reshape(
            sem.T,
            sem.K,
            2 * sem.F,
            sem.H,
        )
        hidden_col = hidden.float().reshape(sem.T, 1, sem.H, 1)
        both = torch.matmul(selected_in.float(), hidden_col).reshape(
            sem.T,
            sem.K,
            2 * sem.F,
        )
        gate, up = both[..., : sem.F], both[..., sem.F :]
        inner = F.silu(gate) * up
        selected_down = w_down.index_select(0, flat_ids).reshape(
            sem.T,
            sem.K,
            sem.H,
            sem.F,
        )
        down = torch.matmul(selected_down.float(), inner.unsqueeze(-1)).squeeze(-1)
        mixed = (down * topk_weights.float().unsqueeze(-1)).sum(dim=1)
        return (mixed * sem.ROUTED_SCALING_FACTOR).to(hidden.dtype)
