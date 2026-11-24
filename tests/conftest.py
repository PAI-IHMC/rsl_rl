# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared test fixtures for rsl_rl tests."""

from __future__ import annotations

import torch
from tensordict import TensorDict


def make_obs(num_envs: int, obs_dim: int = 8, device: str = "cpu") -> TensorDict:
    """Create an observation TensorDict with a single 'policy' key."""
    return TensorDict(
        {"policy": torch.randn(num_envs, obs_dim, device=device)},
        batch_size=[num_envs],
        device=device,
    )


def negate_augmentation(
    env: object,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Toy data-augmentation callable: concatenate the originals with their element-wise negation.

    Produces a 2x-augmented batch ordered as ``[original, mirrored]`` along the batch dimension.
    """
    out_obs = None
    if obs is not None:
        neg = TensorDict({k: -v for k, v in obs.items()}, batch_size=obs.batch_size)
        out_obs = torch.cat([obs, neg], dim=0)

    out_actions = None
    if actions is not None:
        out_actions = torch.cat([actions, -actions], dim=0)

    return out_obs, out_actions
