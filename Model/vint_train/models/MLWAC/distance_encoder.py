"""
Distance Encoder Module
=======================
Encodes the relationship between an observation image and a goal image
using a dual EfficientNet backbone and a Transformer self-attention layer.

Architecture:
    - Observation encoder  : EfficientNet-B0 (3-channel input)
    - Goal encoder         : EfficientNet-B0 (6-channel input, obs + goal concatenated)
    - Positional encoding  : Sinusoidal (max_seq_len=2)
    - Self-attention       : nn.TransformerEncoder (GELU, Pre-LN)

All BatchNorm layers are replaced with GroupNorm for training stability.
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from vint_train.models.vint.self_attention import PositionalEncoding


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class DistanceEncoder(nn.Module):
    """Encode the distance / relationship between an observation and a goal image.

    Args:
        context_size:            Number of context frames (reserved for future use).
        obs_encoder:             EfficientNet variant string, e.g. ``"efficientnet-b0"``.
        obs_encoding_size:       Projected feature dimension for both obs and goal tokens.
        mha_num_attention_heads: Number of attention heads in the Transformer encoder.
        mha_num_attention_layers:Number of Transformer encoder layers.
        mha_ff_dim_factor:       Feed-forward hidden-dim multiplier (ff_dim = factor × d_model).
    """

    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: str = "efficientnet-b0",
        obs_encoding_size: int = 512,
        mha_num_attention_heads: int = 2,
        mha_num_attention_layers: int = 2,
        mha_ff_dim_factor: int = 4,
    ) -> None:
        super().__init__()

        self.obs_encoding_size  = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size       = context_size

        # ------------------------------------------------------------------ #
        # Observation encoder  (3-channel: current frame)
        # ------------------------------------------------------------------ #
        backbone_family = obs_encoder.split("-")[0]
        if backbone_family != "efficientnet":
            raise NotImplementedError(
                f"Unsupported obs_encoder backbone: '{backbone_family}'. "
                "Only 'efficientnet-*' variants are supported."
            )
        self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)
        self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
        self.num_obs_features = self.obs_encoder._fc.in_features
        self.obs_encoder_type = "efficientnet"

        # ------------------------------------------------------------------ #
        # Goal encoder  (6-channel: obs concatenated with goal)
        # ------------------------------------------------------------------ #
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)
        self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
        self.num_goal_features = self.goal_encoder._fc.in_features

        # ------------------------------------------------------------------ #
        # Compression layers (project backbone features → encoding_size)
        # ------------------------------------------------------------------ #
        self.compress_obs_enc = (
            nn.Linear(self.num_obs_features, self.obs_encoding_size)
            if self.num_obs_features != self.obs_encoding_size
            else nn.Identity()
        )
        self.compress_goal_enc = (
            nn.Linear(self.num_goal_features, self.goal_encoding_size)
            if self.num_goal_features != self.goal_encoding_size
            else nn.Identity()
        )

        # ------------------------------------------------------------------ #
        # Positional encoding + Transformer self-attention
        # ------------------------------------------------------------------ #
        self.positional_encoding = PositionalEncoding(
            self.obs_encoding_size, max_seq_len=2
        )

        _sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=mha_num_attention_heads,
            dim_feedforward=mha_ff_dim_factor * self.obs_encoding_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.sa_encoder = nn.TransformerEncoder(
            _sa_layer, num_layers=mha_num_attention_layers
        )

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        input_goal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode the (observation, goal) image pair into a single feature vector.

        Args:
            obs_img:         Observation image tensor  ``(B, 3, H, W)``.
            goal_img:        Goal image tensor          ``(B, 3, H, W)``.
            input_goal_mask: Optional boolean mask      ``(B,)`` — reserved for future use.

        Returns:
            goal_encoding_tokens: Aggregated goal token ``(B, obs_encoding_size)``.
        """
        # Concatenate obs + goal along the channel dimension and encode
        obsgoal_img      = torch.cat([obs_img, goal_img], dim=1)          # (B, 6, H, W)
        obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        obsgoal_encoding = self.goal_encoder._avg_pooling(obsgoal_encoding)

        if self.goal_encoder._global_params.include_top:
            obsgoal_encoding = obsgoal_encoding.flatten(start_dim=1)
            obsgoal_encoding = self.goal_encoder._dropout(obsgoal_encoding)

        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)       # (B, goal_encoding_size)

        # Add sequence dimension: (B, D) → (B, 1, D)
        if obsgoal_encoding.dim() == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)

        assert obsgoal_encoding.shape[2] == self.goal_encoding_size, (
            f"Expected goal encoding size {self.goal_encoding_size}, "
            f"got {obsgoal_encoding.shape[2]}"
        )

        goal_encoding = obsgoal_encoding                                  # (B, 1, D)

        # Positional encoding + self-attention
        goal_encoding        = self.positional_encoding(goal_encoding)
        goal_encoding_tokens = self.sa_encoder(
            goal_encoding, src_key_padding_mask=None
        )                                                                  # (B, 1, D)

        # Aggregate over the sequence dimension
        goal_encoding_tokens = torch.mean(goal_encoding_tokens, dim=1)    # (B, D)

        return goal_encoding_tokens


# ---------------------------------------------------------------------------
# GroupNorm utilities
# ---------------------------------------------------------------------------

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int = 16,
) -> nn.Module:
    """Replace every ``BatchNorm2d`` in *root_module* with ``GroupNorm``.

    Args:
        root_module:       The module to modify in-place.
        features_per_group:Number of channels per group
                           (``num_groups = num_features // features_per_group``).

    Returns:
        The modified *root_module* (same object, modified in-place).
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features,
        ),
    )
    return root_module


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Recursively replace all sub-modules that satisfy *predicate* with *func(module)*.

    Args:
        root_module: Root of the module tree to traverse.
        predicate:   Returns ``True`` for modules that should be replaced.
        func:        Returns the replacement module given the original.

    Returns:
        The modified *root_module*.
    """
    if predicate(root_module):
        return func(root_module)

    target_keys = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]

    for *parent_path, key in target_keys:
        parent = (
            root_module.get_submodule(".".join(parent_path))
            if parent_path
            else root_module
        )
        src = parent[int(key)] if isinstance(parent, nn.Sequential) else getattr(parent, key)
        tgt = func(src)
        if isinstance(parent, nn.Sequential):
            parent[int(key)] = tgt
        else:
            setattr(parent, key, tgt)

    # Sanity check: ensure all target modules have been replaced
    remaining = [
        k for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)
    ]
    assert len(remaining) == 0, f"Failed to replace modules: {remaining}"

    return root_module