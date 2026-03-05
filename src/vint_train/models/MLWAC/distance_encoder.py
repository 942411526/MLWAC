"""
Distance Encoder Module
=======================
Encodes the relationship between an observation image and a goal image
using a dual-direction EfficientNet backbone and a Transformer self-attention layer.

Architecture:
    - Goal encoder      : EfficientNet-B0 (6-channel, obs+goal concatenated, bidirectional)
    - Obs encoder       : EfficientNet-B0 (3-channel, reserved for future use)
    - Positional encoding: Sinusoidal (max_seq_len=2)
    - Self-attention    : nn.TransformerEncoder (GELU, Pre-LN, 2 tokens)

All BatchNorm layers are replaced with GroupNorm for training stability.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from vint_train.models.vint.self_attention import PositionalEncoding


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class DistanceEncoder(nn.Module):
    """Encode the distance/relationship between an observation and a goal image.

    Produces two encodings — forward ``[obs | goal]`` and reverse ``[goal | obs]`` —
    then fuses them via a 2-token Transformer self-attention layer.

    Args:
        context_size:             Number of context frames (reserved for future use).
        obs_encoder:              EfficientNet variant string, e.g. ``"efficientnet-b0"``.
        obs_encoding_size:        Projected feature dimension for all tokens.
        mha_num_attention_heads:  Number of attention heads in the Transformer encoder.
        mha_num_attention_layers: Number of Transformer encoder layers.
        mha_ff_dim_factor:        Feed-forward hidden-dim multiplier
                                  (``ff_dim = factor × obs_encoding_size``).
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
        # Observation encoder  (3-channel, reserved for upstream use)
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
        # Goal encoder  (6-channel: obs and goal concatenated along channels)
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
        # Sequence length = 2  (forward token + reverse token)
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
            norm_first=True,  # Pre-LN for training stability
        )
        self.sa_encoder = nn.TransformerEncoder(
            _sa_layer, num_layers=mha_num_attention_layers
        )

        # ------------------------------------------------------------------ #
        # Goal masking tensors
        # Convention: 0 = not masked, 1 = masked
        # ------------------------------------------------------------------ #
        self.goal_mask = torch.zeros((1, 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True                            # mask goal token
        self.no_mask  = torch.zeros((1, 2), dtype=torch.bool)  # no masking
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat(
            [
                1 - self.no_mask.float(),
                (1 - self.goal_mask.float())
                * ((self.context_size + 2) / (self.context_size + 1)),
            ],
            dim=0,
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
        """Encode an (observation, goal) image pair into a single feature vector.

        Args:
            obs_img:         Observation image tensor  ``(B, 3, H, W)``.
            goal_img:        Goal image tensor          ``(B, 3, H, W)``.
            input_goal_mask: Optional goal mask         ``(B,)`` long tensor
                             (0 = keep goal token, 1 = mask goal token).

        Returns:
            Aggregated distance embedding ``(B, obs_encoding_size)``.
        """
        device = obs_img.device

        # Forward encoding:  [obs | goal]  →  (B, 1, D)
        obsgoal_encoding = self._encode_pair(obs_img, goal_img)

        # Reverse encoding:  [goal | obs]  →  (B, 1, D)
        goalobs_encoding = self._encode_pair(goal_img, obs_img)

        assert obsgoal_encoding.shape[2] == self.goal_encoding_size

        # Concatenate to form a 2-token sequence: (B, 2, D)
        goal_encoding = torch.cat([obsgoal_encoding, goalobs_encoding], dim=1)

        # Optional goal masking
        src_key_padding_mask = None
        if input_goal_mask is not None:
            no_goal_mask = input_goal_mask.long().to(device)
            src_key_padding_mask = torch.index_select(
                self.all_masks.to(device), 0, no_goal_mask
            )

        # Positional encoding + self-attention
        goal_encoding        = self.positional_encoding(goal_encoding)
        goal_encoding_tokens = self.sa_encoder(
            goal_encoding, src_key_padding_mask=None
        )                                                          # (B, 2, D)

        # Mean pooling over the 2 tokens → (B, D)
        return torch.mean(goal_encoding_tokens, dim=1)

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _encode_pair(
        self, img_a: torch.Tensor, img_b: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate two images channel-wise, run the goal encoder, and compress.

        Args:
            img_a: First image  ``(B, 3, H, W)``.
            img_b: Second image ``(B, 3, H, W)``.

        Returns:
            Encoded token ``(B, 1, goal_encoding_size)``.
        """
        x = torch.cat([img_a, img_b], dim=1)          # (B, 6, H, W)
        x = self.goal_encoder.extract_features(x)
        x = self.goal_encoder._avg_pooling(x)
        if self.goal_encoder._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self.goal_encoder._dropout(x)
        x = self.compress_goal_enc(x)                 # (B, goal_encoding_size)
        return x.unsqueeze(1)                         # (B, 1, goal_encoding_size)


# ---------------------------------------------------------------------------
# GroupNorm utilities
# ---------------------------------------------------------------------------

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int = 16,
) -> nn.Module:
    """Replace every ``BatchNorm2d`` in *root_module* with ``GroupNorm``.

    Args:
        root_module:        The module to modify in-place.
        features_per_group: Channels per group
                            (``num_groups = num_features // features_per_group``).

    Returns:
        The modified *root_module*.
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
    """Recursively replace all sub-modules matching *predicate* with ``func(module)``.

    Args:
        root_module: Root of the module tree to traverse.
        predicate:   Returns ``True`` for modules to be replaced.
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
        src = (
            parent[int(key)]
            if isinstance(parent, nn.Sequential)
            else getattr(parent, key)
        )
        tgt = func(src)
        if isinstance(parent, nn.Sequential):
            parent[int(key)] = tgt
        else:
            setattr(parent, key, tgt)

    # Sanity check: ensure all targets were replaced
    remaining = [
        k for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(remaining) == 0, f"Failed to replace modules: {remaining}"

    return root_module
