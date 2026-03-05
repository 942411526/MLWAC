import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable

from efficientnet_pytorch import EfficientNet
from vint_train.models.vint.self_attention import PositionalEncoding


class DistanceEncoder(nn.Module):
    """
    Distance-aware goal encoder for visual navigation.

    Encodes a pair of (observation, goal) images into a fixed-size embedding
    by computing both forward (obs->goal) and reverse (goal->obs) encodings,
    then fusing them via a Transformer self-attention layer.

    Args:
        context_size:             Number of context frames (unused in encoding, reserved for future).
        obs_encoder:              EfficientNet variant for observation encoding (e.g. "efficientnet-b0").
        obs_encoding_size:        Output dimensionality for all encodings.
        mha_num_attention_heads:  Number of heads in the Transformer encoder.
        mha_num_attention_layers: Number of Transformer encoder layers.
        mha_ff_dim_factor:        Feedforward dimension multiplier (ff_dim = factor * obs_encoding_size).
    """

    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        super().__init__()

        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size

        # ── Observation encoder (single image, 3-channel) ──────────────────────
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError(f"Unsupported obs_encoder: {obs_encoder}")

        # ── Goal encoder (obs + goal concatenated, 6-channel) ──────────────────
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)
        self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
        self.num_goal_features = self.goal_encoder._fc.in_features

        # ── Linear compression (if feature dim != target encoding size) ────────
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

        # ── Positional encoding + Transformer self-attention ───────────────────
        # Sequence length = 2 (forward token + reverse token)
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=2)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=mha_num_attention_heads,
            dim_feedforward=mha_ff_dim_factor * self.obs_encoding_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        # ── Goal masking tensors ────────────────────────────────────────────────
        # Convention: 0 = not masked, 1 = masked
        self.goal_mask = torch.zeros((1, 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True                          # mask the goal token
        self.no_mask = torch.zeros((1, 2), dtype=torch.bool)  # no masking
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat(
            [
                1 - self.no_mask.float(),
                (1 - self.goal_mask.float()) * ((self.context_size + 2) / (self.context_size + 1)),
            ],
            dim=0,
        )

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        input_goal_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            obs_img:         Observation image tensor  [B, 3, H, W].
            goal_img:        Goal image tensor         [B, 3, H, W].
            input_goal_mask: Optional goal mask        [B], long tensor (0=no mask, 1=mask goal).

        Returns:
            goal_encoding_tokens: Fused distance embedding [B, obs_encoding_size].
        """
        device = obs_img.device

        # ── Forward encoding:  [obs | goal] ────────────────────────────────────
        obsgoal_img = torch.cat(
            [obs_img.repeat(len(goal_img), 1, 1, 1), goal_img], dim=1
        )  # [B, 6, H, W]
        obsgoal_encoding = self._encode_goal(obsgoal_img)   # [B, goal_encoding_size]

        # ── Reverse encoding: [goal | obs] ─────────────────────────────────────
        goalobs_img = torch.cat(
            [goal_img, obs_img.repeat(len(goal_img), 1, 1, 1)], dim=1
        )  # [B, 6, H, W]
        goalobs_encoding = self._encode_goal(goalobs_img)   # [B, goal_encoding_size]

        # ── Compress and reshape to sequence tokens ─────────────────────────────
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)   # [B, D]
        goalobs_encoding = self.compress_goal_enc(goalobs_encoding)   # [B, D]

        obsgoal_encoding = obsgoal_encoding.unsqueeze(1)  # [B, 1, D]
        goalobs_encoding = goalobs_encoding.unsqueeze(1)  # [B, 1, D]

        assert obsgoal_encoding.shape[2] == self.goal_encoding_size

        # ── Concatenate to form a 2-token sequence ─────────────────────────────
        goal_encoding = torch.cat([obsgoal_encoding, goalobs_encoding], dim=1)  # [B, 2, D]

        # ── Optional goal masking ───────────────────────────────────────────────
        src_key_padding_mask = None
        if input_goal_mask is not None:
            no_goal_mask = input_goal_mask.long().to(device)
            src_key_padding_mask = torch.index_select(
                self.all_masks.to(device), 0, no_goal_mask
            )

        # ── Positional encoding + self-attention ───────────────────────────────
        goal_encoding = self.positional_encoding(goal_encoding)
        goal_encoding_tokens = self.sa_encoder(goal_encoding, src_key_padding_mask=None)

        # ── Mean pooling over the 2 tokens → [B, D] ────────────────────────────
        goal_encoding_tokens = torch.mean(goal_encoding_tokens, dim=1)

        return goal_encoding_tokens

    # ── Private helpers ────────────────────────────────────────────────────────

    def _encode_goal(self, img: torch.Tensor) -> torch.Tensor:
        """Run the 6-channel goal encoder and return a flat feature vector."""
        x = self.goal_encoder.extract_features(img)
        x = self.goal_encoder._avg_pooling(x)
        if self.goal_encoder._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self.goal_encoder._dropout(x)
        return x


# ── GroupNorm utilities ────────────────────────────────────────────────────────

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int = 16,
) -> nn.Module:
    """Replace all BatchNorm2d layers with GroupNorm."""
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
    """
    Recursively replace all submodules matching `predicate` with `func(module)`.

    Args:
        root_module: The root nn.Module to traverse.
        predicate:   Returns True for modules to be replaced.
        func:        Returns the replacement module.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)

    # Verify all targets were replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0, "Some modules were not replaced."
    return root_module
