"""
MLWAC — Multi-Level Waypoint-Angular Coordinated Navigation Model
=================================================================
Core model components for visual navigation in unstructured environments.

Modules
-------
SequenceEncoder
    Hybrid Transformer + GRU encoder for sequential action / waypoint data.
MLWAC
    Main navigation model combining a vision encoder, distance network,
    waypoint decoder, and action decoder.
DenseNetwork
    Lightweight MLP head for scalar regression (e.g. distance prediction).
"""

import os
import argparse
import time
import pdb
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from vint_train.models.vint.self_attention import MultiLayerDecoder


# ---------------------------------------------------------------------------
# Sequence Encoder
# ---------------------------------------------------------------------------

class SequenceEncoder(nn.Module):
    """Encode a variable-length action / waypoint sequence into a fixed-size vector.

    The encoder first applies a Transformer for global context, then feeds the
    result through a GRU for temporal summarisation, and finally projects the
    last hidden state to the desired output dimension.

    Args:
        action_input_size: Feature dimension of each input token (``d_model``).
        hidden_size:       Hidden state size of the GRU.
        num_heads:         Number of attention heads in the Transformer layer.
        num_layers:        Number of layers shared by both the Transformer and GRU.
        dropout:           Dropout rate applied inside the Transformer layer.
        seq_len:           Expected sequence length (informational; not enforced).
        output_size:       Dimension of the final projected output vector.
    """

    def __init__(
        self,
        action_input_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        seq_len: int = 8,
        output_size: int = 64,
    ) -> None:
        super().__init__()

        # Transformer encoder for global context
        _tf_layer = TransformerEncoderLayer(
            d_model=action_input_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(
            encoder_layer=_tf_layer, num_layers=num_layers
        )

        # GRU for temporal feature extraction
        self.gru = nn.GRU(
            input_size=action_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Project GRU hidden state to output_size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, action_sequence: torch.Tensor) -> torch.Tensor:
        """Encode an action sequence.

        Args:
            action_sequence: Input tensor of shape ``(B, T, action_input_size)``.

        Returns:
            Encoded vector of shape ``(B, output_size)``.
        """
        x = self.transformer(action_sequence)   # (B, T, action_input_size)
        gru_out, _ = self.gru(x)                # (B, T, hidden_size)
        x = gru_out[:, -1, :]                   # (B, hidden_size) — last time step
        x = self.fc(x)                          # (B, output_size)
        return x


# ---------------------------------------------------------------------------
# MLWAC — Main Navigation Model
# ---------------------------------------------------------------------------

class MLWAC(nn.Module):
    """

    The model is designed to be called through a *dispatch* interface via
    ``forward(func_name, **kwargs)``, allowing different sub-networks to be
    invoked independently during training and inference.

    Sub-networks
    ------------
    vision_encoder
        Encodes (observation, goal) image pairs into a context token.
    distance_net
        Predicts the distance between an observation and a goal image.
    temporal_waypoints_encoder
        Encodes a sequence of historical waypoints.
    waypoint_pred_net
        Predicts future waypoints from the fused feature token.
    action_pred_net
        Predicts angular actions from the fused feature token.

    Args:
        vision_encoder: Pre-built vision encoder module (e.g. ViNT encoder).
        distance_net:   Pre-built distance estimation module.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        distance_net: nn.Module,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Sub-network references
        # ------------------------------------------------------------------ #
        self.vision_encoder = vision_encoder
        self.distance_net   = distance_net

        # ------------------------------------------------------------------ #
        # Hyper-parameters
        # ------------------------------------------------------------------ #
        self.learn_angle         = True
        self.context_size        = 5
        self.obs_encoding_size   = 256
        self.len_trajectory_pred = 4
        self.num_action_params   = 2

        # ------------------------------------------------------------------ #
        # Waypoint decoder + predictor
        # ------------------------------------------------------------------ #
        self.wp_decoder = MultiLayerDecoder(
            embed_dim=256,
            seq_len=self.context_size + 2,
            output_layers=[256, 128, 64],
            nhead=4,
            num_layers=2,
            ff_dim_factor=4,
        )
        self.waypoint_predictor = nn.Sequential(
            nn.Linear(64, self.len_trajectory_pred * self.num_action_params),
        )

        # ------------------------------------------------------------------ #
        # Action (angular) decoder + predictor
        # ------------------------------------------------------------------ #
        self.ac_decoder = MultiLayerDecoder(
            embed_dim=256,
            seq_len=self.context_size + 3,
            output_layers=[256, 128, 64],
            nhead=4,
            num_layers=2,
            ff_dim_factor=4,
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(64, self.len_trajectory_pred * 1),
        )

        # ------------------------------------------------------------------ #
        # Sequence encoders
        # ------------------------------------------------------------------ #
        self.wp_encoder = SequenceEncoder(
            action_input_size=2,
            hidden_size=64,
            num_heads=2,
            num_layers=4,
            dropout=0.1,
            seq_len=4,
            output_size=256,
        )
        self.temporal_waypoints_encoder = SequenceEncoder(
            action_input_size=2,
            hidden_size=64,
            num_heads=2,
            num_layers=4,
            dropout=0.1,
            seq_len=5,
            output_size=256,
        )

        # ------------------------------------------------------------------ #
        # Probabilistic feature fusion (2-channel conv)
        # ------------------------------------------------------------------ #
        self.pro_fuse = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1),
        )

    # ---------------------------------------------------------------------- #
    # Forward dispatch
    # ---------------------------------------------------------------------- #

    def forward(self, func_name: str, **kwargs) -> torch.Tensor:
        """Dispatch a forward call to the named sub-network.

        Args:
            func_name: One of ``"vision_encoder"``, ``"distance_net"``,
                       ``"temporal_waypoints_encoder"``, ``"waypoint_pred_net"``,
                       ``"action_pred_net"``.
            **kwargs:  Sub-network-specific keyword arguments (see below).

        Keyword Args (by func_name)
        ---------------------------
        ``vision_encoder``:
            obs_img, goal_img, input_goal_mask
        ``distance_net``:
            obs_img, goal_img
        ``temporal_waypoints_encoder``:
            temporal_waypoints  — shape ``(B, T, 2)``
        ``waypoint_pred_net``:
            feature             — shape ``(B, seq_len, 256)``
        ``action_pred_net``:
            feature             — shape ``(B, seq_len, 256)``

        Returns:
            Sub-network output tensor(s).

        Raises:
            NotImplementedError: If *func_name* is not recognised.
        """
        if func_name == "vision_encoder":
            return self.vision_encoder(
                kwargs["obs_img"],
                kwargs["goal_img"],
                input_goal_mask=kwargs["input_goal_mask"],
            )

        elif func_name == "distance_net":
            return self.distance_net(
                obs_img=kwargs["obs_img"],
                goal_img=kwargs["goal_img"],
            )

        elif func_name == "temporal_waypoints_encoder":
            return self.temporal_waypoints_encoder(kwargs["temporal_waypoints"])

        elif func_name == "waypoint_pred_net":
            context = self.wp_decoder(kwargs["feature"])        # (B, 64)
            output  = self.waypoint_predictor(context)          # (B, T*2)
            output  = output.reshape(
                output.shape[0], self.len_trajectory_pred, self.num_action_params
            )                                                   # (B, T, 2)

            # Convert position deltas → cumulative waypoints
            output[:, :, :2] = torch.cumsum(output[:, :, :2], dim=1)

            # Normalise angle predictions (last channel onwards)
            if self.learn_angle:
                output[:, :, 2:] = F.normalize(
                    output[:, :, 2:].clone(), dim=-1
                )

            wp_enc = self.wp_encoder(output.detach())           # (B, 256)
            return output, wp_enc.view(-1, 1, 256)

        elif func_name == "action_pred_net":
            context = self.ac_decoder(kwargs["feature"])        # (B, 64)
            output  = self.action_predictor(context)            # (B, T)
            output  = output.reshape(
                output.shape[0], self.len_trajectory_pred, 1
            )                                                   # (B, T, 1)
            return output

        else:
            raise NotImplementedError(
                f"Unknown func_name: '{func_name}'. "
                "Choose from: 'vision_encoder', 'distance_net', "
                "'temporal_waypoints_encoder', 'waypoint_pred_net', 'action_pred_net'."
            )


# ---------------------------------------------------------------------------
# Dense Network (scalar regression head)
# ---------------------------------------------------------------------------

class DenseNetwork(nn.Module):
    """Lightweight MLP for scalar regression (e.g. distance estimation).

    Args:
        embedding_dim: Dimension of the input feature vector.
                       The network projects: D → D/4 → D/16 → 1.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim,       embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4,  embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(embedding_dim // 16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP on a batch of embeddings.

        Args:
            x: Input tensor of shape ``(B, embedding_dim)`` or any shape
               whose total per-sample elements equal ``embedding_dim``.

        Returns:
            Scalar predictions of shape ``(B, 1)``.
        """
        x = x.reshape(-1, self.embedding_dim)
        return self.network(x)