"""Core architecture adapted from the DAWL / SparseDrive-PNN implementation.
See the repository source attribution notice. No data or pretrained weights are bundled.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .control_net import encode_masked_map

class SceneBackbone(nn.Module):

    def __init__(self, embed_dim: int=128, num_heads: int=4, num_tasks: int=8, temperature: float=1.0, num_transformer_layers: int=2, decoder_hidden_dim: int=128, dropout: float=0.1, use_prior_context: bool=False, prior_context_mode: str='log', initial_refine_gate: float=0.01) -> None:
        super().__init__()
        self.temperature = temperature
        self.num_tasks = num_tasks
        self.embed_dim = embed_dim
        self.use_prior_context = bool(use_prior_context)
        self.prior_context_mode = str(prior_context_mode).lower()
        if self.prior_context_mode not in {'log', 'raw', 'prob'}:
            raise ValueError('prior_context_mode must be one of: log, raw, prob')
        self.ego_encoder = nn.Sequential(nn.Linear(10, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, embed_dim))
        self.ped_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, embed_dim))
        self.veh_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, embed_dim))
        self.map_encoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2048, dropout=dropout, activation="relu", layer_norm_eps=1e-5, batch_first=True, norm_first=False)
        self.attn_fusion = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.res_fc1 = nn.Linear(embed_dim, embed_dim)
        self.res_fc2 = nn.Linear(embed_dim, embed_dim)
        decoder_input_dim = embed_dim + (num_tasks if self.use_prior_context else 0)
        self.weight_decoder = nn.Sequential(nn.Linear(decoder_input_dim, decoder_hidden_dim), nn.ReLU(), nn.Linear(decoder_hidden_dim, num_tasks))
        self._init_decoder()
        initial_refine_gate = min(max(float(initial_refine_gate), 0.0001), 1.0 - 0.0001)

    def _init_decoder(self) -> None:
        for m in self.weight_decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_prior_context(self, prior_weights: torch.Tensor=None, prior_log_weights: torch.Tensor=None, device: torch.device=None, dtype: torch.dtype=None) -> torch.Tensor:
        if prior_log_weights is None and prior_weights is None:
            raise ValueError('prior_weights or prior_log_weights is required when use_prior_context=True')
        if prior_log_weights is not None:
            prior_context = prior_log_weights
        elif self.prior_context_mode == 'log':
            prior_context = prior_weights.clamp_min(1e-08).log()
        elif self.prior_context_mode == 'prob':
            prior_context = prior_weights.clamp_min(1e-08)
            prior_context = prior_context / prior_context.sum(dim=-1, keepdim=True).clamp_min(1e-08)
        else:
            prior_context = prior_weights
        prior_context = prior_context.to(device=device, dtype=dtype)
        if prior_context.dim() != 2 or prior_context.shape[-1] != self.num_tasks:
            raise ValueError(f'prior context must have shape [B,{self.num_tasks}], got {tuple(prior_context.shape)}')
        return torch.nan_to_num(prior_context, nan=0.0, posinf=20.0, neginf=-20.0)

class WeightNet(SceneBackbone):
    """WeightNet with the exact StaticAwareControlNet ego-residual path."""
    STATIC_FEATURE_DIM = 12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.embed_dim
        self.static_encoder = nn.Sequential(nn.Linear(self.STATIC_FEATURE_DIM, 64), nn.ReLU(), nn.Linear(64, d), nn.LayerNorm(d))
        self.static_attention = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.static_attention[-1].weight)
        nn.init.zeros_(self.static_attention[-1].bias)
        self.static_risk_gate = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.static_risk_gate[-1].weight)
        nn.init.constant_(self.static_risk_gate[-1].bias, -2.9444389791664403)
        self.static_to_ego = nn.Linear(d, d)
        nn.init.zeros_(self.static_to_ego.weight)
        nn.init.zeros_(self.static_to_ego.bias)
        self.last_static_gate = None

    def forward(self, ego_state, ped_states, veh_states, lane_points, ped_mask=None, veh_mask=None, static_states=None, static_mask=None, prior_weights=None, prior_log_weights=None, return_logits=False, map_mask=None):
        b, nped, nveh = (ego_state.size(0), ped_states.size(1), veh_states.size(1))
        ego_feat = self.ego_encoder(ego_state).unsqueeze(1)
        ped_feat = self.ped_encoder(ped_states)
        veh_feat = self.veh_encoder(veh_states)
        map_feat, map_padding = encode_masked_map(self.map_encoder, lane_points, map_mask)
        fused = torch.cat([ego_feat, ped_feat, veh_feat, map_feat], dim=1)
        padding = torch.zeros(b, fused.shape[1], dtype=torch.bool, device=ego_state.device)
        if ped_mask is not None:
            padding[:, 1:1 + nped] = ~ped_mask.bool()
        if veh_mask is not None:
            padding[:, 1 + nped:1 + nped + nveh] = ~veh_mask.bool()
        if map_padding is not None:padding[:,1+nped+nveh:] = map_padding
        fused = self.attn_fusion(fused, src_key_padding_mask=padding)
        fused = fused + torch.relu(self.res_fc2(torch.relu(self.res_fc1(fused))))
        if static_states is not None and static_states.shape[1] > 0:
            static_feat = self.static_encoder(static_states)
            valid_bool = torch.ones(static_feat.shape[:2], dtype=torch.bool, device=static_feat.device) if static_mask is None else static_mask.bool()
            valid = valid_bool.to(static_feat.dtype)
            valid_count = valid.sum(dim=1, keepdim=True)
            ego_context = fused[:, :1].expand(-1, static_feat.shape[1], -1)
            logits = self.static_attention(torch.cat([static_feat, ego_context], dim=-1)).squeeze(-1).masked_fill(~valid_bool, -10000.0)
            attention = torch.softmax(logits, dim=1) * valid
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-06)
            pooled = (static_feat * attention.unsqueeze(-1)).sum(dim=1)
            gate = torch.sigmoid(self.static_risk_gate(torch.cat([pooled, fused[:, 0]], dim=-1)).squeeze(-1))
            gate = gate * (valid_count.squeeze(1) > 0).to(static_feat.dtype)
            self.last_static_gate = gate
            residual = self.static_to_ego(pooled) * (valid_count > 0).to(static_feat.dtype) * gate.unsqueeze(-1)
            fused[:, 0] = fused[:, 0] + residual
        else:
            self.last_static_gate = ego_state.new_zeros(b)
        scene_feat = fused[:, 0]
        if self.use_prior_context:
            context = self._build_prior_context(prior_weights=prior_weights, prior_log_weights=prior_log_weights, device=scene_feat.device, dtype=scene_feat.dtype)
            scene_feat = torch.cat([scene_feat, context], dim=-1)
        logits = self.weight_decoder(scene_feat)
        return logits
