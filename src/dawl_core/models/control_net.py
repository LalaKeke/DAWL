"""Core architecture adapted from the DAWL / SparseDrive-PNN implementation.
See the repository source attribution notice. No data or pretrained weights are bundled.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def encode_masked_map(encoder, lane_points, map_mask=None):
    """Non-parametric Phase22 extension; True means observed map point.

    None retains the original path. Unknown coordinates are removed BEFORE
    the point MLP, its bias outputs are zeroed, and fusion must also mask keys.
    This changes operators on missing input, not layers or checkpoint keys.
    """
    b=lane_points.shape[0]
    points=lane_points.reshape(b,-1,2)
    if map_mask is None:return encoder(points),None
    if map_mask.dtype!=torch.bool or map_mask.shape!=lane_points.shape[:-1]:
        raise ValueError('map_mask must be bool, True=valid, with lane_points.shape[:-1]')
    valid=map_mask.to(device=points.device).reshape(b,-1)
    if not torch.isfinite(points[valid]).all():raise FloatingPointError('Nonfinite observed map point')
    # Exact original operators for all-valid inputs, including their gradients.
    if bool(valid.all()):return encoder(points),~valid
    safe=torch.where(valid[...,None],points,torch.zeros_like(points))
    features=encoder(safe)
    return torch.where(valid[...,None],features,torch.zeros_like(features)),~valid

class EluTimeControlEnhanced(nn.Module):

    def __init__(self, embed_dim=128, num_heads=4, future_steps=30):
        super().__init__()
        self.future_steps = future_steps
        self.embed_dim = embed_dim
        self.ego_encoder = nn.Sequential(nn.Linear(10, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, embed_dim))
        self.ped_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, embed_dim))
        self.veh_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, embed_dim))
        self.map_encoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-5, batch_first=True, norm_first=False)
        self.attn_fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.res_fc1 = nn.Linear(embed_dim, embed_dim)
        self.res_fc2 = nn.Linear(embed_dim, embed_dim)
        self.ego_output = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, future_steps * 2), nn.Tanh())
        self.ped_output = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, future_steps * 2), nn.Tanh())
        self.veh_output = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, future_steps * 2), nn.Tanh())

    def forward(self, ego_state, ped_states, veh_states, lane_points, ped_mask=None, veh_mask=None, map_mask=None):
        B = ego_state.size(0)
        Np = ped_states.size(1)
        Nv = veh_states.size(1)
        ego_feat = self.ego_encoder(ego_state).unsqueeze(1)
        ped_feat = self.ped_encoder(ped_states)
        veh_feat = self.veh_encoder(veh_states)
        map_feat, map_padding = encode_masked_map(self.map_encoder, lane_points, map_mask)
        all_feat = torch.cat([ego_feat, ped_feat, veh_feat, map_feat], dim=1)
        total_len = 1 + Np + Nv + map_feat.shape[1]
        device = ego_state.device
        key_padding_mask = torch.zeros(B, total_len, dtype=torch.bool, device=device)
        if ped_mask is not None:
            key_padding_mask[:, 1:1 + Np] = ped_mask
        if veh_mask is not None:
            key_padding_mask[:, 1 + Np:1 + Np + Nv] = veh_mask
        if map_padding is not None:key_padding_mask[:,1+Np+Nv:] = map_padding
        fused_feat = self.attn_fusion(all_feat, src_key_padding_mask=key_padding_mask)
        fused_feat = fused_feat + F.relu(self.res_fc2(F.relu(self.res_fc1(fused_feat))))
        ego_out = self.ego_output(fused_feat[:, 0])
        ped_out = self.ped_output(fused_feat[:, 1:1 + Np])
        veh_out = self.veh_output(fused_feat[:, 1 + Np:1 + Np + Nv])
        ego_out = ego_out.view(B, self.future_steps, 2)
        ped_out = ped_out.view(B, Np, self.future_steps, 2)
        veh_out = veh_out.view(B, Nv, self.future_steps, 2)
        return (ego_out, ped_out, veh_out)

class StaticAwareControlNet(EluTimeControlEnhanced):
    """ControlNet variant that consumes HiP-AD's existing static detections.

    ``static_states`` layout:
      [x, y, z, yaw, length, width, height, score,
       is_sign, is_cone, is_light, is_other]

    The static residual projection is zero-initialized. Loading an existing
    ControlNet checkpoint therefore preserves its output exactly at step zero,
    while training can learn a scene-dependent correction from static boxes.
    """
    STATIC_FEATURE_DIM = 12

    def __init__(self, embed_dim=128, num_heads=4, future_steps=30):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, future_steps=future_steps)
        self.static_encoder = nn.Sequential(nn.Linear(self.STATIC_FEATURE_DIM, 64), nn.ReLU(), nn.Linear(64, embed_dim), nn.LayerNorm(embed_dim))
        self.static_attention = nn.Sequential(nn.Linear(2 * embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.static_attention[-1].weight)
        nn.init.zeros_(self.static_attention[-1].bias)
        self.static_risk_gate = nn.Sequential(nn.Linear(2 * embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.static_risk_gate[-1].weight)
        nn.init.constant_(self.static_risk_gate[-1].bias, -2.9444389791664403)
        self.static_to_ego = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.static_to_ego.weight)
        nn.init.zeros_(self.static_to_ego.bias)
        self.last_static_gate = None

    @staticmethod
    def _static_attention_state_keys():
        return {'static_attention.0.weight', 'static_attention.0.bias', 'static_attention.2.weight', 'static_attention.2.bias'}

    @staticmethod
    def _static_risk_gate_state_keys():
        return {'static_risk_gate.0.weight', 'static_risk_gate.0.bias', 'static_risk_gate.2.weight', 'static_risk_gate.2.bias'}

    def load_legacy_control_state_dict(self, state_dict):
        """Load an old ControlNet while retaining new zero-init parameters."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_missing = {'static_encoder.0.weight', 'static_encoder.0.bias', 'static_encoder.2.weight', 'static_encoder.2.bias', 'static_encoder.3.weight', 'static_encoder.3.bias', 'static_to_ego.weight', 'static_to_ego.bias'}
        allowed_missing.update(self._static_attention_state_keys())
        allowed_missing.update(self._static_risk_gate_state_keys())
        extra_missing = set(missing).difference(allowed_missing)
        if extra_missing or unexpected:
            raise RuntimeError(f'Legacy ControlNet checkpoint mismatch: missing={sorted(extra_missing)}, unexpected={sorted(unexpected)}')
        return (missing, unexpected)

    def load_compatible_static_state_dict(self, state_dict):
        """Load a static-v1/v2/v3 checkpoint without changing old behavior."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_missing = set(self._static_attention_state_keys())
        allowed_missing.update(self._static_risk_gate_state_keys())
        extra_missing = set(missing).difference(allowed_missing)
        if extra_missing or unexpected:
            raise RuntimeError(f'Static ControlNet checkpoint mismatch: missing={sorted(extra_missing)}, unexpected={sorted(unexpected)}')
        if set(missing).intersection(self._static_risk_gate_state_keys()):
            nn.init.zeros_(self.static_risk_gate[-1].weight)
            nn.init.constant_(self.static_risk_gate[-1].bias, 20.0)
        return (missing, unexpected)

    def forward(self, ego_state, ped_states, veh_states, lane_points, ped_mask=None, veh_mask=None, static_states=None, static_mask=None, map_mask=None):
        B = ego_state.size(0)
        Np = ped_states.size(1)
        Nv = veh_states.size(1)
        ego_feat = self.ego_encoder(ego_state).unsqueeze(1)
        ped_feat = self.ped_encoder(ped_states)
        veh_feat = self.veh_encoder(veh_states)
        map_feat, map_padding = encode_masked_map(self.map_encoder, lane_points, map_mask)
        all_feat = torch.cat([ego_feat, ped_feat, veh_feat, map_feat], dim=1)
        total_len = all_feat.shape[1]
        key_padding_mask = torch.zeros(B, total_len, dtype=torch.bool, device=ego_state.device)
        if ped_mask is not None:
            key_padding_mask[:, 1:1 + Np] = ped_mask
        if veh_mask is not None:
            key_padding_mask[:, 1 + Np:1 + Np + Nv] = veh_mask
        if map_padding is not None:key_padding_mask[:,1+Np+Nv:] = map_padding
        fused_feat = self.attn_fusion(all_feat, src_key_padding_mask=key_padding_mask)
        fused_feat = fused_feat + F.relu(self.res_fc2(F.relu(self.res_fc1(fused_feat))))
        if static_states is not None and static_states.shape[1] > 0:
            static_feat = self.static_encoder(static_states)
            if static_mask is None:
                valid_bool = torch.ones(static_feat.shape[:2], device=static_feat.device, dtype=torch.bool)
            else:
                valid_bool = static_mask.bool()
            valid = valid_bool.to(dtype=static_feat.dtype)
            valid_count = valid.sum(dim=1, keepdim=True)
            ego_context = fused_feat[:, :1].expand(-1, static_feat.shape[1], -1)
            attention_logits = self.static_attention(torch.cat([static_feat, ego_context], dim=-1)).squeeze(-1)
            attention_logits = attention_logits.masked_fill(~valid_bool, -10000.0)
            attention = torch.softmax(attention_logits, dim=1) * valid
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-06)
            pooled = (static_feat * attention.unsqueeze(-1)).sum(dim=1)
            static_gate = torch.sigmoid(self.static_risk_gate(torch.cat([pooled, fused_feat[:, 0]], dim=-1)).squeeze(-1))
            static_gate = static_gate * (valid_count.squeeze(1) > 0).to(dtype=static_feat.dtype)
            self.last_static_gate = static_gate
            has_static = (valid_count > 0).to(dtype=static_feat.dtype)
            static_residual = self.static_to_ego(pooled) * has_static * static_gate.unsqueeze(-1)
            fused_feat[:, 0] = fused_feat[:, 0] + static_residual
        else:
            self.last_static_gate = ego_state.new_zeros(B)
        ego_out = self.ego_output(fused_feat[:, 0])
        ped_out = self.ped_output(fused_feat[:, 1:1 + Np])
        veh_out = self.veh_output(fused_feat[:, 1 + Np:1 + Np + Nv])
        return (ego_out.view(B, self.future_steps, 2), ped_out.view(B, Np, self.future_steps, 2), veh_out.view(B, Nv, self.future_steps, 2))
