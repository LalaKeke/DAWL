"""Core architecture adapted from the DAWL / SparseDrive-PNN implementation.
See the repository source attribution notice. No data or pretrained weights are bundled.
"""
from typing import Union
import torch

class BicycleModel(torch.nn.Module):

    def __init__(self, wheelbase: float, dt: float=0.1):
        super().__init__()
        self.state_var_list=['x_pos','y_pos','theta','v']
        from .vehicle import validate_wheelbase
        validate_wheelbase(wheelbase)
        if torch.is_tensor(wheelbase):
            wb = wheelbase.detach().clone().float()
        else:
            wb = torch.tensor(float(wheelbase), dtype=torch.float32)
        self.register_buffer('wheelbase', wb)
        self.dt = dt

    def forward(self, x: torch.Tensor, u: torch.Tensor, t: Union[torch.Tensor, float]=None, d: torch.Tensor=None) -> torch.Tensor:
        wheelbase = self.wheelbase.to(device=x.device, dtype=x.dtype)
        theta = x[:, 2]
        v = x[:, 3]
        a = u[:, 0]
        delta = u[:, 1]
        theta_normalized = torch.atan2(torch.sin(theta), torch.cos(theta))
        v_safe = torch.clamp(v, min=1e-08)
        dx_pos = v_safe * torch.cos(theta_normalized)
        dy_pos = v_safe * torch.sin(theta_normalized)
        dtheta = v_safe / wheelbase * torch.tan(delta)
        dv = a
        dx = torch.stack((dx_pos, dy_pos, dtheta, dv), dim=1)
        return dx
DT = 0.1
TRAJ_LEN = 30

def rollout_all_agents(dynamics_model, ego_state, ped_states, veh_states, u_ego, u_peds, u_vehs, N_step=TRAJ_LEN, dt=DT):
    B = ego_state.size(0)
    Np = ped_states.size(1)
    Nv = veh_states.size(1)
    ego = ego_state[:, :4]
    ped = ped_states[:, :, :4]
    veh = veh_states[:, :, :4]
    ego_list = []
    ped_list = []
    veh_list = []
    for t in range(N_step):
        dx_ego = dynamics_model(ego, u_ego[:, t])
        ego = ego + dt * dx_ego
        dx_ped = dynamics_model(ped.reshape(B * Np, 4), u_peds[:, :, t].reshape(B * Np, 2))
        ped = ped + dt * dx_ped.view(B, Np, 4)
        dx_veh = dynamics_model(veh.reshape(B * Nv, 4), u_vehs[:, :, t].reshape(B * Nv, 2))
        veh = veh + dt * dx_veh.view(B, Nv, 4)
        ego = torch.stack([ego[:, 0], ego[:, 1], torch.atan2(torch.sin(ego[:, 2]), torch.cos(ego[:, 2])), torch.clamp(ego[:, 3], min=0.0)], dim=1)
        ped = torch.stack([ped[:, :, 0], ped[:, :, 1], torch.atan2(torch.sin(ped[:, :, 2]), torch.cos(ped[:, :, 2])), torch.clamp(ped[:, :, 3], min=0.0)], dim=2)
        veh = torch.stack([veh[:, :, 0], veh[:, :, 1], torch.atan2(torch.sin(veh[:, :, 2]), torch.cos(veh[:, :, 2])), torch.clamp(veh[:, :, 3], min=0.0)], dim=2)
        ego_list.append(ego)
        ped_list.append(ped)
        veh_list.append(veh)
    ego_traj = torch.stack(ego_list, dim=1)
    ped_traj = torch.stack(ped_list, dim=2)
    veh_traj = torch.stack(veh_list, dim=2)
    return (ego_traj, ped_traj, veh_traj)
