"""Portable ControlNet inference with the current absolute-angle physical map.

Inputs must already follow the documented structured feature and physical raw
schemas. This module does not fit normalization, create navigation, load data,
run Theseus, or infer a missing previous steering angle.
"""
import torch
from torch import nn
from ..models.control_net import StaticAwareControlNet
from .vehicle import WHEELBASE_M
from .rollout import BicycleModel, rollout_all_agents
from .absolute_angle import map_absolute_angle_controls
from .readout import sample_lidar_sensor_origin


def decode_controls(outputs):
    """Decode three normalized Tanh control heads using the original bounds."""
    if len(outputs) != 3:
        raise ValueError("Expected ego, pedestrian and vehicle control outputs")
    bounds=((-10.,10.,1.066),(-1.,1.,torch.pi/4),(-10.,10.,1.066))
    return tuple(torch.stack((lo+(value[...,0]+1)*(hi-lo)/2,value[...,1]*angle),-1)
                 for value,(lo,hi,angle) in zip(outputs,bounds))


class ControlPlanner(nn.Module):
    """Control-only 30 x .1s planner; no WeightNet or training optimizer.

    ``raw`` uses physical ego_state/ped_states/veh_states and True-valid masks.
    ``features`` uses separately normalized ego/ped/veh/lane/static tensors.
    The caller must preserve raw physical initial states and real calibration.
    """
    def __init__(self, control, *, wheelbase=WHEELBASE_M, dt=.1):
        super().__init__()
        if float(wheelbase)!=WHEELBASE_M or float(dt)!=.1:
            raise ValueError("This physical profile is wheelbase2.588 / dt0.1")
        if getattr(control,"future_steps",None)!=30:
            raise ValueError("A 30-step ControlNet is required")
        self.control=control
        self.dynamics=BicycleModel(wheelbase,dt)

    def forward(self, raw, features, *, control_to_lidar=None, sensor_offset=None):
        if features["ego"].ndim!=2 or features["ego"].shape[-1]!=10:
            raise ValueError("features.ego must have shape [B,10]")
        if features["lane"].ndim!=4 or features["lane"].shape[1]!=2 or features["lane"].shape[-1]!=2:
            raise ValueError("features.lane must be an ordered [B,2,L,2] boundary pair")
        if "navigation_mask"in raw and not bool(raw["navigation_mask"].all()):
            raise ValueError("Incomplete navigation is outside this encoder schema")
        for name in ("ped_mask","veh_mask"):
            if raw[name].dtype!=torch.bool:
                raise TypeError("raw actor masks must be bool, True=valid")
        args=(features["ego"],features["ped"],features["veh"],features["lane"],
              ~raw["ped_mask"],~raw["veh_mask"])
        kwargs={"map_mask":raw["map_mask"]} if "map_mask"in raw else {}
        if isinstance(self.control,StaticAwareControlNet):
            outputs=self.control(*args,features.get("static"),raw.get("static_mask"),**kwargs)
        else:
            outputs=self.control(*args,**kwargs)
        decoded=decode_controls(outputs)
        if decoded[0].dtype not in (torch.float32,torch.float64):
            raise TypeError("The portable physical profile requires FP32 or FP64")
        previous=raw.get("previous_steering")
        if previous is None or previous.shape!=(decoded[0].shape[0],):
            raise ValueError("An explicit previous front-wheel angle [B] is required")
        if raw.get("previous_steering_source")not in ("estimated","assumed"):
            raise ValueError("Declare the sourced previous angle as estimated or assumed")
        ego,physical=map_absolute_angle_controls(
            decoded[0],raw["ego_state"][:,3].to(decoded[0]),previous.to(decoded[0]),
            ay_max=3.,rate_max=.4,dt=.1,epsilon_v2=.25,strategy="anticipate",
            viability_speed_margin=.0001,recovery_rate=.8)
        controls=(ego,decoded[1],decoded[2])
        trajectories=rollout_all_agents(self.dynamics,raw["ego_state"],raw["ped_states"],raw["veh_states"],*controls)
        result=dict(normalized_controls=outputs,raw_controls=decoded,controls=controls,
                    trajectories=trajectories,physical=physical)
        if (control_to_lidar is None)!=(sensor_offset is None):
            raise ValueError("Supply both control_to_lidar and sensor_offset")
        if control_to_lidar is not None:
            result["final_planning"]=sample_lidar_sensor_origin(trajectories[0],control_to_lidar,sensor_offset)
        return result


def predict_trajectory(control, scene):
    """Convenience interface for a prepared scene; returns cumulative LiDAR XY.

    Expected keys: raw, features, control_to_lidar, sensor_offset. Calling code
    owns train/eval mode and grad/no_grad context; these are not silently changed.
    """
    planner=ControlPlanner(control).to(scene["features"]["ego"].device)
    return planner(scene["raw"],scene["features"],
                   control_to_lidar=scene["control_to_lidar"],
                   sensor_offset=scene["sensor_offset"])["final_planning"]
