"""Original current-LiDAR sensor-origin readout, including yaw-dependent lever arm."""
import torch
EVALUATION_INDICES=(4,9,14,19,24,29)

def sample_rollout_2hz(trajectory):
    if trajectory.ndim != 3 or trajectory.shape[1:] != (30, 4):
        raise ValueError('Expected 30 integrated .1s states, not six reshaped points')
    return trajectory[:, EVALUATION_INDICES, :2]

def to_lidar(points, transform):
    """Explicit validated homogeneous control-frame->LiDAR transform."""
    if transform.shape != (points.shape[0], 4, 4):
        raise ValueError('Explicit [B,4,4] transform required')
    expected = transform.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(points.shape[0], -1)
    if not torch.allclose(transform[:, 3], expected):
        raise ValueError('Invalid rigid transform')
    rotation = transform[:, :3, :3]
    identity = torch.eye(3, device=points.device, dtype=points.dtype).expand(points.shape[0], -1, -1)
    if not torch.allclose(rotation @ rotation.transpose(1, 2), identity, atol=1e-05, rtol=1e-05):
        raise ValueError('Nonrigid frame conversion')
    if not torch.allclose(torch.linalg.det(rotation), points.new_ones(points.shape[0]), atol=1e-05, rtol=1e-05):
        raise ValueError('Reflection is not a coordinate rotation')
    xyz = torch.cat([points, torch.zeros_like(points[..., :1])], -1)
    return (xyz @ rotation.transpose(1, 2) + transform[:, None, :3, 3])[..., :2]

def _sensor_readout_validated_inputs(trajectory, transform, sensor_offset):
    sample_rollout_2hz(trajectory)
    if sensor_offset.shape != (trajectory.shape[0], 3):
        raise ValueError('Explicit current sensor offset [B,3] required')
    to_lidar(trajectory[:, EVALUATION_INDICES, :2], transform)
    if trajectory.dtype not in (torch.float32, torch.float64):
        raise ValueError('Diagnostic domain is FP32 or FP64 only')
    if transform.dtype != trajectory.dtype or sensor_offset.dtype != trajectory.dtype or transform.device != trajectory.device or (sensor_offset.device != trajectory.device):
        raise ValueError('Same input dtype and device required; no silent mixed-domain admission')
    return trajectory[:, EVALUATION_INDICES]

def sample_lidar_sensor_origin(trajectory, transform, sensor_offset):
    """Same real readout using a stable small-yaw increment and actual calibration remainder."""
    sampled = _sensor_readout_validated_inputs(trajectory, transform, sensor_offset).double()
    yaw = sampled[..., 2]
    s = yaw.sin()
    cm1 = -2 * (yaw / 2).sin().square()
    offset = sensor_offset.double()[:, None].expand(-1, 6, -1)
    dlever = torch.stack((cm1 * offset[..., 0] - s * offset[..., 1], s * offset[..., 0] + cm1 * offset[..., 1], torch.zeros_like(yaw)), -1)
    p = torch.cat((sampled[..., :2], torch.zeros_like(sampled[..., :1])), -1)
    R = transform.double()[:, :3, :3]
    t = transform.double()[:, :3, 3]
    remainder = (R @ sensor_offset.double()[..., None]).squeeze(-1) + t
    value = p @ R.transpose(1, 2) + dlever @ R.transpose(1, 2) + remainder[:, None]
    return value[..., :2].to(trajectory.dtype)
