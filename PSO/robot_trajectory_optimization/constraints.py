# constraints.py
import numpy as np
from config import JOINT_NUM, JOINT_LIMITS, VELOCITY_LIMITS, TRAJECTORY_CONFIG, USE_GPU

try:
    import torch
except ImportError:
    torch = None

class ConstraintChecker:
    def __init__(self):
        self.joint_limits = JOINT_LIMITS
        self.velocity_limits = VELOCITY_LIMITS
        self.sample_points = TRAJECTORY_CONFIG["sample_points"]
        self.use_gpu = USE_GPU and torch is not None

        # 预处理约束为张量（如果使用GPU）
        if self.use_gpu:
            self.joint_limits_tensor = torch.tensor(self.joint_limits, dtype=torch.float32, device='cuda')
            self.velocity_limits_tensor = torch.tensor(self.velocity_limits, dtype=torch.float32, device='cuda')
        else:
            self.joint_limits_tensor = None
            self.velocity_limits_tensor = None

    def check_obstacle_constraints(self, positions):
        """检查所有采样点是否满足Z < 1000，返回惩罚值"""
        penalty = 0
        violation_count = 0
        z_threshold = 1000  # 避障阈值

        if self.use_gpu and isinstance(positions, torch.Tensor):
            # GPU计算
            z_values = positions[:, 2]
            violations = z_values >= z_threshold
            violation_count = violations.sum().item()
            if violation_count > 0:
                over_distances = z_values[violations] - z_threshold
                penalty = (1e4 * (over_distances + 1)).sum().item()
        else:
            # CPU计算
            for pos in positions:
                x, y, z = pos
                if z >= z_threshold:
                    over_distance = z - z_threshold
                    penalty += 1e4 * (over_distance + 1)
                    violation_count += 1
        return penalty, violation_count

    def check_joint_constraints(self, angles):
        """检查关节角度约束，返回惩罚值"""
        penalty = 0
        violation_count = 0

        if self.use_gpu and isinstance(angles, torch.Tensor):
            # GPU计算
            for j in range(JOINT_NUM):
                j_min, j_max = self.joint_limits_tensor[j]
                joint_angles = angles[:, j]
                violations = ~((j_min - 1e-6 <= joint_angles) & (joint_angles <= j_max + 1e-6))
                violation_count += violations.sum().item()
            penalty = violation_count * 1e3
        else:
            # CPU计算
            for angles_t in angles:
                for j in range(JOINT_NUM):
                    j_min, j_max = self.joint_limits[j]
                    if not (j_min - 1e-6 <= angles_t[j] <= j_max + 1e-6):  # 浮点容错
                        penalty += 1e3
                        violation_count += 1
        return penalty, violation_count

    def check_velocity_constraints(self, angles, t_samples):
        """检查关节速度约束，返回惩罚值"""
        penalty = 0
        violation_count = 0
        dt = t_samples[1] - t_samples[0] if len(t_samples) > 1 else 1e-6

        if self.use_gpu and isinstance(angles, torch.Tensor):
            # GPU计算速度
            velocity = torch.diff(angles, dim=0) / dt
            for j in range(JOINT_NUM):
                v_min, v_max = self.velocity_limits_tensor[j]
                joint_vel = velocity[:, j]
                violations = ~((v_min - 1e-6 <= joint_vel) & (joint_vel <= v_max + 1e-6))
                violation_count += violations.sum().item()
            penalty = violation_count * 1e3
        else:
            # CPU计算
            velocity = np.diff(angles, axis=0) / dt
            for vel_t in velocity:
                for j in range(JOINT_NUM):
                    v_min, v_max = self.velocity_limits[j]
                    if not (v_min - 1e-6 <= vel_t[j] <= v_max + 1e-6):
                        penalty += 1e3
                        violation_count += 1
        return penalty, violation_count

    def enforce_time_constraints(self, particles, iteration, max_iter):
        """强制时间约束（每段时间范围）"""
        min_time = TRAJECTORY_CONFIG["min_time_segment"]
        max_time = TRAJECTORY_CONFIG["max_time_initial"] - TRAJECTORY_CONFIG["time_decay_rate"] * (iteration / max_iter)
        time_part = particles[:3].copy()
        time_part = np.maximum(time_part, min_time)
        time_part = np.minimum(time_part, max_time)
        particles[:3] = time_part
        return particles