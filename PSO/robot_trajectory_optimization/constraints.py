# constraints.py
import numpy as np
from config import JOINT_NUM, JOINT_LIMITS, VELOCITY_LIMITS, TRAJECTORY_CONFIG

class ConstraintChecker:
    def __init__(self):
        self.joint_limits = JOINT_LIMITS
        self.velocity_limits = VELOCITY_LIMITS
        self.sample_points = TRAJECTORY_CONFIG["sample_points"]

    def check_obstacle_constraints(self, positions):
        """检查所有采样点是否满足Z < 1000，返回惩罚值"""
        penalty = 0
        violation_count = 0
        z_threshold = 1000  # 避障阈值

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
        max_time = TRAJECTORY_CONFIG["max_time_initial"] - TRAJECTORY_CONFIG["time_decay_rate"] * (iteration/max_iter)
        time_part = particles[:3].copy()
        time_part = np.maximum(time_part, min_time)
        time_part = np.minimum(time_part, max_time)
        particles[:3] = time_part
        return particles