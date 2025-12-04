# trajectory_generator.py
import numpy as np
from spatialmath import SE3
from roboticstoolbox import ERobot
from config import JOINT_NUM, TRAJECTORY_CONFIG, URDF_PATH, BASE_ROTATION, BASE_TRANSLATION, USE_GPU

try:
    import torch
except ImportError:
    torch = None


class TrajectoryGenerator:
    def __init__(self, start_point, end_point):
        self.start_point = np.radians(start_point)
        self.end_point = np.radians(end_point)
        self.num_segments = TRAJECTORY_CONFIG["num_segments"]
        self.prev_coeffs = []  # 保存每段多项式系数
        self._load_robot_urdf()
        self.use_gpu = USE_GPU and torch is not None

        # 转换基座变换为张量（如果使用GPU）
        if self.use_gpu:
            self.r_base_tensor = torch.tensor(self.r_base, dtype=torch.float32, device='cuda')
            self.base_translation_tensor = torch.tensor(BASE_TRANSLATION, dtype=torch.float32, device='cuda')
        else:
            self.r_base_tensor = None
            self.base_translation_tensor = None

    def _load_robot_urdf(self):
        """加载URDF模型，初始化正运动学链"""
        self.assemble_robot = ERobot.URDF(URDF_PATH)
        # 基座旋转矩阵
        self.r_base = SE3.Rz(BASE_ROTATION).R

    def polynomial_trajectory(self, particles, t):
        """生成指定时间t的关节角度（五阶多项式）"""
        # 计算各段时间累计
        cum_time = np.cumsum(np.r_[0, particles[:self.num_segments]])
        segment = np.searchsorted(cum_time, t) - 1
        segment = max(0, min(segment, self.num_segments - 1))  # 防止越界

        # 重新计算当前段系数（避免prev_coeffs累积错误）
        self.prev_coeffs = []
        for seg in range(self.num_segments):
            t_start = cum_time[seg]
            t_end = cum_time[seg + 1]
            T = t_end - t_start

            # 确定当前段起止关节角度
            if seg == 0:
                q_start = self.start_point
                q_end = np.radians(particles[self.num_segments:self.num_segments + 6])
            elif seg == self.num_segments - 1:
                q_start = np.radians(particles[self.num_segments:self.num_segments + 6])
                q_end = self.end_point
            else:
                q_start = np.radians(particles[self.num_segments:self.num_segments + 6])
                q_end = np.radians(particles[-6:])

            self.prev_coeffs.append(self.calculate_segment_coeffs(q_start, q_end, T, seg))

        # 计算当前段的关节角度
        coeffes = self.prev_coeffs[segment]
        t_segment = t - cum_time[segment]
        T_segment = cum_time[segment + 1] - cum_time[segment]
        tau = t_segment / T_segment if T_segment > 1e-6 else 0

        angles = np.zeros(JOINT_NUM)
        for j in range(JOINT_NUM):
            coeff = coeffes[j]
            angles[j] = (
                    coeff[0] * tau ** 5 + coeff[1] * tau ** 4 + coeff[2] * tau ** 3 +
                    coeff[3] * tau ** 2 + coeff[4] * tau + coeff[5]
            )
        return angles

    def calculate_segment_coeffs(self, q_start, q_end, T, segment):
        """计算单段五阶多项式系数"""
        coeffs = []
        for j in range(JOINT_NUM):
            # 五阶多项式求解矩阵
            A = np.array([
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 2, 0, 0],
                [T ** 5, T ** 4, T ** 3, T ** 2, T, 1],
                [5 * T ** 4, 4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0],
                [20 * T ** 3, 12 * T ** 2, 6 * T, 2, 0, 0]
            ])

            # 确定边界条件
            if segment == 0:
                next_velocity = (q_end[j] - q_start[j]) / T if T > 1e-6 else 0
                next_accel = next_velocity / T if T > 1e-6 else 0
                b = [q_start[j], 0, 0, q_end[j], next_velocity, next_accel]
            elif segment == self.num_segments - 1:
                prev_vel = self._calc_prev_velocity(j, T) if self.prev_coeffs else 0
                prev_accel = self._calc_prev_acceleration(j, T) if self.prev_coeffs else 0
                b = [q_start[j], prev_vel, prev_accel, q_end[j], 0, 0]
            else:
                prev_vel = self._calc_prev_velocity(j, T) if self.prev_coeffs else 0
                prev_accel = self._calc_prev_acceleration(j, T) if self.prev_coeffs else 0
                b = [q_start[j], prev_vel, prev_accel, q_end[j], prev_vel, prev_accel]

            # 求解系数
            coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            coeffs.append(coeff)
        return coeffs

    def _calc_prev_velocity(self, joint_idx, T):
        """计算前一段的末端速度"""
        if not self.prev_coeffs:
            return 0
        coeff = self.prev_coeffs[-1][joint_idx]
        return 5 * coeff[0] * T ** 4 + 4 * coeff[1] * T ** 3 + 3 * coeff[2] * T ** 2 + 2 * coeff[3] * T + coeff[4]

    def _calc_prev_acceleration(self, joint_idx, T):
        """计算前一段的末端加速度"""
        if not self.prev_coeffs:
            return 0
        coeff = self.prev_coeffs[-1][joint_idx]
        return 20 * coeff[0] * T ** 3 + 12 * coeff[1] * T ** 2 + 6 * coeff[2] * T + 2 * coeff[3]

    def forward_kinematics(self, q):
        """正运动学：输入关节角度，输出TCP位置"""
        # 如果是torch张量，转换为numpy进行机器人库计算（大多数机器人库不支持GPU）
        if self.use_gpu and isinstance(q, torch.Tensor):
            q_np = q.cpu().numpy()
        else:
            q_np = q

        j1, j2, j3, j4, j5, j6 = q_np
        j3 += j2  # 关节3补偿
        joint_angles_all = np.zeros(9)
        joint_angles_all[1:7] = [j1, j2, j3, j4, j5, j6]

        tcp_matrix = self.assemble_robot.fkine(joint_angles_all)
        tcp_matrix_np = tcp_matrix.A
        tcp_position = tcp_matrix_np[:3, 3]
        # 基座变换
        tcp_position_vc = self.r_base @ tcp_position + BASE_TRANSLATION

        # 如果使用GPU，转换回张量
        if self.use_gpu:
            return torch.tensor(tcp_position_vc, dtype=torch.float32, device='cuda')
        return tcp_position_vc