# config.py
import numpy as np

# 关节参数配置
JOINT_NUM = 6
JOINT_LIMITS = [
    (np.radians(-340), np.radians(360)),  # J1
    (np.radians(-245), np.radians(245)),  # J2
    (np.radians(-430), np.radians(430)),  # J3
    (np.radians(-380), np.radians(380)),  # J4
    (np.radians(-250), np.radians(250)),  # J5
    (np.radians(-720), np.radians(720))   # J6
]
VELOCITY_LIMITS = [
    (np.radians(-370), np.radians(370)),  # J1
    (np.radians(-310), np.radians(310)),  # J2
    (np.radians(-410), np.radians(410)),  # J3
    (np.radians(-550), np.radians(550)),  # J4
    (np.radians(-545), np.radians(545)),  # J5
    (np.radians(-1000), np.radians(1000)) # J6
]

# PSO优化参数
PSO_CONFIG = {
    "n_particles": 50,
    "max_iter": 500,
    "w_max": 1.2,
    "w_min": 0.2,
    "c1_max": 2.0,
    "c1_min": 0.4,
    "c2_max": 2.0,
    "c2_min": 0.4,
    "stagnation_threshold": 10,
    "mutation_rate": 0.25,
    "elite_ratio": 0.1
}

# 轨迹生成参数
TRAJECTORY_CONFIG = {
    "num_segments": 3,
    "sample_points": 300,
    "min_time_segment": 0.0001,
    "max_time_initial": 0.5,
    "time_decay_rate": 0.2
}

# URDF文件路径（正运动学用）
URDF_PATH = r"D:\PSO_FANUC\PSO\robot_trajectory_optimization\lrmate200id7l.urdf"
# 基座变换参数
BASE_ROTATION = np.radians(180)  # z轴旋转180度
BASE_TRANSLATION = np.array([1.756, -0.00416, 1.0])