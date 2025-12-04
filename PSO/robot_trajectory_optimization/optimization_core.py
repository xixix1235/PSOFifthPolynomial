# optimization_core.py
import numpy as np
from roboticstoolbox.examples.icra2021 import obstacle

from config import PSO_CONFIG, TRAJECTORY_CONFIG, JOINT_NUM
from trajectory_generator import TrajectoryGenerator
from constraints import ConstraintChecker

class PSOOptimizer:
    def __init__(self, csv_path, start_point, end_point):
        self.csv_path = csv_path
        self._load_csv_data()
        self.trajectory_gen = TrajectoryGenerator(start_point, end_point)
        self.constraint_checker = ConstraintChecker()
        self.num_segments = TRAJECTORY_CONFIG["num_segments"]
        # PSO参数
        self.n_particles = PSO_CONFIG["n_particles"]
        self.max_iter = PSO_CONFIG["max_iter"]
        #中间点角度+每段时间成本
        self.dim = (self.num_segments - 1) * 6 + self.num_segments

    def _load_csv_data(self):
        """加载CSV中的原始轨迹数据"""
        import pandas as pd
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.iloc[50:63]
        self.time = self.df['Time'].values
        self.original_joint_angles = np.radians(self.df[[f'j1{i}' for i in range(6)]].values)

    def initialize_particles(self):
        """初始化粒子群（时间分配+中间角度）"""
        particles = np.zeros((self.n_particles, self.dim))
        T_total = self.time[-1] - self.time[0]

        # 初始化基准多项式系数
        base_coeffs = []
        for j in range(JOINT_NUM):
            A = np.array([
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 2, 0, 0],
                [T_total**5, T_total**4, T_total**3, T_total**2, T_total, 1],
                [5*T_total**4, 4*T_total**3, 3*T_total**2, 2*T_total, 1, 0],
                [20*T_total**3, 12*T_total**2, 6*T_total, 2, 0, 0]
            ])
            b = [self.trajectory_gen.start_point[j], 0, 0, self.trajectory_gen.end_point[j], 0, 0]
            coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            base_coeffs.append(coeff)

        # 生成每个粒子
        for k in range(self.n_particles):
            # 时间分配初始化
            min_time = TRAJECTORY_CONFIG["min_time_segment"]
            max_total = 1.2
            proportions = np.random.uniform(0.1, 1.0, self.num_segments)
            proportions /= proportions.sum()
            remaining = max_total - self.num_segments * min_time
            time_interval = min_time + proportions * remaining if remaining > 0 else np.array([min_time]*self.num_segments)

            # 中间角度初始化
            angles = np.zeros((self.num_segments-1, JOINT_NUM))
            for i in range(self.num_segments-1):
                times = time_interval[i]
                tau = times / sum(time_interval) if sum(time_interval) > 1e-6 else 0
                for j in range(JOINT_NUM):
                    coeff = base_coeffs[j]
                    angles[i,j] = (
                        coeff[0]*tau**5 + coeff[1]*tau**4 + coeff[2]*tau**3 +
                        coeff[3]*tau**2 + coeff[4]*tau + coeff[5]
                    )
            # 加噪声
            noise = np.random.uniform(-5, 5, size=angles.shape)
            angles_with_noise = angles + noise
            # 拼接粒子（时间+角度）
            particles[k] = np.concatenate((time_interval, angles_with_noise.reshape(-1)))
        return particles

    def fitness_function(self, particle):
        """适应度函数：约束惩罚 + 时间惩罚"""
        # 生成采样轨迹
        t_total = sum(particle[:self.num_segments])
        t_samples = np.linspace(0, t_total, TRAJECTORY_CONFIG["sample_points"])
        angles = np.array([self.trajectory_gen.polynomial_trajectory(particle, t) for t in t_samples])
        positions=np.array([self.trajectory_gen.forward_kinematics(angle) for angle in angles]);

        # 约束惩罚
        joint_penalty, _ = self.constraint_checker.check_joint_constraints(angles)
        vel_penalty, _ = self.constraint_checker.check_velocity_constraints(angles, t_samples)
        obstacle_penalty,_=self.constraint_checker.check_obstacle_constraints(positions)
        total_constraint_penalty = joint_penalty + vel_penalty+obstacle_penalty

        # 时间惩罚（总时间越短越好）
        time_penalty = sum(particle[:self.num_segments])

        total_penalty = total_constraint_penalty + time_penalty
        return total_penalty

    def optimize(self):
        """执行PSO优化主逻辑"""
        # 初始化粒子和速度
        particles = self.initialize_particles()
        velocities = np.random.uniform(-15, 15, (self.n_particles, self.dim))

        # 初始化最优值
        best_positions = particles.copy()
        best_scores = np.full(self.n_particles, np.inf)
        global_best_position = None
        global_best_score = np.inf
        stagnation_count = 0

        for iteration in range(self.max_iter):
            # 动态调整PSO参数
            w = PSO_CONFIG["w_max"] - (PSO_CONFIG["w_max"] - PSO_CONFIG["w_min"]) * abs(np.sin(iteration * np.pi / (2 * self.max_iter)))
            c1 = PSO_CONFIG["c1_min"] + (PSO_CONFIG["c1_max"] - PSO_CONFIG["c1_min"]) * (0.5 + 0.5 * np.cos(iteration * np.pi / self.max_iter))
            c2 = PSO_CONFIG["c2_min"] + (PSO_CONFIG["c2_max"] - PSO_CONFIG["c2_min"]) * (0.5 - 0.5 * np.cos(iteration * np.pi / self.max_iter))

            prev_global_best = global_best_score
            scores = []

            # 计算所有粒子适应度，更新最优
            for i in range(self.n_particles):
                current_score = self.fitness_function(particles[i])
                scores.append(current_score)

                # 更新个体最优
                if current_score < best_scores[i]:
                    best_scores[i] = current_score
                    best_positions[i] = particles[i].copy()
                    # 更新全局最优
                    if current_score < global_best_score:
                        global_best_score = current_score
                        global_best_position = particles[i].copy()

            # 粒子更新（带模拟退火）
            temp = 100 * (1 - iteration / self.max_iter)
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                # 速度更新
                velocities[i] = w * velocities[i] + c1 * r1 * (best_positions[i] - particles[i]) + c2 * r2 * (global_best_position - particles[i])

                # 预计算下一步适应度
                next_particle = particles[i] + velocities[i]
                next_score = self.fitness_function(next_particle)

                # 贪婪更新
                if next_score < scores[i]:
                    particles[i] = next_particle
                # 模拟退火
                elif temp > 1e-5:
                    try:
                        if np.exp((scores[i] - next_score)/temp) > np.random.rand() and temp > 1:
                            particles[i] = next_particle + np.random.normal(0, 0.1 * temp, self.dim)
                    except OverflowError:
                        pass

                # 强制时间约束
                particles[i] = self.constraint_checker.enforce_time_constraints(particles[i], iteration, self.max_iter)

            # 停滞检测与粒子多样化
            if global_best_score == prev_global_best:
                stagnation_count += 1
                if stagnation_count >= PSO_CONFIG["stagnation_threshold"]:
                    print(f"Iter {iteration+1}: 检测到停滞，重新生成粒子...")
                    elite_size = int(PSO_CONFIG["elite_ratio"] * self.n_particles)
                    elite_idx = np.argsort(scores)[:elite_size]
                    new_particles = self.initialize_particles()
                    particles = np.vstack([particles[elite_idx], new_particles[elite_size:]])
                    velocities = np.random.uniform(-1, 1, (self.n_particles, self.dim))
                    stagnation_count = 0

            # 变异策略
            for i in range(self.n_particles):
                # 高斯变异
                if np.random.rand() < PSO_CONFIG["mutation_rate"]:
                    idx = np.random.randint(0, self.dim)
                    particles[i, idx] += np.random.normal(0, 0.5 * (1 - iteration / self.max_iter))
                    particles[i, idx] = max(0.1, particles[i, idx])
                # 柯西突变（后期）
                if iteration > self.max_iter // 2 and np.random.rand() < 0.1:
                    idx = np.random.randint(0, self.dim)
                    particles[i, idx] += np.random.standard_cauchy()
                    particles[i, idx] = max(0.1, particles[i, idx])
                # 强制约束
                particles[i] = self.constraint_checker.enforce_time_constraints(particles[i], iteration, self.max_iter)

            # 打印进度
            if (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1}/{self.max_iter} | 最优适应度: {global_best_score:.2f} | 最优时间分配: {global_best_position[:3]}")

        return global_best_position