# visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import JOINT_NUM, TRAJECTORY_CONFIG

class TrajectoryVisualizer:
    def __init__(self, trajectory_generator):
        self.trajectory_gen = trajectory_generator

    def visualize(self, optimized_params):
        """可视化优化后的轨迹、速度、末端执行器路径"""
        # 1. 关节角度轨迹
        plt.figure(figsize=(12, 6))
        time_allocation = optimized_params[:TRAJECTORY_CONFIG["num_segments"]]
        t_total = sum(time_allocation)
        t_span = np.linspace(0, t_total, 100)
        angles = np.array([self.trajectory_gen.polynomial_trajectory(optimized_params, t) for t in t_span])

        for j in range(JOINT_NUM):
            plt.plot(t_span, np.degrees(angles[:, j]), label=f'Joint {j+1}')
        plt.title('Optimized Joint Trajectories')
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Angle (deg)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 2. 末端执行器3D路径
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        positions = np.array([self.trajectory_gen.forward_kinematics(a) for a in angles])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Optimized Path')
        ax.set_title('End-Effector 3D Path')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.show()

        # 3. 关节速度曲线
        plt.figure(figsize=(12, 6))
        dt = t_span[1] - t_span[0]
        velocities = np.gradient(angles, axis=0) / dt
        for j in range(JOINT_NUM):
            plt.plot(t_span, np.degrees(velocities[:, j]), label=f'Joint {j+1}')
        plt.title('Joint Velocities')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (deg/s)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_optimized_data(self, optimized_params, filename='optimized_trajectory.csv'):
        """保存优化后的轨迹数据到CSV"""
        time_allocation = optimized_params[:TRAJECTORY_CONFIG["num_segments"]]
        t_total = sum(time_allocation)
        t_interp = np.linspace(0, t_total, 100)
        optimized_angles = np.array([self.trajectory_gen.polynomial_trajectory(optimized_params, t) for t in t_interp])

        df = pd.DataFrame({'Time': t_interp})
        for j in range(JOINT_NUM):
            df[f'j1{j}'] = np.degrees(optimized_angles[:, j])
        df.to_csv(filename, index=False)
        print(f"优化后的数据已保存到 {filename}")