# main.py
from optimization_core import PSOOptimizer
from visualization import TrajectoryVisualizer
from trajectory_generator import TrajectoryGenerator

def main():
    # 配置参数
    CSV_PATH = r"robot_trajectory.csv"
    START_POINT = [-94.55, 34.66, -52.71, 5.70, 57.90, -93.45]       # 起始关节角度（度）
    END_POINT = [-110.58, 70.08, -35.55, -76.97, 72.49, 52.44]       # 终止关节角度（度）

    # 初始化优化器并执行优化
    optimizer = PSOOptimizer(CSV_PATH, START_POINT, END_POINT)
    optimized_params = optimizer.optimize()

    # 可视化结果
    trajectory_gen = TrajectoryGenerator(START_POINT, END_POINT)
    visualizer = TrajectoryVisualizer(trajectory_gen)
    visualizer.visualize(optimized_params)

    # 保存优化数据
    visualizer.save_optimized_data(optimized_params)

if __name__ == "__main__":
    main()