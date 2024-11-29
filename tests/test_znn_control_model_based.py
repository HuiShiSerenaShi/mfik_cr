import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
from trajectory_generation import generate_straight_line_trajectory, generate_velocities
from visualization import visualize_tracking_with_arm
from znn_control_model_based import znn_trajectory_tracking

def test_znn_trajectory_tracking():
    # 参数设置
    l1, l2 = 1.0, 1.0
    TD = 5.0  # 总时间
    DT = 0.01  # 时间步长
    start_pos = [-0.86, 1.5] # 轨迹起点
    end_pos = [-0.52, 1.37] # 轨迹终点
    
    print(f"选择的起点: {start_pos}")
    print(f"选择的终点: {end_pos}")

    # 生成目标轨迹
    trajectory, joint_angles = generate_straight_line_trajectory(start_pos, end_pos, TD, DT, l1, l2)
    initial_angles = joint_angles[0]  # 使用轨迹的起点对应的角度
    velocities = generate_velocities(trajectory, TD, DT)
    trajectory_with_velocities = list(zip(trajectory, velocities))

    # 执行 ZNN 轨迹追踪
    joint_angles_history, actual_positions = znn_trajectory_tracking(
        initial_angles=initial_angles,
        trajectory=trajectory_with_velocities,
        tau=0.0001,
        h=1.0
    )

    # 可视化
    visualize_tracking_with_arm(
        target_trajectory=trajectory,
        actual_positions=actual_positions,
        joint_angles=joint_angles_history,
        l1=l1,
        l2=l2,
        interval=50,
        debug=True
    )
    print("测试完成！")

if __name__ == "__main__":
    test_znn_trajectory_tracking()
