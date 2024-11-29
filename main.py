import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # 加入根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))  # 加入 src 目录

import numpy as np
from src.trajectory_generation import generate_straight_line_trajectory, generate_velocities
from src.trajectory_tracking import znn_trajectory_tracking
from src.visualization import visualize_tracking_with_arm
from src.robot_kinematics import forward_kinematics_two_link

if __name__ == "__main__":
    # 参数设置
    l1, l2 = 1.0, 1.0
    TD = 5.0  # 总时间
    DT = 0.01  # 时间步长
    beta1 = 1.0  # ZNN1 的 Beta 参数
    beta2 = 1.0  # ZNN2 的 Beta 参数
    start_pos = [-0.86, 1.5] # 轨迹起点
    end_pos = [-0.52, 1.37] # 轨迹终点
    
    print(f"选择的起点: {start_pos}")
    print(f"选择的终点: {end_pos}")

    # 生成目标轨迹
    print("生成目标轨迹...")
    target_trajectory, joint_angles = generate_straight_line_trajectory(start_pos, end_pos, TD, DT, l1, l2)

    # 验证轨迹的实际末端位置
    print("验证正向运动学...")
    actual_positions = []
    for angles in joint_angles:
        pos = forward_kinematics_two_link(angles[0], angles[1], l1, l2)[:2]
        actual_positions.append(pos)

    # 输出验证信息
    for i, (target_pos, actual_pos) in enumerate(zip(target_trajectory, actual_positions)):
        if not np.allclose(target_pos, actual_pos, atol=1e-4):
            print(f"误差过大！目标点: {target_pos}, 实际点: {actual_pos}")
            break
    else:
        print("验证通过：正向运动学与目标轨迹一致！")

        # 生成初始关节角度
    initial_angles = joint_angles[0]  # 使用轨迹的起点对应的角度

        # 使用轨迹生成模块生成速度
    print("生成目标速度...")
    target_velocities = generate_velocities(target_trajectory, TD, DT)

    # 轨迹追踪
    # 轨迹追踪
    print("开始轨迹追踪...")
    znn_positions, angles_history, errors = znn_trajectory_tracking(
        initial_angles=initial_angles,
        trajectory=target_trajectory,
        velocities=target_velocities,
        dt=DT,
        l1=l1,
        l2=l2,
        beta1=beta1,
        beta2=beta2
    )

 # 设置保存路径
    save_path = None
    #save_path = r"D:\studyVR\thesis\mfik_cr\results\animations\trajectory_tracking.mp4"

    # 可视化轨迹追踪
    print("开始可视化轨迹追踪...")
    visualize_tracking_with_arm(
        target_trajectory=target_trajectory,
        actual_positions=znn_positions,
        joint_angles=angles_history,
        l1=l1,
        l2=l2,
        interval=50,
        debug=True,
        save_path=save_path
    )