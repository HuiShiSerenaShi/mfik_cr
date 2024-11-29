import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from robot_kinematics import forward_kinematics_two_link, inverse_kinematics

def sample_positions(theta1_range, theta2_range, l1=1.0, l2=1.0, num_samples=10):
    """
    在指定的关节角度范围内采样末端执行器的可达位置。
    
    参数:
        theta1_range (tuple): θ1 的范围 (min, max) (弧度)。
        theta2_range (tuple): θ2 的范围 (min, max) (弧度)。
        l1 (float): 第一连杆长度。
        l2 (float): 第二连杆长度。
        num_samples (int): 每个关节角度的采样数量。
        
    返回:
        list: 末端执行器的所有可达位置 [x, y]。
    """
    theta1_values = np.linspace(theta1_range[0], theta1_range[1], num_samples)
    theta2_values = np.linspace(theta2_range[0], theta2_range[1], num_samples)
    positions = []

    for theta1 in theta1_values:
        for theta2 in theta2_values:
            pos = forward_kinematics_two_link(theta1, theta2, l1, l2)[:2]
            positions.append(pos)

    return positions

def visualize_sampled_positions(positions):
    """
    可视化末端执行器的采样位置。
    
    参数:
        positions (list): 末端执行器的所有可达位置 [x, y]。
    """
    x_vals = [pos[0] for pos in positions]
    y_vals = [pos[1] for pos in positions]

    plt.figure()
    plt.scatter(x_vals, y_vals, color='blue', label='Sampled Positions')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Sampled End-Effector Positions")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()

def generate_straight_line_trajectory(start_pos, end_pos, total_time, dt, l1=1.0, l2=1.0):
    """
    生成两连杆机械臂末端的直线轨迹，通过正逆运动学验证。
    
    参数:
        start_pos (list or np.ndarray): 起点 [x_start, y_start]。
        end_pos (list or np.ndarray): 终点 [x_end, y_end]。
        total_time (float): 总运动时间。
        dt (float): 时间步长。
        l1 (float): 第一连杆长度。
        l2 (float): 第二连杆长度。

    返回:
        list: 轨迹上的末端位置 [x, y] 列表。
        list: 对应的关节角度 [theta1, theta2] 列表。
    """
    num_points = int(total_time / dt)  # 计算插值点数量
    trajectory = []  # 末端执行器的轨迹
    joint_angles = []  # 对应的关节角度

    # 在直线段中生成点
    x_values = np.linspace(start_pos[0], end_pos[0], num_points)
    y_values = np.linspace(start_pos[1], end_pos[1], num_points)

    for x, y in zip(x_values, y_values):
        target_pos = [x, y]
        try:
            # 使用逆运动学计算关节角度
            theta1, theta2 = inverse_kinematics(target_pos, l1, l2)
            # 使用正运动学验证生成位置
            verified_pos = forward_kinematics_two_link(theta1, theta2, l1, l2)[:2]

            if np.allclose(target_pos, verified_pos, atol=1e-4):
                trajectory.append(target_pos)
                joint_angles.append([theta1, theta2])
            else:
                raise ValueError("正运动学验证失败！")
        except ValueError as e:
            print(f"点 {target_pos} 无法到达，跳过: {e}")

    # 输出轨迹信息
    print(f"生成的直线轨迹包含 {len(trajectory)} 个点。")
    return trajectory, joint_angles

def generate_velocities(trajectory, total_time, dt):
    """
    根据轨迹生成每个点的速度。
    
    参数:
        trajectory (list): 轨迹上的末端位置 [x, y] 列表。
        total_time (float): 总运动时间。
        dt (float): 时间步长。
        
    返回:
        list: 每个点的速度向量 [vx, vy] 列表。
    """
    num_points = len(trajectory)
    velocities = []

    for i in range(num_points):
        if i == 0:
            # 起点速度为零
            velocities.append([0.0, 0.0])
        else:
            # 差分法计算速度
            vx = (trajectory[i][0] - trajectory[i - 1][0]) / dt
            vy = (trajectory[i][1] - trajectory[i - 1][1]) / dt
            velocities.append([vx, vy])

    return velocities

# 可视化轨迹
def visualize_trajectory(trajectory):
    """
    可视化末端轨迹。
    参数:
        trajectory (list): 轨迹上的末端位置 [x, y] 列表。
    """
    x_vals = [pos[0] for pos in trajectory]
    y_vals = [pos[1] for pos in trajectory]

    plt.figure()
    plt.plot(x_vals, y_vals, 'r-', label='Generated Trajectory')
    plt.scatter(x_vals[0], y_vals[0], color='green', label='Start Point')  # 起点
    plt.scatter(x_vals[-1], y_vals[-1], color='blue', label='End Point')  # 终点
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("End-Effector Straight Line Trajectory")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()

# 示例测试
if __name__ == "__main__":
    # 设定参数
    l1, l2 = 1.0, 1.0
    theta1_range = (np.radians(0), np.radians(90))
    theta2_range = (np.radians(30), np.radians(60))
    num_samples = 20  # 每个关节角度采样数量
    TD = 20.0  # 总时间
    DT = 0.001  # 时间步长
        # 采样末端位置
    sampled_positions = sample_positions(theta1_range, theta2_range, l1, l2, num_samples)

    # 可视化采样位置
    visualize_sampled_positions(sampled_positions)

        # 选择起点和终点（这里手动选择，也可以自动选择）
    start_pos = sampled_positions[0]  # 假设第一个点为起点
    end_pos = sampled_positions[-1]   # 假设最后一个点为终点
    start_pos = [-0.86,1.5]  # 假设第一个点为起点
    end_pos = [-0.52,1.37] 
    print(f"选择的起点: {start_pos}")
    print(f"选择的终点: {end_pos}")
    # 生成直线轨迹
    trajectory, joint_angles = generate_straight_line_trajectory(start_pos, end_pos, TD, DT, l1, l2)

    # 生成速度
    velocities = generate_velocities(trajectory, TD, DT)

# 输出测试结果
    for i, (pos, angles, vel) in enumerate(zip(trajectory, joint_angles, velocities)):
        print(f"点 {i + 1}: 位置 {pos}, 关节角度 θ1={np.degrees(angles[0]):.2f}°, θ2={np.degrees(angles[1]):.2f}°, 速度: vx={vel[0]:.4f}, vy={vel[1]:.4f}")

    #点 1: 位置 [-0.86, 1.5], 关节角度 θ1=89.66°, θ2=60.34° 速度: vx=0.0000, vy=0.0000
    # 点 20000: 位置 [-0.52, 1.37], 关节角度 θ1=67.90°, θ2=85.78° 速度: vx=0.0170, vy=-0.0065
    # 可视化轨迹
    visualize_trajectory(trajectory)