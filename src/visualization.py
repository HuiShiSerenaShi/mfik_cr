import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from trajectory_generation import generate_straight_line_trajectory
from robot_kinematics import forward_kinematics_two_link

def visualize_tracking_with_arm(target_trajectory, actual_positions, joint_angles, l1=1.0, l2=1.0, interval=50, debug=False):
    """
    动态可视化轨迹追踪，包括两连杆的运动和末端执行器轨迹。
    
    参数:
        target_trajectory (list): 目标轨迹 [x, y]。
        actual_positions (list): 实际轨迹 [x, y]。
        joint_angles (list): 关节角度历史记录。
        l1 (float): 第一连杆长度。
        l2 (float): 第二连杆长度。
        interval (int): 动画刷新间隔 (ms)。
        debug (bool): 是否启用调试模式，显示实时误差信息。
    """
    fig, ax = plt.subplots()

    max_range = l1 + l2 + 0.5  # 动态计算最大范围并加边距
    ax.set_xlim(-max_range, max_range)  # x轴范围
    ax.set_ylim(-max_range, max_range)  # y轴范围
    ax.set_aspect('equal')
    ax.grid(True)

    # 绘制目标轨迹
        # 绘制目标轨迹
    target_x, target_y = zip(*target_trajectory)
    ax.plot(target_x, target_y, 'g--', label="Target Trajectory")

    # 动态绘制
    actual_x, actual_y = [], []
    actual_line, = ax.plot([], [], 'r-', label="Actual Trajectory")
    end_effector, = ax.plot([], [], 'bo', label="End Effector")
    link1_line, = ax.plot([], [], 'ro-', label="Link 1")
    link2_line, = ax.plot([], [], 'bo-', label="Link 2")

    # 可选调试信息
    text = None
    if debug:
        text = ax.text(0.05, 0.95, "", transform=ax.transAxes, verticalalignment='top')

    def update(frame):
        # 更新实际轨迹
        actual_x.append(actual_positions[frame][0])
        actual_y.append(actual_positions[frame][1])
        actual_line.set_data(actual_x, actual_y)
        end_effector.set_data(actual_positions[frame])

        # 更新两连杆位置
        theta1, theta2 = joint_angles[frame]
        joint1_x = l1 * np.cos(theta1)
        joint1_y = l1 * np.sin(theta1)
        joint2_x = joint1_x + l2 * np.cos(theta1 + theta2)
        joint2_y = joint1_y + l2 * np.sin(theta1 + theta2)

        link1_line.set_data([0, joint1_x], [0, joint1_y])
        link2_line.set_data([joint1_x, joint2_x], [joint1_y, joint2_y])

        # 更新调试信息
        if debug:
            error = np.linalg.norm(np.array(target_trajectory[frame]) - np.array(actual_positions[frame]))
            joint_info = f"Joint Angles: θ1={np.degrees(theta1):.2f}°, θ2={np.degrees(theta2):.2f}°"
            text.set_text(f"Step: {frame}\nError: {error:.4f}\n{joint_info}")

        return actual_line, end_effector, link1_line, link2_line, text

    ani = FuncAnimation(fig, update, frames=len(actual_positions), interval=interval, blit=True)
    plt.legend()
    plt.show()

# 示例测试
if __name__ == "__main__":
# 设定参数
    l1, l2 = 1.0, 1.0
    TD = 5.0  # 总时间
    DT = 0.01  # 时间步长
    start_pos = [-0.86,1.5]  # 假设第一个点为起点
    end_pos = [-0.52,1.37] 
    print(f"选择的起点: {start_pos}")
    print(f"选择的终点: {end_pos}")

    # 生成目标轨迹和关节角度
    print("生成目标轨迹...")
    target_trajectory, joint_angles = generate_straight_line_trajectory(start_pos, end_pos, TD, DT, l1, l2)
    
# 验证轨迹的实际末端位置（通过正向运动学）
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

    # 可视化轨迹追踪
    print("开始可视化轨迹追踪...")
    visualize_tracking_with_arm(
        target_trajectory=target_trajectory,
        actual_positions=actual_positions,
        joint_angles=joint_angles,
        l1=l1,
        l2=l2,
        interval=50,
        debug=True
    )