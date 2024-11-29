import numpy as np


# DH参数：链接的参数定义
def dh_transform(a, alpha, d, theta):
    """
    Denavit-Hartenberg 转换矩阵。
    参数:
        a (float): x轴的偏移量。
        alpha (float): x旋转角度。
        d (float): z偏移量。
        theta (float): z旋转角度。
    返回:
        np.ndarray: 4x4 转换矩阵。
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),              d],
        [0,              0,                           0,                          1]
    ])

def forward_kinematics_two_link(theta1, theta2, l1=1.0, l2=1.0):
    """
    计算两连杆机械臂的末端位置。
    参数:
        theta1 (float): 第一关节角度。
        theta2 (float): 第二关节角度。
        l1 (float): 第一连杆长度。
        l2 (float): 第二连杆长度。
    返回:
        np.ndarray: 末端执行器的位置 (x, y)。
    """
    # 更新后的两连杆 DH 参数表
    dh_params = [
        (l1, 0, 0, theta1),  # 第一连杆
        (l2, 0, 0, theta2)   # 第二连杆
    ]

    T = np.eye(4)  # 初始单位矩阵
    for params in dh_params:
        T = T @ dh_transform(*params)  # 累乘变换矩阵

    return T[:3, 3]  # 提取末端位置 (x, y, z)


def jacobian_3d(angles):
    """
    计算两连杆机械臂在三维空间中的雅可比矩阵 (6x2)。
    包括线速度和角速度部分。

    参数:
        angles (list or np.ndarray): 关节角度 [theta1, theta2]
    返回:
        np.ndarray: 雅可比矩阵 (6x2)
    """
    l1, l2 = 1.0, 1.0  # 连杆长度
    theta1, theta2 = angles

    # 位置部分 (线速度 Jacobian)
    j11 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    j12 = -l2 * np.sin(theta1 + theta2)
    j21 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    j22 = l2 * np.cos(theta1 + theta2)
    j31 = 0  # z 方向不受影响，因为两连杆在平面内运动
    j32 = 0

    r11 = 0
    r12 = 0
    r21 = 0
    r22 = 0
    r31 = 1  # 第一关节对总的绕 z 旋转有贡献
    r32 = 1  # 第二关节对总的绕 z 旋转有贡献

    # 合并线速度和角速度部分
    jacobian_matrix = np.array([
        [j11, j12],  
        [j21, j22],  
        [j31, j32],  
        [r11, r12],  
        [r21, r22], 
        [r31, r32], 
    ])

    return jacobian_matrix



def inverse_kinematics(target_pos, l1=1.0, l2=1.0):
    """
    计算两连杆机器人的逆运动学。
    参数:
        target_pos (list or np.ndarray): 目标末端位置 [x, y]
        l1 (float): 第一连杆长度
        l2 (float): 第二连杆长度
    返回:
        tuple: 两个关节角度 [theta1, theta2] (单位: 弧度)
    """
    x, y = target_pos

    # 检查目标点是否在可达范围内
    distance = np.sqrt(x**2 + y**2)
    if distance > (l1 + l2):
        raise ValueError("目标点超出机械臂的可达范围！")
    if distance < abs(l1 - l2):
        raise ValueError("目标点在机械臂的奇异区域内！")

    # 计算 theta2
    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    sin_theta2 = np.sqrt(1 - cos_theta2**2)  # 默认选择正解
    theta2 = np.arctan2(sin_theta2, cos_theta2)

    # 计算 theta1
    k1 = l1 + l2 * cos_theta2
    k2 = l2 * sin_theta2
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return theta1, theta2


# 测试代码
if __name__ == "__main__":
    l1, l2 = 1.0, 1.0  # 连杆长度
    theta1, theta2 = np.pi / 7, np.pi / 5  # 两关节角度

    # 打印初始关节角度（角度制）
    print(f"初始关节角度: θ1 = {np.degrees(theta1):.2f}°, θ2 = {np.degrees(theta2):.2f}°")

    # 计算末端位置
    end_effector_pos = forward_kinematics_two_link(theta1, theta2, l1, l2)
    print("末端执行器位置:", end_effector_pos)

    # 理论验证
    expected_x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    expected_y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    print(f"理论计算位置: x={expected_x}, y={expected_y}")

# 测试逆运动学
    computed_theta1, computed_theta2 = inverse_kinematics(end_effector_pos[:2])
    print(f"目标位置: {end_effector_pos}")
    print(f"计算得到的关节角度: θ1 = {np.degrees(computed_theta1):.2f}°, θ2 = {np.degrees(computed_theta2):.2f}°")

    # 验证正逆运动学一致性
    print("\n验证正逆运动学一致性:")
    assert np.allclose([theta1, theta2], [computed_theta1, computed_theta2], atol=1e-4), \
        "正逆运动学结果不一致！"
    print("正逆运动学验证通过！")

# 测试雅可比矩阵
    print("\n测试雅可比矩阵:")
    angles = [theta1, theta2]
    J = jacobian_3d(angles)
    print("雅可比矩阵:\n", J)