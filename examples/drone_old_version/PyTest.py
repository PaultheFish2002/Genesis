import torch
import numpy as np
import math

def euler_to_rotVec(yaw, pitch, roll):
    Rmat = euler_to_rotMat(yaw, pitch, roll)

    theta = math.acos(((Rmat[0, 0] + Rmat[1, 1] + Rmat[2, 2]) - 1) / 2)
    sin_theta = math.sin(theta)
    if sin_theta == 0:
        rx, ry, rz = 0.0, 0.0, 0.0
    else:
        multi = 1 / (2 * math.sin(theta))
        rx = multi * (Rmat[2, 1] - Rmat[1, 2]) * theta
        ry = multi * (Rmat[0, 2] - Rmat[2, 0]) * theta
        rz = multi * (Rmat[1, 0] - Rmat[0, 1]) * theta
    return rx, ry, rz

def euler_to_rotMat(yaw, pitch, roll):
    Rz_yaw = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw),  torch.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = torch.tensor([
        [ torch.cos(pitch), 0, torch.sin(pitch)],
        [             0, 1,             0],
        [-torch.sin(pitch), 0, torch.cos(pitch)]])
    Rx_roll = torch.tensor([
        [1,            0,             0],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll),  torch.cos(roll)]])
    rotMat = torch.dot(Rz_yaw, torch.dot(Ry_pitch, Rx_roll))
    return rotMat

def euler2quat(yaw, pitch, roll): # 输入弧度值
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return (qw, qx, qy, qz)

def calc_axis_angle(quaternions, rel_vector):
    """
    根据多个姿态四元数计算 UAV 体坐标系的 x 轴单位向量，
    并计算每个 x 轴向量与另一个空间向量（非单位向量）之间的夹角（弧度）。
    
    参数:
      quaternions: Tensor, 形状为 (B, 4)，每一行为一个四元数 [w, x, y, z]
      rel_vector: Tensor, 形状为 (B, 3)，另一个空间向量（非单位向量）
      
    返回:
      x_axes: Tensor, 形状为 (B, 3)，每行为对应的 UAV 体坐标系 x 轴单位向量（在世界坐标系下）
      angles:  Tensor, 形状为 (B,)，每个元素是 x 轴向量与 other_vector 的夹角（弧度）
    """
    # 将四元数归一化，防止数值误差
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
    
    # 分解四元数，顺序为 [w, x, y, z]
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]
    
    # 根据四元数转换公式，计算旋转矩阵的第一列，即 x 轴向量
    # 第一列的三个分量为：

    x_axis_x = 1 - 2 * (y ** 2 + z ** 2)
    x_axis_y = 2 * (x * y + z * w)
    x_axis_z = 2 * (x * z - y * w)
    
    # 拼接得到批量的 x 轴向量，形状为 (B, 3)
    x_axes = torch.stack([x_axis_x, x_axis_y, x_axis_z], dim=1)
    # 保证 x_axes 为单位向量（理论上已经是单位向量，但可归一化以防误差）
    x_axes = x_axes / x_axes.norm(dim=1, keepdim=True)
    
    # 计算 other_vector 的模（假设 other_vector 为 (3,)）
    norm_rel = torch.norm(rel_vector, dim=1)
    if torch.any(norm_rel == 0):
        raise ValueError("other_vector 的模不能为 0")
    
    # 计算每个 x 轴向量与 other_vector 的点积
    dot_products = torch.sum(x_axes * rel_vector, dim=1)
    # 由于 x_axes 为单位向量，余弦值为 dot / norm(other_vector)
    cos_angles = dot_products / norm_rel
    # Clamp 限制余弦值在 [-1, 1] 内
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    
    # 计算夹角（弧度）
    angles = torch.acos(cos_angles)
    
    return x_axes, angles

def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    q: (w, x, y, z)
    """
    w, x, y, z = q
    return torch.tensor([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

def calculate_camera_pose(drone_position, drone_quaternion, camera_offset):
    """
    计算摄像机的位置和朝向
    参数：
        drone_position: 无人机全局坐标 (x, y, z)
        drone_quaternion: 无人机姿态四元数 (w, x, y, z)
        camera_offset: 摄像机相对于无人机的偏移量 (x_offset, y_offset, z_offset)
    返回：
        camera_position: 摄像机全局坐标
        camera_forward: 摄像机前方向
        camera_up: 摄像机上方向
    """
    # R_= euler_to_rotMat(math.radians(90.),math.radians(0.),math.radians(0.))
    # 计算旋转矩阵
    R = quaternion_to_rotation_matrix(drone_quaternion)
    
    # 计算摄像机全局位置
    camera_position = drone_position + torch.linalg.inv(R) @ camera_offset
    
    # 计算摄像机前方向和上方向
    F_drone = torch.tensor([1., 0., 0.])  # 无人机前方向
    U_drone = torch.tensor([0., 0., 1.])  # 无人机上方向
    camera_forward = torch.linalg.inv(R) @ F_drone
    camera_up = torch.linalg.inv(R)@ U_drone
    
    return camera_position, camera_forward, camera_up

if __name__ == "__main__":

    # 示例
    drone_position = torch.tensor([10, 20, 30])  # 无人机全局坐标
    q = euler2quat(math.radians(0.),math.radians(0.),math.radians(90.)) # yaw, pitch, roll
    drone_quaternion = torch.tensor(q)  # 无人机姿态四元数
    camera_offset = torch.tensor([0.1, 0, -0.05])  # 摄像机相对于无人机的偏移量
    
    q_input = torch.tensor([euler2quat(math.radians(0.),math.radians(0.),math.radians(90.)),
                            euler2quat(math.radians(90.),math.radians(0.),math.radians(0.))])
    
    vector = torch.tensor([[1.,-1.,1.], [1.,-1.,1.]])

    x_axes, angles = calc_axis_angle(q_input, vector)
    
    camera_position, camera_forward, camera_up = calculate_camera_pose(drone_position, drone_quaternion, camera_offset)
    # print("摄像机全局位置:", camera_position)
    # print("摄像机前方向:", camera_forward)
    # print("摄像机上方向:", camera_up)

    # TODO: Debug