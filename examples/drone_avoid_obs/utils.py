import math
import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum

class ObstacleType(Enum):
    BOX = 0
    CYLINDER = 1
    SPHERE = 2

# 相机绑定
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def vector_to_euler(vector: torch.Tensor): # torch.tensor
    # 将向量转换为单位向量
    # vector = torch.tensor(vector, dtype=torch.float32)
    vector = F.normalize(vector, p=2, dim=1)

    # 计算与x轴的角度（绕z轴旋转的角度，即偏航角yaw）
    yaw = torch.atan2(vector[:,1], vector[:,0])

    # 计算与z轴的角度（绕x轴旋转的角度，即俯仰角pitch）
    # 由于向量可能不在x-y平面上，我们需要先计算向量在x-y平面上的投影
    # proj_xy = torch.norm(vector[:,:2])
    # pitch = torch.atan2(-vector[:,2], proj_xy)
    pitch = torch.asin(vector[:,2])

    # 计算与y轴的角度（绕y轴旋转的角度，即翻滚角roll）
    # 为了计算roll，我们需要先考虑绕z轴旋转到x-y平面，然后计算绕y轴旋转的角度
    # roll = torch.atan2(vector[:,0] * torch.cos(pitch), vector[:,2])
    roll = torch.tensor(0.0, device=vector.device).tile(len(pitch))

    # 将弧度转换为度
    return torch.stack([roll, pitch, yaw], dim=-1) * 180.0 / torch.tensor(np.pi)


def calc_axis_angle(quaternions, rel_vector):
    """
    根据多个姿态四元数计算 UAV 体坐标系的 x 轴单位向量，
    并计算每个 x 轴向量与另一个空间向量（非单位向量）之间的夹角（弧度）。
    
    参数:
      quaternions: Tensor, 形状为 (B, 4)，每一行为一个四元数 [w, x, y, z]
      rel_vector: Tensor, 形状为 (B, 3)，另一个空间向量（非单位向量）
      
    返回:
      x_axes: Tensor, 形状为 (B, 3)，每行为对应的 UAV 体坐标系 x 轴单位向量（在世界坐标系下）
      angles:  Tensor, 形状为 (B,)，每个元素是 x 轴向量与 other_vector 的夹角（弧度）实际上是[0,pi]
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

def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    q: (w, x, y, z)
    """
    res = torch.tensor([])
    
    for i in range(len(q)):
        w, x, y, z = q[i,0], q[i,1], q[i,2], q[i,3]
        R_ = torch.tensor([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ]).unsqueeze(0)
        res = torch.cat((res, R_))
    
    return res

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
    # camera_position = drone_position.cpu() + (torch.linalg.inv(R) @ camera_offset.unsqueeze(2)).squeeze(2)
    camera_position = drone_position.cpu() + (R @ camera_offset.unsqueeze(2)).squeeze(2)
    
    # 计算摄像机前方向和上方向
    F_drone = torch.tensor([1., 0., 0.]).tile([len(R),1])  # 无人机前方向
    U_drone = torch.tensor([0., 0., 1.]).tile([len(R),1])  # 无人机上方向
    # camera_forward = (torch.linalg.inv(R) @ F_drone.unsqueeze(2)).squeeze(2)
    # camera_up = (torch.linalg.inv(R) @ U_drone.unsqueeze(2)).squeeze(2)
    camera_forward = (R @ F_drone.unsqueeze(2)).squeeze(2)
    camera_up = (R @ U_drone.unsqueeze(2)).squeeze(2)
    
    return camera_position, camera_forward + drone_position.cpu(), camera_up

# 示例
# drone_position = np.array([10, 20, 30])  # 无人机全局坐标
# drone_quaternion = np.array([0.707, 0, 0, 0.707])  # 无人机姿态四元数
# camera_offset = np.array([0.1, 0, -0.05])  # 摄像机相对于无人机的偏移量

# camera_position, camera_forward, camera_up = calculate_camera_pose(drone_position, drone_quaternion, camera_offset)
# print("摄像机全局位置:", camera_position)
# print("摄像机前方向:", camera_forward)
# print("摄像机上方向:", camera_up)


# 用于辅助障碍物生成网格的树结构
class TreeNode:
    def __init__(self, parent=-1, value=None, idx=None):
        self.idx = idx
        self.parent = parent
        self.sons = []
        self.value = value
        self.occupies = [value]
    def set_parent(self, parent):
        self.parent = parent
    def set_idx(self, idx):
        self.idx = idx
    def calc_occupies(self):
        for i in range(4):
            if i == 1:
                self.occupies.append([self.value[0], self.value[1]-1])
            elif i == 2:
                self.occupies.append([self.value[0]-1, self.value[1]-1])
            elif i == 3:
                self.occupies.append([self.value[0]-1, self.value[1]])

class TreeMap:
    def __init__(self, root:TreeNode): # root 是一个 TreeNode 类型数据
        self.root = root
        self.root.idx = 0
        # self.value_map = {root : root.value} # 通过value来查找node
        self.idx_map = {self.root.idx : root}
        self.length = 1
        self.expand = False
        
    def calc_leaves(self, node:TreeNode):
        self.expand = True
        for i in range(8):
            if i == 0:
                value = [node.value[0], node.value[1]+1]
            elif i == 1:
                value = [node.value[0]+1, node.value[1]+1]
            elif i == 2:
                value = [node.value[0]+1, node.value[1]]
            elif i == 3:
                value = [node.value[0]+1, node.value[1]-1]
            elif i == 4:
                value = [node.value[0], node.value[1]-1]
            elif i == 5:
                value = [node.value[0]-1, node.value[1]-1]
            elif i == 6:
                value = [node.value[0]-1, node.value[1]]
            elif i == 7:
                value = [node.value[0]-1, node.value[1]+1]
            if value not in [v.value for v in self.idx_map.values()]:
                node.sons.append(TreeNode(node, value, self.length))
                self.idx_map[self.length] = node.sons[-1]
                self.length += 1
    
    def get_node_from_idx(self, idx):
        return self.idx_map[idx]
    
    
    
# Debug
if __name__ == "__main__":
    obstacle_type = ObstacleType(2)
    print(obstacle_type)  # 输出: ObstacleType.Cylinder