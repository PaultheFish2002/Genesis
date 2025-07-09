import torch
import torch.nn.functional as F
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

import argparse
import pickle
import random
import signal
import numpy as np
from QuadrotorMixer import QuadrotorMixer

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
    

class PlaytimeEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        
        # 在此处添加 Mixer
        # self.mixer = QuadrotorMixer("DX141", num_batch=self.num_envs)

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                n_rendered_envs=2,
                show_cameras = True,    # 用于相机固连Debug
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True, 
            ),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())
        
        # add obstacles (mainly Boxes)
        # initialze several kinds
        if self.env_cfg["obstacles"]:
            self.obstacles = []
            self.obstacles_info ={}
            
            for idx in range(self.env_cfg["obstacle_num"]):
                res_x = random.uniform(*self.env_cfg["obstacle_size_range_x"])
                res_y = random.uniform(*self.env_cfg["obstacle_size_range_y"])
                res_z = random.uniform(*self.env_cfg["obstacle_size_range_z"])
                
                self.obstacles.append(
                    self.scene.add_entity(
                    gs.morphs.Box(
                        size=(res_x, res_y, res_z), 
                        pos=(5.0, len(self.obstacles), res_z * 0.5 + 0.001), # 防止坠落影响运算效率
                        quat=euler2quat(0.0, 0.0, 0.0)
                        # quat=euler2quat(random.uniform(0, 2*math.pi), 0.0, 0.0)
                        )
                    ),
                )
                self.obstacles_info[len(self.obstacles)-1] = [res_x, res_y, res_z]

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(0.5, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )
        
        if self.num_envs <=16:
            self.drone_cam = []
            for i in range(self.num_envs):
                self.drone_cam.append(self.scene.add_camera(
                        res=(640, 480),
                        GUI=True,
                    ))

        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # build scene
        self.scene.build(n_envs=num_envs)   # env_spacing=(6.0, 6.0)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        if self.env_cfg["obstacles"]:
            self.base_obstacle_pos = torch.tensor([0, 5.0, 0, 1.0], device=self.device, dtype=gs.tc_float).tile([self.num_envs, len(self.obstacles), 1])
            self.base_obstacle_pos[:,:,0]=torch.range(1, len(self.obstacles))
            self.base_obstacle_pos[:,:,2]=torch.range(1, len(self.obstacles))*0.5

        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        if len(envs_idx):
            while True:
                self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
                self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
                self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)
                
                dist = torch.linalg.norm(self.commands[envs_idx, :2], ord=2, dim=1, keepdim=True)
                dist_mask = dist < 1.5 # 之后设置为参数文件中可调参数
                self.commands[envs_idx, :2][dist_mask.squeeze(1)] /= dist[dist_mask.squeeze(1)] # 尽量远离之
                
                bound_mask = (
                    (self.commands[envs_idx, 0] < self.command_cfg["pos_x_range"][0])
                    | (self.commands[envs_idx, 0] > self.command_cfg["pos_x_range"][1])
                    | (self.commands[envs_idx, 1] < self.command_cfg["pos_y_range"][0])
                    | (self.commands[envs_idx, 1] > self.command_cfg["pos_y_range"][1])
                )
                
                self.commands[envs_idx, 0][bound_mask] = self.commands[envs_idx, 0][bound_mask].clamp(*self.command_cfg["pos_x_range"])
                self.commands[envs_idx, 1][bound_mask] = self.commands[envs_idx, 1][bound_mask].clamp(*self.command_cfg["pos_y_range"])
                
                overlap = False
                
                if self.env_cfg["obstacles"] and not overlap: # 此处亟待修改
                    for idx, pos in self.reset_obstacle_info.items():
                        dist = torch.linalg.norm(torch.cat((self.commands[envs_idx, 0].unsqueeze(1),self.commands[envs_idx, 1].unsqueeze(1)),dim=1).cpu() - torch.tensor(pos[:2]).unsqueeze(0).tile(len(envs_idx.cpu()),1), ord=2, dim=1, keepdim=True).min()
                        if dist < max(max(self.env_cfg["obstacle_size_range_x"]), max(self.env_cfg["obstacle_size_range_y"])):
                            overlap = True
                            break
                    
                if overlap == False:
                    break
                
            if self.target is not None:
                self.target.set_pos(self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx)

    def _at_target(self):
        at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).flatten()
        )
        return at_target

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # exec_actions = self.actions.cpu()
        
        # exec_actions = self.last_actions.cpu() if self.simulate_action_latency else self.actions.cpu()
        # target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        # self.drone.control_dofs_position(target_dof_pos)
        
        
        # 定义当前角速度（rate）和期望角速度（rate_sp）
        current_rate = torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1) * math.pi
        target_rate = self.actions[:, :3] * math.pi # 模拟目标角速度 * math.pi
        thrust = self.actions[:, 3].unsqueeze(1) / 2 + 0.5  # 模拟推力输入
        
        # TODO: 动作输出增量
        
        self.mixer = QuadrotorMixer("DX141", num_batch=self.num_envs)

        # 获取控制输出
        control_outputs = self.mixer.GetMixerControls(current_rate, target_rate, thrust) # [-1,1]
        
        exec_actions = control_outputs[:, :4].cpu()

        # 14468 is hover rpm
        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699) # base_rpm = 14468.429183500699
        # self.drone.set_propellels_rpm((1.2 + exec_actions * 0.3) * 14468.429183500699)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)
    
        # resample commands
        envs_idx = self._at_target()
        self.env_at_target = envs_idx
        self._resample_commands(envs_idx)
        
        # if self.cam is not None:
        if self.env_cfg["visualize_camera"]:
            self.cam.set_pose(
                pos=(6.5 * np.sin(self.episode_length_buf[0].cpu() / 60), 6.5 * np.cos(self.episode_length_buf[0].cpu() / 60), 2.5),
                lookat=(0, 0, 0.5), # 0.5
            )
            
        # set cameras
        if self.num_envs <=16:
            pos_offset = torch.tensor([0.05, 0., -0.025])
            pos_set_cam, forward_set_cam, up_set_cam = calculate_camera_pose(self.base_pos[:], self.base_quat[:], pos_offset.tile([len(self.base_quat),1]))
            for i in range(self.num_envs):
                self.drone_cam[i].set_pose(pos=np.array(pos_set_cam[i]), up=np.array(up_set_cam[i]), lookat=np.array(forward_set_cam[i])) # up=np.array(up_set_cam[0])

        # check termination and reset
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)
        
        # reset cameras
        if self.num_envs <=16:
            pos_offset = torch.tensor([0.05, 0., -0.025])
            pos_set_cam, forward_set_cam, up_set_cam = calculate_camera_pose(self.base_pos[:], self.base_quat[:], pos_offset.tile([len(self.base_quat),1]))
            for i in range(self.num_envs):
                self.drone_cam[i].set_pose(pos=np.array(pos_set_cam[i]), up=np.array(up_set_cam[i]), lookat=np.array(forward_set_cam[i]))
        
        # reset obstacles
        if self.env_cfg["obstacles"]:
            for o_idx, obstacle in enumerate(self.obstacles):
                obstacle.set_pos(self.base_obstacle_pos[envs_idx, o_idx, 1:], zero_velocity=True, envs_idx=envs_idx)
            self.reset_obstacle_info = self.form_obstacles()
            for index, info in self.reset_obstacle_info.items():
                # self.obstacles[index].set_pos(torch.tensor(info).tile([self.num_envs, 1]), zero_velocity=True, envs_idx=envs_idx)
                self.obstacles[index].set_pos(torch.tensor(info).tile([len(envs_idx), 1]), zero_velocity=True, envs_idx=envs_idx)
                self.obstacles[index].set_quat(torch.tensor(euler2quat(random.uniform(0, 2*math.pi), 0.0, 0.0)).tile([len(envs_idx), 1]), zero_velocity=True, envs_idx=envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def form_obstacles(self):
        obstacles_res = {}  # index : pos_x, pos_y, pos_z
        start_p = [0, 0]    # 栅格坐标
        
        node_map = TreeMap(TreeNode(-1, start_p))
        node_map.calc_leaves(node_map.root) # 计算邻接节点地图
        cur_node = node_map.root
        cur_node_idx = cur_node.idx
        cur_node.calc_occupies()
        o_list, c_list = [p for p in cur_node.occupies], [] # 待探索，已探索
        #hist_xy_list = [[0.2, 0.2, 0.0, 0.0]] # 原点安全距离 size_x, size_y, pos_x, pos_y
        
        # 灵活计算水平边界范围为最大变长之两倍
        step_l = 4 * max(self.env_cfg["obstacle_min_spacing"], max(self.env_cfg["obstacle_size_range_x"]), max(self.env_cfg["obstacle_size_range_y"]))
        
        for index, obstacle_info in self.obstacles_info.items():
            
            size_x = obstacle_info[0]
            size_y = obstacle_info[1]
            size_z = obstacle_info[2]
            
            cur_p = torch.tensor(o_list[0])*step_l
            
            pos_z = 0.0     # gs_rand_float(*self.env_cfg["obstacle_pos_range_z"], (1,), torch.device("cpu")).item()
            if pos_z - size_z *0.5 < 0.0: pos_z = math.ceil(size_z*0.5*100)*0.01
            
            while True:
                # 随机生成位置
                pos_x = gs_rand_float(cur_p[0].item(), cur_p[0].item() + step_l, (1,), torch.device("cpu")).item()
                pos_y = gs_rand_float(cur_p[1].item(), cur_p[1].item() + step_l, (1,), torch.device("cpu")).item()
                
                # 检查水平位置与出生点的冲突
                if torch.linalg.norm(torch.tensor([pos_x, pos_y])) < max(0.2 + 0.5 * step_l, 0.5) : continue

                # 检查与相邻其他障碍物的冲突
                overlap = False 
                for idx, pos in obstacles_res.items():
                    dist = torch.linalg.norm(torch.tensor([pos_x, pos_y]) - torch.tensor(pos[:2]))
                    if dist < self.env_cfg["obstacle_min_spacing"]+0.5*max(max(self.env_cfg["obstacle_size_range_x"]), max(self.env_cfg["obstacle_size_range_y"])):
                        overlap = True
                        break
                
                if not overlap:
                    obstacles_res[index] = [pos_x, pos_y, pos_z]
                    c_list.append(o_list.pop(0))
                    if len(o_list) == 0:
                        cur_node_idx += 1
                        cur_node = node_map.get_node_from_idx(cur_node_idx)
                        cur_node.calc_occupies()
                        o_list = [p for p in cur_node.occupies]
                        for o in o_list:
                            if o in c_list:
                                o_list.remove(o)
                    
                    break
        return obstacles_res
        
    
    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        pitch = self.base_euler[:, 1] - vector_to_euler(self.rel_pos)[:,1]
        pitch = torch.where(pitch > 180., pitch - 360., pitch) 
        pitch = torch.where(pitch < -180., pitch + 360., pitch)
        pitch = pitch / 180 * math.pi * 0.5
        
        yaw = self.base_euler[:, 2] - vector_to_euler(self.rel_pos)[:,2]
        yaw = torch.where(yaw > 180.0, yaw - 360., yaw)   
        yaw = torch.where(yaw < -180.0, yaw + 360., yaw)
        yaw = yaw / 180 * math.pi * 0.5 # use rad for yaw_reward, reshape in pi/2
        
        # _, angles = calc_axis_angle(self.base_quat, self.rel_pos)
        rew_yaw = torch.exp(self.reward_cfg["yaw_lambda"] * (torch.abs(yaw)**4)) # 4 in article test in 2
        rew_pitch = torch.exp(self.reward_cfg["yaw_lambda"] * (torch.abs(pitch)**2)) * 0.0 # do not include pitch
        rew = rew_yaw + rew_pitch * 0.5
        # yaw = self.base_euler[:, 2]
        # yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        # yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw)) # TODO: 修改为减去目标相对角度
        return rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
    
    def _reward_attarget(self):
        attarget_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        attarget_rew[self.env_at_target] = 1
        return attarget_rew

from rsl_rl.runners import OnPolicyRunner
import os

if __name__ == "__main__":
    # for Test
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-playtime")
    parser.add_argument("--ckpt", type=int, default=5000) # 随机动作观察环境
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu,
        seed=1,
        logging_level="debug"
    )

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # visualize the target
    env_cfg["visualize_target"] = True
    # for video recording
    env_cfg["visualize_camera"] = args.record
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60
    # set obstacles
    env_cfg["obstacles"] = False

    env = PlaytimeEnv(
        num_envs=4,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"]*2)
    with torch.no_grad():
        if args.record:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, _, rews, dones, infos = env.step(actions)
                env.cam.render()
            env.cam.stop_recording(save_to_filename="playtime_env_test.mp4", fps=env_cfg["max_visualize_FPS"])
        else:
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, _, rews, dones, infos = env.step(actions)
                # for i in range(4):
                #     env.drone_cam[i].render(rgb=False, depth=True)