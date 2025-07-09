import torch
import torch.nn.functional as F
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np


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

class HoverEnv:
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

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())
        
        # horizontal_scale = 0.25
        # vertical_scale = 0.005
        # height_field = np.zeros([40, 40])
        # heights_range = np.arange(-10, 20, 10)
        # height_field[5:35, 5:35] = np.random.choice(heights_range, (30, 30))
        # ########################## entities ##########################
        # terrain = self.scene.add_entity(
        #     morph=gs.morphs.Terrain(
        #         horizontal_scale=horizontal_scale,
        #         vertical_scale=vertical_scale,
        #         height_field=height_field,
        #     ),
        # )

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
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(4.0, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # build scene
        self.scene.build(n_envs=num_envs)

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
        
        # target 超时指标
        self.target_timeout = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.at_target_flag = torch.zeros_like(self.target_timeout)

        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)
        
        dist_mask = torch.linalg.norm(self.commands[envs_idx, :2], ord=2, dim=1, keepdim=True) < 1.0 # 之后设置为参数文件中可调参数
        self.commands[envs_idx, :2][dist_mask.squeeze(1)] *= -1.0 # 尽量远离之
        # d = torch.linalg.norm(self.commands[envs_idx, :2], ord=2, dim=1, keepdim=True)
        # if d.min() < 0.2:
        #     while True:    
        #         overlap = False
        #         d = torch.linalg.norm(self.commands[envs_idx, :2], ord=2, dim=1, keepdim=True)
        #         if d.min() < 0.2:
        #             overlap = True
        #         if overlap == False:
        #             break        
        if self.target is not None:
            self.target.set_pos(self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx)

    def _at_target(self):
        at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).flatten()
        )
        new_at_target_mask = (self.at_target_flag[at_target] == 0).nonzero(as_tuple=False).flatten()
        self.at_target_flag[at_target[new_at_target_mask]] = 1
        return at_target

    def _at_target_timeout(self, at_target):
        once_at_target_mask = (self.at_target_flag == 1).nonzero(as_tuple=False).flatten()
        # self.target_timeout[once_at_target_mask] += 1
        self.target_timeout += 1 # 所有目标点定时保留，与是否到达目标点无关
        at_target_timeout = (self.target_timeout >= 1000).nonzero(as_tuple=False).flatten()
        self.target_timeout[at_target_timeout] = 0
        return at_target_timeout
    
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions.cpu()
        # exec_actions = self.last_actions.cpu() if self.simulate_action_latency else self.actions.cpu()
        # target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        # self.drone.control_dofs_position(target_dof_pos)
        
        # 在此处添加 Mixer

        # 14468 is hover rpm
        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)
        self.scene.step()
        # self.step_counter += 1
        # self.cam.set_pose(
        #     pos=(4.0 * np.sin(self.step_counter / 60), 4.0 * np.cos(self.step_counter / 60), 2.5),
        #     lookat=(0, 0, 0.5), # 0.5
        # )

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
        at_target = self._at_target()
        self.env_at_target = at_target
        envs_idx = self._at_target_timeout(at_target)
        # self.env_at_target = envs_idx
        self._resample_commands(envs_idx)

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

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.step_counter = 0
        
        self.at_target_flag[envs_idx] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        self.env_at_target = envs_idx

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_target(self):
        # TODO: 接入相机视线的影响
        # pitch = self.base_euler[:, 1] - vector_to_euler(self.rel_pos)[:,1]
        # pitch = torch.where(pitch > 180, pitch - 360, pitch) 
        # yaw = self.base_euler[:, 2] - vector_to_euler(self.rel_pos)[:,2]
        # yaw = torch.where(yaw > 180, yaw - 360, yaw) 
        
        # _, angles = calc_axis_angle(self.base_quat, self.rel_pos)
        
        # mask = angles * 180 / math.pi > 40 # 超过40度可认为target不在视线范围内
        
        approach_vec_diff = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        
        # 加投影 
        toward_target_vec = torch.nn.functional.normalize(self.rel_pos, p=2, dim=1)
        
        projected_vel = torch.sum(torch.nn.functional.normalize(self.base_lin_vel, p=2, dim=1) * toward_target_vec, dim=1)
        projected_vel = torch.relu(projected_vel) # 只给正向奖励
        
        projected_approach_vec = projected_vel * approach_vec_diff
        
        at_target_mask = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).flatten()
        )
        target_rew = projected_approach_vec # 掩码
        target_rew[at_target_mask] = 0.001
        
        # if abs(pitch) > 29 or abs(yaw) > 43.5:
        # target_rew[mask] = 0.0
        # target_rew = torch.zeros_like(target_rew, type=gs.tc_float)
        # target_rew = torch.abs(target_rew) * -0.01 # 鼓励原地不动，训练朝向
        
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        # TODO: 改成向量夹角并乘四次方
        # pitch = self.base_euler[:, 1] - vector_to_euler(self.rel_pos)[:,1]
        # pitch = torch.where(pitch > 180., pitch - 360., pitch)
        # pitch = torch.where(pitch < -180., pitch + 360., pitch)
        # pitch = pitch / 180 * math.pi * 0.5 # / 180 * math.pi * 0.5
        
        # yaw = self.base_euler[:, 2] - vector_to_euler(self.rel_pos)[:,2]
        # yaw = torch.where(yaw > 180., yaw - 360., yaw) 
        # yaw = torch.where(yaw < -180., yaw + 360., yaw)
        # yaw = yaw / 180 * math.pi * 0.5 # / 180 * math.pi * 0.5  # use rad for yaw_reward
        
        # _, angles = calc_axis_angle(self.base_quat, self.rel_pos)
        rew_yaw = torch.exp(self.reward_cfg["yaw_lambda"] * (torch.abs(yaw)**4))
        # rew_pitch = torch.exp(self.reward_cfg["yaw_lambda"] * (torch.abs(pitch)**2)) * 0.0
        rew = rew_yaw 
        # rew = torch.exp(self.reward_cfg["yaw_lambda"] * (torch.abs(yaw) + torch.abs(pitch)))
        # rew = torch.exp(self.reward_cfg["yaw_lambda"] * (torch.abs(angles)**4))
        # rew = rew - 0.2
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
    
    def _reward_offset(self):
        toward_target_vel = torch.nn.functional.normalize(self.rel_pos, dim=1)
        cur_vel_vec = torch.nn.functional.normalize(self.base_lin_vel, dim=1)
        offset_rew = torch.norm(toward_target_vel - cur_vel_vec, dim=1)
        return offset_rew
