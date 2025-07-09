import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import random
from typing import List, Tuple

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class BallDodgingEnv:
    def __init__(self, num_envs, env_cfg, drone_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.num_drone = drone_cfg["swarm_num_drone"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = env_cfg["sim_dt"]  # 0.01 = run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.drone_cfg = drone_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=1), # substeps =2
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
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

        # add target
        if self.env_cfg["visualize_target"]:
            self.targets = []
            for i in range(self.num_drone):
                self.targets.append(self.scene.add_entity(
                    morph=gs.morphs.Mesh(
                        file="meshes/sphere.obj",
                        scale=0.05,
                        fixed=False,
                        collision=False,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(
                            color=(1.0, 0.5, 0.5),
                        ),
                    ),
                ))
        else:
            self.targets = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone # TODO: 修改 init_pos
        
        # self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device).tile(self.num_drone, 1)
        self.base_init_pos = self._generate_random_positions(self.num_drone, 
                                                             self.env_cfg["base_init_pos"],
                                                             self.drone_cfg["swarm_init_offset"], 
                                                             self.drone_cfg["swarm_init_random_bias"])
        
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device).tile(self.num_drone, 1)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        
        self.drones = []
        for i in range(self.num_drone):
            self.drones.append(self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf")))
        
        # self.drone_test = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0.0, 0.0, 1.0)))

        # build scene
        self.scene.build(n_envs=num_envs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers # TODO: 修改 为多机
        self.obs_buf = torch.zeros((self.num_envs, self.num_drone, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs, self.num_drone), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_drone, self.num_commands), device=gs.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_drone, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, self.num_drone, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, self.num_drone, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, self.num_drone, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, self.num_drone, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        
        from quadcopter_controller import DronePIDController
        pid_params = [
            [8.0, 0.1, 2.5],
            [8.0, 0.1, 2.5],
            [16.0, 0.12, 8.0], # i=0.3
            [20.0, 0.8, 20.0],
            [20.0, 0.8, 20.0],
            [25.0, 1.6, 25.0], # i=0.5
            [10.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],
            [2.0, 0.0, 0.2],
        ]
        self.base_rpm = 14468.429183500699
        self.controller_test = DronePIDController(drone=self.drones[0], dt=0.01, base_rpm=self.base_rpm, pid_params=pid_params)
        
    @staticmethod
    def _generate_random_positions(
        num_random: int,
        origin: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        grid_size: float = 1.0,
        offset_range: float = 0.2
    ) -> torch.Tensor:
        """
        按网格生成多个随机点的位置。

        参数:
            num_random: 无人机数量
            origin: 生成区域的中心点坐标 (x, y, z)
            grid_size: 每个无人机所占的网格间距（单位米）
            offset_range: 在网格中心的最大随机偏移范围（±offset_range）

        返回:
            Tensor of 位置坐标 (x, y, z)
        """
        positions = torch.zeros((num_random, 3), device=gs.device)
        cols = int(num_random ** 0.5) + 1  # 每行列的数量（近似正方形布局）
        
        for i in range(num_random):
            row = i // cols
            col = i % cols

            # 计算对应的网格中心
            center_x = origin[0] + (col - cols // 2) * grid_size
            center_y = origin[1] + (row - cols // 2) * grid_size
            z = origin[2]

            # 添加随机偏移
            dx = random.uniform(-offset_range, offset_range)
            dy = random.uniform(-offset_range, offset_range)

            x = center_x + dx
            y = center_y + dy

            positions[i,:] = torch.tensor([x, y, z], device=gs.device)

        return positions
    
    def _resample_commands(self, envs_idx):
        # print(envs_idx)
        test = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),self.num_drone), gs.device)
        # print(test)
        # print(self.commands)
        self.commands[envs_idx,:, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),self.num_drone), gs.device)
        self.commands[envs_idx,:, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),self.num_drone), gs.device)
        self.commands[envs_idx,:, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),self.num_drone), gs.device)

    def _at_target(self):
        at_target = (
            (torch.norm(self.rel_pos, dim=2) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).flatten()
        ) # 修改dim使计算正确的距离
        return at_target

    def step(self, actions):
        if abs(actions.mean()) / self.drone_cfg["base_rpm"] > 0.9:
            prop_rpms = actions
        else:
            self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
            exec_actions = self.actions
            base, factor = sum(self.drone_cfg["rpm_scale"]) / 2, (max(self.drone_cfg["rpm_scale"]) - min(self.drone_cfg["rpm_scale"]))/2
            prop_rpms = (base + exec_actions * factor) * self.drone_cfg["base_rpm"]

        # 测试
        target_test = (self.commands[0,0]).cpu().tolist() 
        prop_rpms_test = self.controller_test.update(target_test)
        
        min_rpm = 0.5 * self.base_rpm
        max_rpm = 1.5 * self.base_rpm      
          
        prop_rpms_test = torch.clamp(prop_rpms_test, min_rpm, max_rpm)
        
        # 14468 is hover rpm
        # self.drones[0].set_propellels_rpm(prop_rpms_test.tile(self.num_envs,1))
        self.drones[0].set_propellels_rpm(prop_rpms[:, 0])
        
        # for idx, drone in enumerate(self.drones):
        #     # drone.set_propellels_rpm(prop_rpms[:, idx, :]) 
        #     drone.set_propellels_rpm(prop_rpms[:, idx]) # 
        # update target pos
        if self.targets is not None:
            for idx, target in enumerate(self.targets):
                target.set_pos(self.commands[:, idx, :], zero_velocity=True, envs_idx=list(range(self.num_envs)))
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        for idx, drone in enumerate(self.drones):
            self.base_pos[:, idx] = drone.get_pos()
            self.base_quat[:, idx] = drone.get_quat()
        
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        for idx, drone in enumerate(self.drones):
            # self.base_lin_vel[:, idx] = transform_by_quat(drone.get_vel(), inv_base_quat[:, idx]) # 修改了坐标系 RL 中需要使用
            self.base_lin_vel[:, idx] = drone.get_vel()
            self.base_ang_vel[:, idx] = transform_by_quat(drone.get_ang(), inv_base_quat[:, idx])

        # resample commands
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        # check termination and reset
        self.crash_condition = torch.zeros((self.num_envs,self.num_drone), device=gs.device, dtype=torch.bool)
        
        self.crash_condition = (
            (torch.abs(self.base_euler[:,:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:,:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:,:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:,:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:,:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:,:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        self.crash_condition_all = torch.any(self.crash_condition, dim=1) # 取或值：只有环境中所有智能体失效时才reset
        
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition_all

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward 待修改 # 非训练时不计算reward
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
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset base # 此处待修改为多机 reset
        self._resample_commands(envs_idx) # 先给定目标点以保证 pid setup
        
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.unsqueeze(0)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        for idx, drone in enumerate(self.drones):
            drone.set_pos(self.base_pos[envs_idx, idx], zero_velocity=True, envs_idx=envs_idx)
            drone.set_quat(self.base_quat[envs_idx, idx], zero_velocity=True, envs_idx=envs_idx)
            drone.zero_all_dofs_velocity(envs_idx)

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

        # TODO: 待改逻辑
        #self._resample_commands(envs_idx)
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
        

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    # 待改之
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
