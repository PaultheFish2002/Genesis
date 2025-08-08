import torch
import argparse
import time
import random
import math
import numpy as np
import genesis as gs
import copy
from shapely.geometry import Polygon, Point, MultiPoint
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from obstacle_manager import ObstacleManager, ObstManagerInScene
from utils import *

# Perception Env with Step
class DronePerceptionTest:
    def __init__(self, scene: gs.Scene, drone: gs.morphs.Drone, perc_cfg: dict):
        # support Parallel scene
        self.scene = scene
        self.drone = drone
        self.device = gs.device
        
        self.perc_cfg = perc_cfg
        self.perc_range = self.perc_cfg.get("perc_range", 3.0)
        
        # obst_info 对应的 idx
        self.perc_obst_idx = torch.zeros((self.scene.n_envs, self.perc_cfg.get("obst_num", 10)), device=self.device, dtype=gs.tc_int)
        
    def update_cfg(self, assign_region: list = None):
        if assign_region is not None:
            self.assignment_region = assign_region
    
    def perc_obst_in_range(self, drone_pos: torch.Tensor, obst_info: torch.Tensor):
        # 观测范围内所有的 obst 对象，且按照距离矩阵排序， return obst_info 中观测到的索引值
        # drone_pos vector2d [num_ebvs [x,y,z]] 
        # obst_info vector2d [num_obst [idx, l, r, h, x, y, z, qw, qx, qy, qz]]
        
        drone_posxy = drone_pos[:, :2]
        obst_posxy = obst_info[:, 4:6]
        obst_radius = obst_info[:, 2]
        
        dist_mat = torch.linalg.norm(
            drone_posxy.unsqueeze(1) - obst_posxy.unsqueeze(0), 
            dim=2
        )
        # concerning radius
        perc_mask = dist_mat - obst_radius.unsqueeze(0) <= self.perc_range
        
        # 此时如果直接取idx，则每个env中观测到障碍物数量不一致，需要平齐处理 [num_envs, num_obst] -> [idx1, idx2, ..., idxf, -1, ..., -1]
        num_envs, num_obst = perc_mask.shape
        perc_obst_idx = torch.full((num_envs, num_obst), -1, device=perc_mask.device, dtype=torch.long)
        
        for i in range(num_envs):
            in_range = torch.nonzero(perc_mask[i], as_tuple=False).flatten()
            
            if in_range.numel() > 0:
                # 增加排序 & Copy the indices, leaving the rest as -1
                in_range_distmat = dist_mat[i, in_range]
                sorted_idx = torch.argsort(in_range_distmat)
                sorted_in_range = in_range[sorted_idx]
                perc_obst_idx[i, :sorted_in_range.numel()] = sorted_in_range

        return perc_obst_idx
    def perc_obst_in_sight(self, drone_pos: torch.Tensor, obst_info: torch.Tensor):
        # 带obst视角遮盖的方法，用于进一步筛选视线范围内障碍物，为后续深度估计做准备
        
        pass
    
    def calc_observation_mat(self):
        pass

class DebugEnv:
    # Now for Perception Debugging
    # Later includes Controller Debugging 
    def __init__(self,
                 num_envs,
                 env_cfg,
                 scene_cfg,
                 show_viewer=False
                 ):
        self.num_envs = num_envs
        self.rendered_env_num = min(9, self.num_envs)
        self.env_cfg = env_cfg
        self.scene_cfg = scene_cfg
        
        # env cfg to make scene
        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        
        # other cfg variables
        # self.assign_region = self.scene_cfg.get("default_obst_region", [0.0, 0.0, 5.0])
        self.takeoff_prot_region = self.scene_cfg.get("takeoff_protection_region", [0.0, 0.0, 0.5])
        
        robot_safe_r = self.scene_cfg.get("robot_colli_safe_r", 0.25)
        sparse_f = self.scene_cfg.get("feasible_inflation_factor", 3.0)
        self.reserve_dist = robot_safe_r * sparse_f
        
        self.at_target_threshold = self.scene_cfg.get("at_target_threshold", 0.1)
        
        self.device = gs.device
        
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
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
            self.target = self.scene.add_entity(
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
        
        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))
        
        # add obstacles
        self.obst_manager = ObstManagerInScene(scene=self.scene, obst_cfg_dict=scene_cfg)
        self.obst_manager.render_obst_entities()
        
        self.assign_region = self.obst_manager.obst_region
        
        # build scene
        self.scene.build(n_envs=num_envs, env_spacing=(20.0, 20.0),)
        
        # initialize buffers
        self.target_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float) # the commands
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        
        # target timeout flag
        self.target_timeout_length = math.ceil(self.env_cfg.get("target_timeout_s", 3.0) / self.dt)
        self.target_timeout = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.at_target_flag = torch.zeros_like(self.target_timeout)
        
        self.episode_sums = dict()
        self.extras = dict()  # extra information for logging
        
    def step(self):
        # TODO:drone controller response, for deubg directly set pos is enough
        debug_idx = torch.arange(self.num_envs, device=gs.device)
        self.drone.set_pos(self.target_pos[debug_idx], zero_velocity=True, envs_idx=debug_idx)
        
        # update target pos
        if self.target is not None:
            self.target.set_pos(self.target_pos, zero_velocity=True)
        
        # update scene
        self.scene.step()
        
        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.target_pos - self.base_pos
        self.last_rel_pos = self.target_pos - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            ),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # resample commands
        at_target = self._at_target_idx()
        envs_idx = self._at_target_timeout(at_target)
        self._resample_target_pos(envs_idx)
        
        # termination 
        self.reset_buf = (self.episode_length_buf > self.max_episode_length)
        
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

    def _at_target_idx(self):
        # debug with hovering time delay
        at_target = (torch.norm(self.rel_pos, dim=1) < self.at_target_threshold).nonzero(as_tuple=False).reshape((-1,))
        new_at_target_mask = (self.at_target_flag[at_target] == 0).nonzero(as_tuple=False).flatten()
        self.at_target_flag[at_target[new_at_target_mask]] = 1
        return at_target
    
    def _at_target_timeout(self, at_target):
        # once_at_target_mask = (self.at_target_flag == 1).nonzero(as_tuple=False).flatten()
        # self.target_timeout[once_at_target_mask] += 1
        self.target_timeout += 1 # 所有目标点定时保留，与是否到达目标点无关
        at_target_timeout = (self.target_timeout >= self.target_timeout_length).nonzero(as_tuple=False).flatten()
        self.target_timeout[at_target_timeout] = 0
        return at_target_timeout
    
    def _resample_target_pos(self, envs_idx, include_takeoff_r = True):
        # select a random pos outside obst region (circle) or within takeoff protection region
        tpr_x, tpr_y, tpr_r = self.takeoff_prot_region
        ass_x, ass_y, ass_r = self.assign_region
        
        # considering parallel scene, only get at_target envs_idx
        cur_x = self.base_pos[envs_idx, 0].reshape(-1,1)
        cur_y = self.base_pos[envs_idx, 1].reshape(-1,1)
        base_z = self.base_init_pos[2]
        
        # Check if drones are in takeoff protection region, idx like torch.Tensor([id1, idn, ..., idm])
        # in_tpr_idx = (torch.linalg.norm(torch.stack([cur_x - tpr_x, cur_y - tpr_y]), dim=0) < tpr_r).squeeze(1).nonzero(as_tuple=False).reshape((-1,))
        tpr_mask = torch.norm(torch.cat([cur_x - tpr_x, cur_y - tpr_y], dim=1), dim=1) < tpr_r
        in_tpr_idx = tpr_mask.nonzero(as_tuple=False).reshape((-1,))
        out_tpr_idx = (~tpr_mask).nonzero(as_tuple=False).reshape((-1,))
        
        empty_buffer = torch.empty([envs_idx.shape[0], 1]).to(gs.device)
        # in takeoff region, set target to be outside obst region circle
        # angle_in_tpr = torch.rand([in_tpr_idx.shape[0], 1]).to(gs.device) * 2 * torch.pi
        angle = copy.deepcopy(empty_buffer.uniform_(0, 2 * torch.pi))
        pos_z = copy.deepcopy(empty_buffer.uniform_(- base_z / 2, base_z / 2))
        radius = ass_r + self.reserve_dist
        
        if in_tpr_idx.shape[0]:
            self.target_pos[in_tpr_idx, 0] = (ass_x + radius * torch.cos(angle[in_tpr_idx])).reshape((-1,))
            self.target_pos[in_tpr_idx, 1] = (ass_y + radius * torch.sin(angle[in_tpr_idx])).reshape((-1,))
            self.target_pos[in_tpr_idx, 2] = (base_z + pos_z[in_tpr_idx]).reshape((-1,))
        
        # out of takeoff region, set target either in takeoff region or in the opposite direction
        if out_tpr_idx.shape[0]:
            if include_takeoff_r:
                radius = random.uniform(0, min(tpr_r - self.reserve_dist, 0))
                
                self.target_pos[out_tpr_idx, 0] = (tpr_x + radius * torch.cos(angle[out_tpr_idx])).reshape((-1,))
                self.target_pos[out_tpr_idx, 1] = (tpr_y + radius * torch.sin(angle[out_tpr_idx])).reshape((-1,))
                self.target_pos[out_tpr_idx, 2] = (base_z + pos_z[out_tpr_idx]).reshape((-1,))
            
            else:
                # Select pos in opposite semicircle
                # Calculate angle from assignment region center to current position
                angle_to_cur = torch.atan2(cur_y[out_tpr_idx] - ass_y, cur_x[out_tpr_idx] - ass_x)
                # Sample from the opposite semicircle (±π/2 from the opposite direction)
                opposite_angle = angle_to_cur + torch.pi
                min_angle = opposite_angle - torch.pi/2
                max_angle = opposite_angle + torch.pi/2
                sample_angle = torch.empty([out_tpr_idx.shape[0], 1]).to(gs.device).uniform_(min_angle, max_angle)
                
                self.target_pos[out_tpr_idx, 0] = (ass_x + radius * torch.cos(sample_angle)).reshape((-1,))
                self.target_pos[out_tpr_idx, 1] = (ass_y + radius * torch.sin(sample_angle)).reshape((-1,))
                self.target_pos[out_tpr_idx, 2] = (base_z + pos_z[out_tpr_idx]).reshape((-1,))
        
        # if torch.linalg.norm(torch.tensor([cur_x - tpr_x, cur_y - tpr_y])) < tpr_r:
        #     angle = random.uniform(0, 2 * math.pi)
        #     radius = ass_r + self.reserve_dist
            
        #     set_x = ass_x + radius * math.cos(angle)
        #     set_y = ass_y + radius * math.sin(angle)
        #     set_z = base_z + random.uniform(- base_z / 2, base_z / 2)
        # else:
        #     if include_takeoff_r:
        #         angle = random.uniform(0, 2 * math.pi)
        #         radius = random.uniform(0, tpr_r - self.reserve_dist)
                
        #         set_x = tpr_x + radius * math.cos(angle)
        #         set_y = tpr_y + radius * math.sin(angle)
        #         set_z = base_z + random.uniform(- base_z / 2, base_z / 2)
        #     else:
        #         # Select pos in opposite semicircle
        #         # Calculate angle from assignment region center to current position
        #         angle_to_cur = math.atan2(cur_y - ass_y, cur_x - ass_x)
        #         # Sample from the opposite semicircle (±π/2 from the opposite direction)
        #         opposite_angle = angle_to_cur + math.pi
        #         min_angle = opposite_angle - math.pi/2
        #         max_angle = opposite_angle + math.pi/2
        #         sample_angle = random.uniform(min_angle, max_angle)
                
        #         # Sample radius within assignment region minus reserve distance
        #         radius = ass_r + self.reserve_dist
                
        #         set_x = ass_x + radius * math.cos(sample_angle)
        #         set_y = ass_y + radius * math.sin(sample_angle)
        #         set_z = base_z + random.uniform(-base_z / 2, base_z / 2)
        
        # return set_x, set_y, set_z    
        
    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.target_pos - self.base_pos
        self.last_rel_pos = self.target_pos - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        debug_obst_region = [0.0, 0.0, random.uniform(6.0, 10.0)]
        # debug_obst_region = [0.0, 0.0, 6.0]
        self.obst_manager.reset_obst(obst_region=debug_obst_region, obst_region_type="circle")
        
        self.obst_manager.draw_obst_region(color=(1, 0.5, 0.5, 0.5))
        self.obst_manager.render_obst_entities()
        
        self.assign_region = self.obst_manager.obst_region
        self._resample_target_pos(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-n", "--num_envs", type=int, default=9)
    parser.add_argument("-w", "--wrapped", action="store_true", default=True)
    args = parser.parse_args()
    
    ########################## init ##########################
    gs.init(seed=0, backend=gs.cpu if args.cpu else gs.gpu)

    env_cfg = {
        # base pose
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 5.0,
        "at_target_threshold": 0.1,
        "target_timeout_s": 1.0,
        # "resampling_time_s": 3.0,
        "simulate_action_latency": True,
        # visualization
        "visualize_target": True,
        "visualize_camera": False,
        "max_visualize_FPS": 100,
    }
    
    # obst scene cfg
    scene_cfg = {
        # robot settings
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "robot_colli_safe_r": 0.25,
        "takeoff_protection_region": [0.0, 0.0, 0.5], # takeoff protection, 改成保护半径，位置由 base pos 决定
        # perception settings
        "perc_range": 3.0, # r or [r] for cylinder perc mode, TODO: sphere perc mode
        "perc_obst_max_num": 10, # max num of obstacles percepted
        # interaction settings (debug)
        "at_target_thres": 0.2, # distance threshold to target"
        # obstacle settings
        "obst_enable": True,
        "obst_type": [1], # Detailed in class ObstacleType
        "obst_variety_num": 10, # obstacle shape variety
        "obst_region_type": "circle",
        "feasible_inflation_factor": 3.0, # 可行域膨胀系数，建议大于1.0，用于限制障碍物的数量和分布密度，该参数为robot_colli_safe_r的倍数，划定了智能体最小可行域的大小，grid_map_enable后决定 resolution,否则只决定数量和最小距离
        "obst_overlap":False,
        "grid_map_enable":False, 
        "default_obst_region": [0.0, 0.0, 6.0], # [x, y, r] default, for UAV tasks needs to be modified
        "obst_range_radius": [0.1, 0.5], # for Box sample 2 times to born XY
        "obst_range_height": [1.5, 3.0],
    }
    
    if args.wrapped:
        env = DebugEnv(num_envs=args.num_envs, 
                       env_cfg=env_cfg, 
                       scene_cfg=scene_cfg, 
                       show_viewer=True)
        
        env.reset()
        # max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])
        with torch.no_grad():
            # for _ in range(max_sim_step):
            while True:
                env.step()
    else:
        ########################## create a scene ##########################

        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-5.0, -5.0, 10.0),
                camera_lookat=(5.0, 5.0, 0.0),
                camera_fov=40,
            ),
            show_viewer=args.vis,
            rigid_options=gs.options.RigidOptions(
                dt=0.01,
                constraint_solver=gs.constraint_solver.Newton,
            ),
            vis_options=gs.options.VisOptions(
                # geom_type='sdf',
            ),
        )
        
        obst_manager = ObstManagerInScene(scene=scene, obst_cfg_dict=scene_cfg)
        
        # perception_test class, 最好还是独立出来
        # drone_perc = DronePerceptionTest(scene=scene, drone=drone, perc_cfg=scene_cfg)
        
        ########################## entities ##########################
        # add plane
        plane = scene.add_entity(gs.morphs.Plane())
        
        # add drone
        drone = scene.add_entity(
            gs.morphs.Drone(file="urdf/drones/racer.urdf")
        )
        
        # initialize obstacles
        obst_manager.render_obst_entities()
        
        ########################## build ##########################
        scene.build(n_envs=args.num_envs, env_spacing=(20.0, 20.0),)
        
        # build 之后设置
        drone.set_pos(torch.tensor(scene_cfg["base_init_pos"]).tile([args.num_envs, 1]), zero_velocity=True) # , envs_idx=torch.arange(args.num_envs)
        drone.set_quat(torch.tensor(scene_cfg["base_init_quat"]).tile([args.num_envs, 1]), zero_velocity=True)
        
        # 可视化边界
        obst_manager.draw_obst_region(color=(1, 0.5, 0.5, 0.5))
        
        last_update_time = time.time()
        
        while True:
            current_time = time.time()
            
            if current_time - last_update_time >= 5.0:
                
                debug_obst_region = [0.0, 0.0, random.uniform(6.0, 10.0)]
                obst_manager.reset_obst(obst_region=debug_obst_region, obst_region_type="circle")
                
                obst_manager.draw_obst_region(color=(1, 0.5, 0.5, 0.5))
                obst_manager.render_obst_entities()

                last_update_time = current_time
                print("🔄 正在更新 poses 和障碍物...")

            scene.step()

if __name__ == "__main__":
    main()
