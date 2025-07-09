''' 
[Debug] Generate random obstacles between poses 
- 在多个位置点之间的水平内切圆区域内生成随机障碍物，并返回障碍物信息矩阵
- [Inputs] env_cfg
    - 指定起终点位置列表 [序列点列表]
    - 额外限制点位置列表 [无序列点列表，需做可行性检验]
    - 障碍物配置 [障碍物类型，随机尺寸范围，分布密度函数]
- [Outputs]
    - 障碍物信息矩阵 [位置，尺寸]
- [API] 待补充
'''

import argparse
import time
import random
import numpy as np
import torch
import genesis as gs
from shapely.geometry import Polygon, Point, MultiPoint
from scipy.spatial import ConvexHull
from utils import *

class ObstacleManager:
    # 不基于 scene 的平面障碍物管理器
    def __init__(self, 
                 obst_cfg_dict : dict = None, 
                 cntr_pts_list : list = [], 
                 default_obst_space : list = [0.0, 0.0, 5.0],
                 dense_func : str = "gaussian",
                 is_sampled : bool = True, # 初始化障碍物，后期采样生成随机位置，使用scene时默认为 True
                 debug_mode : bool = False,
                 torch_device = "cuda" # 外部判断改 cuda
                 ):
        # 障碍物基础配置, 默认为 dict 类型
        self.obst_cfg = obst_cfg_dict
        self.pts_list = cntr_pts_list
        
        if len(self.pts_list) < 2:
            if self.debug_mode:
                # automatically debug
                self.pts_list = ObstacleManager.generate_debug_pts_list(extr_pts_num=0, init_mode=True)
                self.traj_pts = self.pts_list[:2]
                self.extr_pts = self.pts_list[2:]
            else:
                # using default obst space constrained by pts_list
                self.traj_pts = self.pts_list
                self.extr_pts = []
        else:        
            self.traj_pts = self.pts_list[:2]
            self.extr_pts = self.pts_list[2:]
        
        self.default_obst_space = self.obst_cfg.get("default_obst_space",default_obst_space) # [x, y, r]
        self.obst_space = self.default_obst_space
        self.dense_func = self.obst_cfg.get("dense_func",dense_func)
        self.debug_mode = self.obst_cfg.get("debug_mode",debug_mode)
        self.is_sampled = is_sampled
        self.device = torch_device 
        if self.device == "cuda" and not torch.cuda.is_available():
            print("cuda device not available, using cpu instead")
            self.device = "cpu"
        
        # self.obstacles_info = {} # 初始化tensor,用下面这个函数完成
        self.obst_num = self.obst_cfg.get("obst_max_num", 10)
        self.obstacles_info = torch.zeros((self.obst_num, 6), device=self.device, dtype=torch.float)
        self.reset_obstacles_info()
        # self.init_flag = True 改用 self.obstacles_info 来判断
    @ staticmethod
    def generate_debug_pts_list(has_traj_pts=True, extr_pts_num=0, default_region=[0.0, 0.0, 10.0], default_min_dist=1.0, init_mode=True, intr_pt = [0.0, 0.0]):
        cx, cy, cr = default_region  # 圆心 x, y 和半径
        if init_mode: # only for debug
            points = []
        else:
            # 如果并非初始化模式，则使用传入的中继点作为新一轮debug点的初始点
            points = intr_pt

        # Step 1: 如果有轨迹点，先生成两个端点
        if has_traj_pts:
            attempts = 0
            max_attempts = 500
            while len(points) < 2 and attempts < max_attempts:
                attempts += 1
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, cr * 0.8)  # 留出边界空间
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
                pt = [x, y]

                # 检查与已有点的距离
                too_close = any(np.linalg.norm([x - px, y - py]) < default_min_dist for (px, py, *_) in points)
                if not too_close:
                    points.append(pt)

            if len(points) < 2:
                raise ValueError("无法生成两个满足最小距离要求的轨迹点")

        # Step 2: 如果需要 bound pts，生成 1~3 个额外点
        if extr_pts_num:
            attempts = 0
            max_attempts = 10000
            while len(points) < 2 + extr_pts_num and attempts < max_attempts:
                attempts += 1
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, cr * 0.9)
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
                pt = [x, y]

                # 检查与已有点的距离
                too_close = any(np.linalg.norm([x - px, y - py]) < default_min_dist for (px, py, *_) in points)
                if not too_close:
                    points.append(pt)

            if len(points) < 2 + extr_pts_num:
                print(f"警告：仅生成 {len(points) - 2} / {extr_pts_num} 个额外约束点")

        return points
    
    def init_obst_space(self, type="circle"):
        if type == "circle":
            # Step 0: 确保至少有两个轨迹点
            if len(self.traj_pts) < 2:
                raise ValueError("至少需要两个轨迹点来构建直径圆")
            
            # 两点之间取直线中点构建内切圆
            p1, p2 = np.array(self.traj_pts[0]), np.array(self.traj_pts[1])
            center = (p1 + p2) / 2
            radius = np.linalg.norm(p1 - p2) / 2
            safe_radius = radius - self.cntr_cfg["colli_safe_dist"]
                        
            extr_pts_in_circle = []
            for pt in self.extr_pts:
                dist_to_center = np.linalg.norm(np.array(pt) - center)
                if dist_to_center < safe_radius:
                    extr_pts_in_circle.append(pt)
            
            if not extr_pts_in_circle:
                self.obst_space = [center[0], center[1], safe_radius]
            else:
                points = np.vstack((np.array(self.traj_pts), np.array(extr_pts_in_circle)))

                # Step 1: 构建凸包
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                polygon = Polygon(hull_points)

                # Step 2: 生成凸包内的网格点
                x_coords = hull_points[:, 0]
                y_coords = hull_points[:, 1]
                min_x, max_x = x_coords.min(), x_coords.max()
                min_y, max_y = y_coords.min(), y_coords.max()

                grid_size = 0.05  # 网格精度
                x_range = np.arange(min_x, max_x, grid_size)
                y_range = np.arange(min_y, max_y, grid_size)
                max_radius = 0.0
                best_center = None

                # Step 3: 遍历网格点，寻找离边界最远的点
                for x in x_range:
                    for y in y_range:
                        pt = Point(x, y)
                        if polygon.contains(pt):
                            # 到最近边界的距离
                            dist = polygon.exterior.distance(pt)
                            if dist > max_radius:
                                max_radius = dist
                                best_center = (x, y)

                if best_center and max_radius > 0:
                    print(f"最大内切圆: 圆心 {best_center}, 半径 {max_radius}")
                    self.obst_space = [best_center[0], best_center[1], max_radius - self.cntr_cfg["colli_safe_dist"]]
                else:
                    raise ValueError("无法找到给定点集凸包内的最大内切圆")
            
        elif type == "polygon":
            raise ValueError("Yet To Be Developed...")
    
    def reset_obstacles_info(self, is_generate = True, type=ObstacleType.CYLINDER):
        # 初始化障碍物，提供采样矩阵
        obst_num = self.obst_cfg.get("obst_max_num", 10)
        if not torch.allclose(self.obstacles_info, torch.zeros_like(self.obstacles_info)):
            # 调用reset, 先清空 obstacle_info, 默认为空的Tensor, 相当于init
            # 更新 obstacle_info 为 [r1, r2, h, x, y, z]， 其中对于cylinder r1=r2=r, 对于 box r1=l, r2=w
            self.obstacles_info = torch.zeros((obst_num, 6), device=self.device, dtype=torch.float)
        
        if is_generate:    
            for i in range(obst_num):
                r_obst = random.uniform(*self.obst_cfg["obst_range_radius"])
                h_obst = random.uniform(*self.obst_cfg["obst_range_height"])
                if type == ObstacleType.BOX:
                    l_obst = random.uniform(*self.obst_cfg["obst_range_radius"])
                elif type == ObstacleType.CYLINDER:
                    l_obst = r_obst
                # 用torch 来表达矩阵佳矣
                self.obstacles_info[i, :] = torch.tensor([r_obst, l_obst, h_obst, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float)

    def randomly_sample_obst_with_poses(self, is_overlap=True):
        # 初始化随机生成障碍物 [位置]
        if not torch.allclose(self.obstacles_info, torch.zeros_like(self.obstacles_info)):
            cx, cy, cr = self.default_obst_space  # 圆心 x, y 和半径
        else:
            cx, cy, cr = self.obst_space  # 圆心 x, y 和半径
        num_obstacles = self.obst_cfg["obstacle_max_num"]
        min_spacing = self.obst_cfg.get("obstacle_min_spacing", 0.2)

        if cr < min_spacing:
            print(f"提示：无障碍物生成")
            return 
        
        # 安全距离（障碍物不能紧贴边界）
        safe_radius = cr * 0.95
        positions = []
        attempts = 0
        max_attempts = 1000  # 防止无限循环

        while len(positions) < num_obstacles and attempts < max_attempts:
            attempts += 1

            # 生成位置
            if self.dense_func == "uniform":
                # 均匀分布采样
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, safe_radius)
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
            elif self.dense_func == "gaussian":
                # 高斯分布采样（靠近中心）
                angle = random.uniform(0, 2 * np.pi)
                radius = abs(np.random.normal(0, safe_radius / 2))
                radius = min(radius, safe_radius)  # 确保不超过边界
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
            else:
                raise ValueError(f"不支持的密度函数: {self.dense_func}")

            # 检查与已有障碍物的距离是否足够
            if is_overlap:
                too_close = any(
                    np.linalg.norm([x - px, y - py]) < self.obst_cfg["obstacle_range_radius"][1] for (px, py, _, _) in positions
                )
                dynamic_range = self.obst_cfg["obstacle_range_radius"]
            else:
                dist_list = [np.linalg.norm([x - px, y - py]) - (min_spacing + pr) for (px, py, pr, _) in positions]
                too_close = any( dist_list < self.obst_cfg["obstacle_range_radius"][0])
                dynamic_range = [self.obst_cfg["obstacle_range_radius"][0], min(self.obst_cfg["obstacle_range_radius"][1], min(dist_list))]

            if not too_close:
                # 生成随机半径和高度 TODO: 改为采样
                if not self.obstacles_info:
                    # 初始化阶段
                    r_obst = random.uniform(*dynamic_range)
                    h_obst = random.uniform(*self.obst_cfg["obstacle_range_height"])
                else:
                    # 运行阶段, 半径和高度不可更改，进入obstacles_info采样模式
                    sampled_obstacle = random.choice(list(self.obstacles_info.values()))
                    r_obst, h_obst = sampled_obstacle[2], sampled_obstacle[3]
                    # 可选：再次验证该障碍物是否仍符合当前环境条件
                    if r_obst > dynamic_range[1]:  # 如果不符合当前动态范围则跳过
                        continue
                
                # 确保障碍物完全在圆内
                dist_from_center = np.linalg.norm([x - cx, y - cy])
                if dist_from_center + r_obst <= safe_radius:
                    positions.append((x, y, r_obst, h_obst))

        if len(positions) < num_obstacles:
            print(f"提示：仅成功生成 {len(positions)} / {num_obstacles} 个障碍物")

        # 存储障碍物信息：index -> [x, y, r, h]
        for i, pos in enumerate(positions):
            self.obstacles_info[i] = list(pos)
        
        return self.obstacles_info


class ObstacleSceneManager(ObstacleManager):
    # 障碍物场景管理器，继承上述不依赖scene的障碍物管理器
    def __init__(self, 
                 scene : gs.Scene,
                 obst_cfg_dict = None, 
                 cntr_pts_list = [], 
                 default_obst_space=[0.0, 0.0, 5.0],
                 dense_func = "gaussian",
                 debug_mode = False,
                 ):
        super().__init__(
            obst_cfg_dict = obst_cfg_dict,
            cntr_pts_list = cntr_pts_list,
            default_obst_space = default_obst_space,
            dense_func = dense_func,
            debug_mode = debug_mode,
        )
        self.scene = scene

def initialize_debug_poses(scene: gs.Scene, pts_list, scale = 0.05, color_traj = (0.5, 1.0, 0.5), color_extr = (0.5, 0.5, 0.5)):
    pos_entities = []
    for idx, pt in enumerate(pts_list):
        if idx<2:
            color = color_traj
        else:
            color = color_extr
        sphere = scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/sphere.obj",
                pos=[pt[0], pt[1], 1.0],  # z=0 默认值，可修改
                scale=scale,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=color)
            )
        )
    pos_entities.append(sphere)
    return pos_entities
def initialize_obstacle_entities(scene: gs.Scene, obst_info, base_z = 0.0):
    obstacles = []
    for idx, (x, y, r, h) in obst_info.items():
        cylinder = scene.add_entity(
            gs.morphs.Cylinder(
                radius=r,
                height=h,
                pos=[x, y, base_z + h / 2],  # 放置在地面上
                quat=euler2quat(0.0, 0.0, 0.0),
                fixed=True,
                collision=False,
            )
        )
        obstacles.append(cylinder)
    return obstacles

def visualize_debug_poses(scene: gs.Scene, pts_list, scale = 0.05, color_traj = (0.5, 1.0, 0.5), color_extr = (0.5, 0.5, 0.5)):
    pos_entities = []
    for idx, pt in enumerate(pts_list):
        if idx<2:
            color = color_traj
        else:
            color = color_extr
        sphere = scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/sphere.obj",
                pos=[pt[0], pt[1], 1.0],  # z=0 默认值，可修改
                scale=scale,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=color)
            )
        )
    pos_entities.append(sphere)
    return pos_entities
def visualize_obstacle_entities(scene: gs.Scene, obst_info, base_z = 0.0):
    obstacles = []
    for idx, (x, y, r, h) in obst_info.items():
        cylinder = scene.add_entity(
            gs.morphs.Cylinder(
                radius=r,
                height=h,
                pos=[x, y, base_z + h / 2],  # 放置在地面上
                quat=euler2quat(0.0, 0.0, 0.0),
                fixed=True,
                collision=False,
            )
        )
        obstacles.append(cylinder)
    return obstacles

def visualize_obst_circle(scene: gs.Scene, center, radius, color=(1, 0, 0), num_segments=64):
    """
    在 scene 中绘制一个圆用于可视化边界区域
    :param scene: gs.Scene 实例
    :param center: 圆心 (cx, cy)
    :param radius: 半径 r
    :param color: 颜色 (r, g, b)
    :param num_segments: 圆周分段数
    """
    cx, cy = center
    points = []
    for i in range(num_segments + 1):
        angle = 2 * np.pi * i / num_segments
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        points.append([x, y, 0.0])  # z=0 平面

    # 绘制闭合圆圈
    for i in range(num_segments):
        scene.draw_debug_line(points[i], points[i + 1], color=color)

def form_obstacles_with_TreeMap(env_cfg, obstacles_info):
    obstacles_res = {}  # index : pos_x, pos_y, pos_z
    start_p = [0, 0]    # 栅格坐标
    
    node_map = TreeMap(TreeNode(-1, start_p))
    node_map.calc_leaves(node_map.root) # 计算邻接节点地图
    cur_node = node_map.root
    cur_node_idx = cur_node.idx
    cur_node.calc_occupies()
    o_list, c_list = [p for p in cur_node.occupies], [] # 待探索，已探索
    #hist_xy_list = [[0.2, 0.2, 0.0, 0.0]] # 原点安全距离 size_x, size_y, pos_x, pos_y
    
    # 灵活计算水平边界范围为最大边长之两倍
    step_l = 4 * max(env_cfg["obstacle_min_spacing"], max(env_cfg["obstacle_size_range_x"]), max(env_cfg["obstacle_size_range_y"]))
    
    for index, obstacle_info in obstacles_info.items():
        
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
                if dist < env_cfg["obstacle_min_spacing"]+0.5*max(max(env_cfg["obstacle_size_range_x"]), max(env_cfg["obstacle_size_range_y"])):
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

    # Debug Code for Generating Obstacle between two poses
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, backend=gs.cpu if args.cpu else gs.gpu)

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
    
    debug_obst_cfg = {
        # obstacle settings
        "obstacles": True,
        "obst_type": [1], # Detailed in class ObstacleType
        "obst_max_num": 10,
        "obst_pos_fixed": False,
        "default_obst_space": [0.0, 0.0, 5.0], # [x, y, r] default, for UAV tasks needs to be modified
        "obst_range_radius": [0.05, 0.3], # for Box sample 2 times to born XY
        "obst_range_height": [1.5, 3.5],
    }
    
    debug_cntr_cfg = {
        # assignment
        "debug_pos_min_dist": 0.5,  # default between 2 poses
        "hover_safe_dist": 0.2,     # default to RACER.urdf
        "colli_safe_dist": 0.2,
    }
    
    obst_manager = ObstacleManager([], [], debug_obst_cfg, debug_cntr_cfg, dense_func="uniform")
    
    pts_list = obst_manager.pts_list
    obst_manager.generate_obst_space(type="circle")
    obst_info = obst_manager.generate_obstacles(type=ObstacleType.CYLINDER)
    
    ########################## entities ##########################
    # add plane
    plane = scene.add_entity(gs.morphs.Plane())
    
    
    
    # for debug
    # initialize debug poses
    pose_entities = visualize_debug_poses(scene, pts_list)
    
    # initialize obstacles
    obstacle_entities = visualize_obstacle_entities(scene, obst_info)

    ########################## build ##########################
    scene.build(n_envs=1)
    
    # 可视化内切圆边界
    cx, cy, cr = obst_manager.obst_space
    visualize_obst_circle(scene, (cx, cy), cr, color=(1, 0, 0))  # 红色圆表示边界
    
    last_update_time = time.time()
    
    while True:
        current_time = time.time()
        
        if current_time - last_update_time >= 10.0:
            print("🔄 正在更新 poses 和障碍物...")
            
        
        scene.step()
            
    
    # for _ in range(1000):
    #     time.sleep(0.5)
    #     scene.step()


if __name__ == "__main__":
    main()
