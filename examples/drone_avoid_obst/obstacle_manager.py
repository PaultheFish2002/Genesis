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
    # 不基于 scene 的平面障碍物管理器, 输出 tensor 形式的障碍物信息
    def __init__(self, 
                 obst_cfg_dict : dict = None, 
                 cntr_pts_list : list = [[0.0, 0.0]], 
                 dense_func : str = "U",
                 debug_mode : bool = False,
                 torch_device = "cuda" # 外部判断改 cuda
                 ):
        
        # 障碍物基础配置, 默认为 dict 类型
        self.obst_cfg = obst_cfg_dict
        self.pts_list = cntr_pts_list
        
        self.dense_func = self.obst_cfg.get("dense_func",dense_func)
        self.debug_mode = self.obst_cfg.get("debug_mode",debug_mode)
        
        self.device = torch_device 
        if self.device == "cuda" and not torch.cuda.is_available():
            print("cuda device not available, using cpu instead")
            self.device = "cpu"
        
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
            # 默认 pts_list 中前两个点为 traj pts, 其余为 cntr pts, 该分类不影响无轨迹点的障碍物区域生成
            self.traj_pts = self.pts_list[:2]
            self.extr_pts = self.pts_list[2:]
        
        # 障碍物属性 [l, r, h] 及其类型， tensor
        self.obst_variety_num = self.obst_cfg.get("obst_variety_num", 10)
        self.obstacles_attr = torch.zeros((self.obst_variety_num, 3), device=self.device, dtype=torch.float)
        self.reset_obst_attr()
        
        self.default_obst_region = self.obst_cfg.get("default_obst_region", [0.0, 0.0, 5.0]) # [x, y, r]
        self.obst_region = self.default_obst_region
        self.obst_region_type = self.obst_cfg.get("obst_region_type",None) # 默认为none
        self.reset_obst_region()
            
        self.robot_safe_r = self.obst_cfg.get("robot_colli_safe_r", 0.2)
        self.sparse_f = self.obst_cfg.get("feasible_inflation_factor", 1.0)
        self.obst_min_r = self.obst_cfg.get("obst_range_radius", [0.05, 0.3])[0]
        self.obst_max_r = self.obst_cfg.get("obst_range_radius", [0.05, 0.3])[1]
        self.reset_obst_sampled() # 初始化预期障碍物数量
        
        self.takeoff_protection_region = self.obst_cfg.get("takeoff_protection_region", [0.0, 0.0, 0.5])
        
        # 初始化障碍物信息矩阵， 包含属性和坐标 [idx, l, r, h, x, y, z, qw, qx, qy, qz] 默认姿态
        self.obstacles_info = torch.zeros((self.obst_num, 11), device=self.device, dtype=torch.float)
        self.generate_obst_poses_InSample() # 默认采样点的方式生成障碍物位置
        
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
    
    @ staticmethod
    def calc_polygon_area(pts_xy_list): # 一系列平铺的 x,y 点
        if len(pts_xy_list) < 6:
            raise ValueError("顶点数量不足，无法构成多边形")

        # 转换为二维点列表
        points = [(pts_xy_list[i], pts_xy_list[i+1]) for i in range(0, len(pts_xy_list), 2)]
        
        # Shoelace formula
        area = 0.0
        n = len(points)
        for i in range(n):
            x_i, y_i = points[i]
            x_next, y_next = points[(i + 1) % n]  # 确保最后一个点与第一个点相连
            area += (x_i * y_next) - (x_next * y_i)

        return abs(area) / 2.0
    
    def reset_obst(self, 
                   reset_obst_attr: bool = False,   # 是否重置障碍物属性, scene 中默认不重置
                   reset_obst_sampled: bool = False,
                   obst_var_num: int = None, 
                   obst_region: list = None, 
                   robot_safe_r: float = None, 
                   sparse_f: float = None,
                   obst_range_r: list = None,
                   takeoff_prot_region: list = None):

        if reset_obst_attr:
            self.obst_variety_num = obst_var_num if obst_var_num is not None else self.obst_variety_num
            self.reset_obst_attr()
        if obst_region is not None:
            self.obst_region = obst_region
            self.reset_obst_region()
        
        self.robot_safe_r = robot_safe_r if robot_safe_r is not None else self.robot_safe_r
        self.sparse_f = sparse_f if sparse_f is not None else self.sparse_f
        if obst_range_r is not None:
            self.obst_min_r = obst_range_r[0]
            self.obst_max_r = obst_range_r[1]
        if reset_obst_sampled:
            self.reset_obst_sampled()
        
        self.takeoff_protection_region = takeoff_prot_region if takeoff_prot_region is not None else self.takeoff_protection_region
        # self.generate_obst_poses_InSample()

    def reset_obst_attr(self, type=ObstacleType.CYLINDER):
        # 初始化障碍物，提供采样矩阵
        obst_variety_num = self.obst_cfg.get("obst_variety_num", 10)
        if not torch.allclose(self.obstacles_attr, torch.zeros_like(self.obstacles_attr)):
            # 调用reset, 先清空 obstacle_info, 默认为空的Tensor, 相当于init
            # 此处为 obstales_attr 仅记录属性 [l, r, h], 其中对于cylinder r1=-1 r2=r, 对于 box r1=l, r2=w
            self.obstacles_attr = torch.zeros((obst_variety_num, 3), device=self.device, dtype=torch.float)
        
        # 之后,升级成异构也可行的障碍物生成管理器
        for i in range(obst_variety_num):
            r_obst = random.uniform(*self.obst_cfg["obst_range_radius"])
            h_obst = random.uniform(*self.obst_cfg["obst_range_height"])
            if type == ObstacleType.BOX:
                l_obst = random.uniform(*self.obst_cfg["obst_range_radius"])
            elif type == ObstacleType.CYLINDER:
                l_obst = -1
            # 用torch 来表达矩阵佳矣
            self.obstacles_attr[i, :] = torch.tensor([l_obst, r_obst, h_obst], device=self.device, dtype=torch.float)

    def reset_obst_region(self):
        # 生成障碍物平面区域初始化, 默认为 圆形 [cx, cy, r]
        if self.obst_region_type:
            type = self.obst_region_type
        if len(self.traj_pts) < 2:
            # 小于等于1个约束点，则使用默认区域
            if type == "circle": # [cx, cy, r]
                self.obst_region = self.default_obst_region
            elif type == "polygon": # [x1, y1, x2, y2, ...] # 默认生成 坐标轴正方形
                cx, cy, r = self.default_obst_region
                # 构造正方形四个顶点（顺时针）
                pt1 = (cx - r, cy + r)   # 左上
                pt2 = (cx + r, cy + r)   # 右上
                pt3 = (cx + r, cy - r)   # 右下
                pt4 = (cx - r, cy - r)   # 左下
                # 按顺序拼接成 list[float]
                self.obst_region = [pt1[0], pt1[1],
                                    pt2[0], pt2[1],
                                    pt3[0], pt3[1],
                                    pt4[0], pt4[1]]
        else:
            if type == "circle":
                # 两点之间取直线中点构建内切圆
                p1, p2 = np.array(self.traj_pts[0]), np.array(self.traj_pts[1])
                center = (p1 + p2) / 2
                radius = np.linalg.norm(p1 - p2) / 2
                colli_safe_r = self.obst_cfg.get(["robot_colli_safe_r"], 0.2)
                safe_radius = radius - colli_safe_r
                            
                extr_pts_in_circle = []
                for pt in self.extr_pts:
                    dist_to_center = np.linalg.norm(np.array(pt) - center)
                    if dist_to_center < safe_radius:
                        extr_pts_in_circle.append(pt)
                
                if not extr_pts_in_circle:
                    self.obst_region = [center[0], center[1], safe_radius]
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
                        self.obst_region = [best_center[0], best_center[1], max_radius - colli_safe_r]
                    else:
                        raise ValueError("无法找到给定点集凸包内的最大内切圆")
                
            elif type == "polygon":
                # Step 0: 确保至少有两个轨迹点， 仅两点时默认生成正方形
                if len(self.extr_pts) == 0:
                    p1, p2 = np.array(self.traj_pts[0]), np.array(self.traj_pts[1])
                    center = (p1 + p2) / 2
                    # radius = np.linalg.norm(p1 - p2) / 2
                    colli_safe_r = self.obst_cfg.get(["robot_colli_safe_r"], 0.2)
                    # safe_radius = radius - colli_safe_r
                    diff = p1 - center
                    # 单位向量和模长
                    norm_diff = np.linalg.norm(diff)
                    if norm_diff <= colli_safe_r:
                        raise ValueError("安全距离大于路径半径")
                    unit_diff = diff / norm_diff
                    shrinked_diff = unit_diff * (norm_diff - colli_safe_r)
                    
                    # 旋转 90 度（绕 z 轴）
                    def rotate_90(v):
                        return np.array([-v[1], v[0]])

                    diff_rotated = rotate_90(shrinked_diff)

                    # 四个顶点
                    pt1 = center + shrinked_diff
                    pt2 = center + diff_rotated
                    pt3 = center - shrinked_diff
                    pt4 = center - diff_rotated

                    # 按顺时针排列四个点
                    square_points = [pt1, pt2, pt3, pt4]

                    # 转换为浮点列表并展平
                    self.obst_region = list(np.round(np.concatenate(square_points), 4))
                else:
                # Step 1: 合并轨迹点与额外约束点
                    all_pts = np.vstack((np.array(self.traj_pts), np.array(self.extr_pts)))

                    # Step 2: 构建凸包
                    try:
                        hull = ConvexHull(all_pts)
                        poly_points = all_pts[hull.vertices]  # 按顺/逆时针顺序排列
                    except:
                        raise ValueError("无法构建凸包，请检查输入点是否共线或数量不足")

                    # Step 3: 设置 obst_region 为多边形顶点坐标列表
                    self.obst_region = list(np.round(poly_points.flatten(), 4))  # [x1, y1, x2, y2, ...]
                
                print(f"生成多边形区域顶点数: {len(poly_points)}, 坐标: {self.obst_region}")
    
    def reset_obst_sampled(self):
        # 根据 obst_region 面积和 sparse_f 计算需要的obs数量
        obst_max_domain_area = ((self.obst_max_r + self.sparse_f * self.robot_safe_r)*2) ** 2 # 计算方形面积，避免圆形面积引起障碍物数量虚高导致难收敛
        # 判定逻辑待改
        if self.obst_region_type == "circle":
            obst_region_area = torch.pi * self.obst_region[2] ** 2
        elif self.obst_region_type == "polygon":
            obst_region_area = ObstacleManager.calc_polygon_area(self.obst_region)
        # self.obst_num = int(obst_region_area / obst_max_domain_area)
        self.obst_num = int(obst_region_area / obst_max_domain_area) # 整除，得到最大数量
        # 仅随机采样一次，因build之后不再改变采样的obst个数和属性
        self.obst_var_idx = torch.randint(0, self.obstacles_attr.shape[0], (self.obst_num,), device=self.device)
    
    def generate_obst_poses_InSample(self, is_overlap=False):
        # 初始化随机生成所选障碍物的位置 [x, y, z]
        if not self.obst_region:
            cx, cy, cr = self.default_obst_region  # 圆心 x, y 和半径
        else:
            cx, cy, cr = self.obst_region  # 圆心 x, y 和半径 TODO: 此处还得增加对polygon的判定
            
        # 安全距离（障碍物不能紧贴边界）
        safe_radius = cr - self.obst_max_r
        obst_min_spacing = self.robot_safe_r * self.sparse_f
        
        if safe_radius < self.obst_max_r + self.robot_safe_r * self.sparse_f:
            print(f"提示：无障碍物生成")
            # obst_info
        else:
            attempts = 0
            max_attempts = 1000  # 防止无限循环
            
            overlap_attr = [] # temporary variety, for obst summary & overlap considerations [x,y,l,r]
            tpx, tpy, tpr  = self.takeoff_protection_region # protective safe region
            
            while len(overlap_attr) < self.obst_num and attempts < max_attempts:
                attempts += 1
                
                # 障碍物属性提取
                cur_obst_idx = len(overlap_attr)
                cur_obst_l = self.obstacles_attr[self.obst_var_idx[cur_obst_idx]][0]
                cur_obst_r = self.obstacles_attr[self.obst_var_idx[cur_obst_idx]][1]
                                
                # 生成位置, 默认在圆形中生成
                if self.dense_func == "uniform" or "U":
                    # 均匀分布采样
                    angle = random.uniform(0, 2 * np.pi)
                    radius = random.uniform(tpr + self.obst_max_r, safe_radius)
                    x = cx + radius * np.cos(angle)
                    y = cy + radius * np.sin(angle)
                elif self.dense_func == "gaussian" or "G":
                    # 高斯分布采样（待改，靠近圆环样条中心）
                    angle = random.uniform(0, 2 * np.pi)
                    radius = abs(np.random.normal(0, safe_radius / 2))
                    radius = min(radius, safe_radius)  # 确保不超过边界
                    x = cx + radius * np.cos(angle)
                    y = cy + radius * np.sin(angle)
                    # 检查是否在起飞保护区域内
                    dist_to_takeoff = np.linalg.norm([x - tpx, y - tpy])
                    if dist_to_takeoff < tpr + self.obst_max_r:  # 如果不在起飞保护区域，则接受该点
                        continue
                else:
                    raise ValueError(f"不支持的密度函数: {self.dense_func}")

                # 检查与已有障碍物的平面距离是否足够
                if is_overlap:
                    too_close = any(
                        np.linalg.norm([x - px, y - py]) < self.obst_max_r for (px, py, _, _) in overlap_attr
                    )
                else:
                    dist_list = [np.linalg.norm([x - px, y - py]) - (obst_min_spacing + max(pl, pr)) for (px, py, pl, pr) in overlap_attr]
                    too_close = any(torch.tensor(dist_list) < max(cur_obst_l, cur_obst_r))

                if not too_close:
                    overlap_attr.append((x, y, cur_obst_l, cur_obst_r))

            if len(overlap_attr) < self.obst_num:
                print(f"提示：仅成功生成 {len(overlap_attr)} / {self.obst_num} 个障碍物")

            # 存储障碍物信息：index -> [x, y, l, r]
            for i, attr in enumerate(overlap_attr):
                # [idx, l, r, h, x, y, z, qw, qx, qy, qz]
                self.obstacles_info[i][0] = self.obst_var_idx[i]
                self.obstacles_info[i][1:4] = self.obstacles_attr[self.obst_var_idx[i]]
                self.obstacles_info[i][4:6] = torch.tensor(attr[:2])
                self.obstacles_info[i][6] = math.floor((self.obstacles_info[i][3] / 2) * 10 ** 1 ) / (10 ** 1)# h/2 + 0.
                self.obstacles_info[i][7] = 1.0 # 默认原姿态
            
            return self.obstacles_info

    def generate_obst_poses_InGridMap(self):
        pass
    
    def save_obst_info(self, filetype='csv', filename='config/obstacle_info'):
        # as yaml or ini or csv
        pass
    
    def load_obst_info(self, filetype='csv', filename='config/obstacle_info'):
        # load from yaml or ini or csv
        pass
    @staticmethod # 预留
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
    

class ObstManagerInScene(ObstacleManager):
    # 障碍物场景管理器，继承上述不依赖scene的障碍物管理器
    def __init__(self, 
                 scene : gs.Scene,
                 obst_cfg_dict : dict = None, 
                 cntr_pts_list : list = [[0.0, 0.0]], 
                 dense_func : str = "uniform",
                 debug_mode : bool = False,
                 ):
        super().__init__(
            obst_cfg_dict = obst_cfg_dict,
            cntr_pts_list = cntr_pts_list,
            dense_func = dense_func,
            debug_mode = debug_mode,
            torch_device = gs.device # gs init 必然会初始化 device
        )
        self.scene = scene
        self.debug_pos_entities = []
        self.obst_entities = []
    def render_debug_poses(self, pts_list, scale = 0.05, color_traj = (0.5, 1.0, 0.5), color_extr = (0.5, 0.5, 0.5)):
        if not self.debug_pos_entities:
            for idx, pt in enumerate(pts_list):
                if idx<2:
                    color = color_traj
                else:
                    color = color_extr
                sphere = self.scene.add_entity(
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
            self.debug_pos_entities.append(sphere)
        else:
            pass
    
    def reset_obst(self, 
                   obst_region: list = None, # 此处需要保证格式符合要求
                   obst_region_type: str = None,
                   robot_safe_r: float = None, 
                   sparse_f: float = None,
                   obst_range_r: list = None,
                   takeoff_prot_region: list = None):

        if obst_region is not None and obst_region_type is not None:
            self.obst_region = obst_region
            self.obst_region_type = obst_region_type
            # self.reset_obst_region()
        
        self.robot_safe_r = robot_safe_r if robot_safe_r is not None else self.robot_safe_r
        self.sparse_f = sparse_f if sparse_f is not None else self.sparse_f
        if obst_range_r is not None:
            self.obst_min_r = obst_range_r[0]
            self.obst_max_r = obst_range_r[1]

        self.takeoff_protection_region = takeoff_prot_region if takeoff_prot_region is not None else self.takeoff_protection_region
        self.generate_obst_poses_InSample()
    
    @staticmethod # TODO: 把位置初始化放到外边去
    def init_obst_entities(scene: gs.Scene, obst_info: torch.Tensor):
        obst_entities = []
        for obst in obst_info.tolist():
            if obst[1] < 0:
                obst_e = scene.add_entity(
                    gs.morphs.Cylinder(
                        radius=obst[2],
                        height=obst[3],
                        pos=tuple(obst[4:7]),
                        quat=tuple(obst[7:]),
                        fixed=True,
                        collision=False,
                    )
                )
            else:
                obst_e = scene.add_entity(
                    gs.morphs.Box(
                        size=tuple(obst[1:4]),
                        pos=tuple(obst[4:7]),
                        quat=tuple(obst[7:]),
                        fixed=True,
                        collision=False,
                    )
                )
            obst_entities.append(obst_e)
        return obst_entities
    
    @staticmethod
    def update_obst_poses(obst_entities: list, obst_info: torch.Tensor, num_envs: int):
        # 此处仅考虑位置发生改变的情况, 在scene.build之后才调用
        for idx, obst in enumerate(obst_info.tolist()):
            obst_entities[idx].set_pos(torch.tensor(obst[4:7]).tile([num_envs, 1]), zero_velocity=True, envs_idx=list(range(num_envs)))
            obst_entities[idx].set_quat(torch.tensor(obst[7:]).tile([num_envs, 1]), zero_velocity=True, envs_idx=list(range(num_envs)))
        return obst_entities
    
    def render_obst_entities(self, obst_info: torch.Tensor = None):
        if obst_info is None:
            obst_info = self.obstacles_info
        if not self.obst_entities:
            self.obst_entities = ObstManagerInScene.init_obst_entities(self.scene, obst_info)
        else:
            # scene after build
            self.obst_entities = ObstManagerInScene.update_obst_poses(self.obst_entities, obst_info, self.scene.n_envs)

    def draw_obst_region(self, color=(1, 1, 0, 0.5), num_segments=64):
        """
        在 scene 中绘制 debug line 用于可视化边界区域
        """
        
        self.scene.clear_debug_objects()
        
        points = []
        if len(self.obst_region) == 3:
            cx, cy, cr = self.obst_region
            for i in range(num_segments + 1):
                angle = 2 * np.pi * i / num_segments
                x = cx + cr * np.cos(angle)
                y = cy + cr * np.sin(angle)
                points.append([x, y, 0.0])  # z=0 平面
        elif len(self.obst_region) > 3:
            for i in range(0, len(self.obst_region), 2):
                points.append([self.obst_region[i], self.obst_region[i+1], 0.0])
                # 闭合多边形
                points.append([self.obst_region[0], self.obst_region[1], 0.0])
        
        # 绘制闭合曲线
        if points:
            for i in range(len(points) - 1):
                self.scene.draw_debug_line(np.array(points[i]), np.array(points[i + 1]), radius=0.02, color=color)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-n", "--num_envs", type=int, default=25)
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
        "debug_pos_min_dist": 0.5,  # default between 2 poses
        # "hover_safe_dist": 0.2,     # default to RACER.urdf
        "robot_colli_safe_r": 0.2,
        "takeoff_protection_region": [0.0, 0.0, 0.5], # takeoff protection
        # obstacle settings
        "obst_enable": True,
        "obst_type": [1], # Detailed in class ObstacleType
        "obst_variety_num": 10, # obstacle shape variety
        "obst_region_type": "circle",
        "feasible_inflation_factor": 2.0, # 可行域膨胀系数，建议大于1.0，用于限制障碍物的数量和分布密度，该参数为robot_colli_safe_r的倍数，划定了智能体最小可行域的大小，grid_map_enable后决定 resolution,否则只决定数量和最小距离
        "obst_overlap":False,
        "grid_map_enable":False, 
        # "gridmap_resolution":1.0, # Grid map resolution r, default with only 1 obstacle in a r*r grid
        "cluster_local_enable":False, # 是否使用分组膨胀grid，默认为True"
        "cluster_max_inflation_times":5, # 分组膨胀grid次数
        "cluster_target_pos_num": 5, # 障碍物分小组管理，在其中套用给定的密度函数
        # "obst_occupancy_threshold": 0.5,
        "obst_movable": False,
        "dynamic_mode" : False,
        "default_obst_region": [0.0, 0.0, 5.0], # [x, y, r] default, for UAV tasks needs to be modified
        "obst_range_radius": [0.1, 0.5], # for Box sample 2 times to born XY
        "obst_range_height": [1.5, 3.0],
    }
    
    obst_manager = ObstManagerInScene(scene=scene, obst_cfg_dict=debug_obst_cfg)
    
    ########################## entities ##########################
    # add plane
    plane = scene.add_entity(gs.morphs.Plane())
    
    # initialize obstacles
    obst_manager.render_obst_entities()
    # render 结束后就不能再添加障碍物了
    
    ########################## build ##########################
    scene.build(n_envs=args.num_envs, env_spacing=(20.0, 20.0),)
    
    # 可视化边界
    obst_manager.draw_obst_region()
    
    last_update_time = time.time()
    
    while True:
        current_time = time.time()
        
        if current_time - last_update_time >= 5.0:
            debug_obst_region = [0.0, 0.0, random.uniform(5.0, 10.0)]
            obst_manager.reset_obst(obst_region=debug_obst_region, obst_region_type="circle")
            obst_manager.render_obst_entities()
            scene.clear_debug_objects()
            obst_manager.draw_obst_region()
            last_update_time = current_time
            print("🔄 正在更新 poses 和障碍物...")

        scene.step()

if __name__ == "__main__":
    main()
