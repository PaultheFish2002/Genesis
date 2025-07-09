import torch
import numpy as np
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.utils.geom import quat_to_xyz
import math

class PIDController:
    def __init__(self, kp, ki, kd, intergral_limit=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = None
        self.integral_limit = intergral_limit

    def update(self, error, dt):
        self.integral += error * dt # 修改限幅
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)  # 限幅
        # 微分计算，第一次调用时无微分
        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error

        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

class DronePIDController:
    def __init__(self, drone: DroneEntity, dt, base_rpm, pid_params):
        self.__pid_pos_x = PIDController(kp=pid_params[0][0], ki=pid_params[0][1], kd=pid_params[0][2])
        self.__pid_pos_y = PIDController(kp=pid_params[1][0], ki=pid_params[1][1], kd=pid_params[1][2])
        self.__pid_pos_z = PIDController(kp=pid_params[2][0], ki=pid_params[2][1], kd=pid_params[2][2])

        self.__pid_vel_x = PIDController(kp=pid_params[3][0], ki=pid_params[3][1], kd=pid_params[3][2])
        self.__pid_vel_y = PIDController(kp=pid_params[4][0], ki=pid_params[4][1], kd=pid_params[4][2])
        self.__pid_vel_z = PIDController(kp=pid_params[5][0], ki=pid_params[5][1], kd=pid_params[5][2])

        self.__pid_att_roll = PIDController(kp=pid_params[6][0], ki=pid_params[6][1], kd=pid_params[6][2])
        self.__pid_att_pitch = PIDController(kp=pid_params[7][0], ki=pid_params[7][1], kd=pid_params[7][2])
        self.__pid_att_yaw = PIDController(kp=pid_params[8][0], ki=pid_params[8][1], kd=pid_params[8][2])

        self.drone = drone
        self.__dt = dt
        self.__base_rpm = base_rpm

    def __get_drone_pos(self) -> torch.Tensor:
        return self.drone.get_pos()

    def __get_drone_vel(self) -> torch.Tensor:
        return self.drone.get_vel()

    def __get_drone_att(self) -> torch.Tensor:
        quat = self.drone.get_quat()
        return quat_to_xyz(quat, rpy=True, degrees=True)

    def __mixer(self, thrust, roll, pitch, yaw, x_vel, y_vel) -> torch.Tensor:
        M1 = self.__base_rpm + (thrust - roll - pitch - yaw - x_vel + y_vel)
        M2 = self.__base_rpm + (thrust - roll + pitch + yaw + x_vel + y_vel)
        M3 = self.__base_rpm + (thrust + roll + pitch - yaw + x_vel - y_vel)
        M4 = self.__base_rpm + (thrust + roll - pitch + yaw - x_vel - y_vel)
        return torch.Tensor([M1, M2, M3, M4])

    def update(self, target) -> np.ndarray:
        curr_pos = self.__get_drone_pos()
        curr_vel = self.__get_drone_vel()
        curr_att = self.__get_drone_att()

        err_pos_x = target[0] - curr_pos[:,0]
        err_pos_y = target[1] - curr_pos[:,1]
        err_pos_z = target[2] - curr_pos[:,2]
        # err_pos_x = target[0] - curr_pos[0]
        # err_pos_y = target[1] - curr_pos[1]
        # err_pos_z = target[2] - curr_pos[2]

        vel_des_x = self.__pid_pos_x.update(err_pos_x, self.__dt)
        vel_des_y = self.__pid_pos_y.update(err_pos_y, self.__dt)
        vel_des_z = self.__pid_pos_z.update(err_pos_z, self.__dt)

        error_vel_x = vel_des_x - curr_vel[:,0]
        error_vel_y = vel_des_y - curr_vel[:,1]
        error_vel_z = vel_des_z - curr_vel[:,2]
        # error_vel_x = vel_des_x - curr_vel[0]
        # error_vel_y = vel_des_y - curr_vel[1]
        # error_vel_z = vel_des_z - curr_vel[2]

        x_vel_del = self.__pid_vel_x.update(error_vel_x, self.__dt)
        y_vel_del = self.__pid_vel_y.update(error_vel_y, self.__dt)
        thrust_des = self.__pid_vel_z.update(error_vel_z, self.__dt)

        err_roll = 0.0 - curr_att[:,0]
        err_pitch = 0.0 - curr_att[:,1]
        err_yaw = 0.0 - curr_att[:,2]
        # err_roll = 0.0 - curr_att[0]
        # err_pitch = 0.0 - curr_att[1]
        # err_yaw = 0.0 - curr_att[2]

        roll_del = self.__pid_att_roll.update(err_roll, self.__dt)
        pitch_del = self.__pid_att_pitch.update(err_pitch, self.__dt)
        yaw_del = self.__pid_att_yaw.update(err_yaw, self.__dt)

        prop_rpms = self.__mixer(thrust_des, roll_del, pitch_del, yaw_del, x_vel_del, y_vel_del)
        prop_rpms = prop_rpms.cpu()
        prop_rpms - prop_rpms.numpy()

        return prop_rpms

class DronePIDControllerTest:
    def __init__(self, drone: DroneEntity, dt, base_rpm, pid_params):
        self.__pid_pos_x = PIDController(kp=pid_params[0][0], ki=pid_params[0][1], kd=pid_params[0][2])
        self.__pid_pos_y = PIDController(kp=pid_params[1][0], ki=pid_params[1][1], kd=pid_params[1][2])
        self.__pid_pos_z = PIDController(kp=pid_params[2][0], ki=pid_params[2][1], kd=pid_params[2][2])

        self.__pid_vel_x = PIDController(kp=pid_params[3][0], ki=pid_params[3][1], kd=pid_params[3][2])
        self.__pid_vel_y = PIDController(kp=pid_params[4][0], ki=pid_params[4][1], kd=pid_params[4][2])
        self.__pid_vel_z = PIDController(kp=pid_params[5][0], ki=pid_params[5][1], kd=pid_params[5][2])

        self.__pid_att_roll = PIDController(kp=pid_params[6][0], ki=pid_params[6][1], kd=pid_params[6][2])
        self.__pid_att_pitch = PIDController(kp=pid_params[7][0], ki=pid_params[7][1], kd=pid_params[7][2])
        self.__pid_att_yaw = PIDController(kp=pid_params[8][0], ki=pid_params[8][1], kd=pid_params[8][2])

        self.drone = drone
        self.__dt = dt
        self.__base_rpm = base_rpm

    def __get_drone_pos(self) -> torch.Tensor:
        return self.drone.get_pos()

    def __get_drone_vel(self) -> torch.Tensor:
        return self.drone.get_vel()

    def __get_drone_att(self) -> torch.Tensor:
        quat = self.drone.get_quat()
        return quat_to_xyz(quat, rpy=True, degrees=True)

    def __mixer(self, thrust, roll, pitch, yaw, x_vel, y_vel) -> torch.Tensor:
        
        M1 = self.__base_rpm + (thrust - roll - pitch - yaw - x_vel + y_vel)
        M2 = self.__base_rpm + (thrust - roll + pitch + yaw + x_vel + y_vel)
        M3 = self.__base_rpm + (thrust + roll + pitch - yaw + x_vel - y_vel)
        M4 = self.__base_rpm + (thrust + roll - pitch + yaw - x_vel - y_vel)
        return torch.Tensor([M1, M2, M3, M4])

    def improved_mixer(self, thrust, roll_torque, pitch_torque, yaw_torque):
        arm = 0.0397  # URDF电机臂长
        kf = 3.16e-10
        km = 7.94e-12
        km_over_kf = km / kf

        f1 = 0.25 * (thrust + pitch_torque / arm - roll_torque / arm + yaw_torque / km_over_kf)
        f2 = 0.25 * (thrust - pitch_torque / arm - roll_torque / arm - yaw_torque / km_over_kf)
        f3 = 0.25 * (thrust - pitch_torque / arm + roll_torque / arm + yaw_torque / km_over_kf)
        f4 = 0.25 * (thrust + pitch_torque / arm + roll_torque / arm - yaw_torque / km_over_kf)

        forces = torch.stack([f1, f2, f3, f4], dim=-1)
        forces = torch.clamp(forces, min=1e-6)  # 防止负推力
        
        omega = torch.sqrt(forces / kf)  # 力和转速平方成正比
        rpms = omega * 60 / (2 * torch.pi)
        
        
        return rpms
    
    def update(self, target) -> np.ndarray:
        curr_pos = self.__get_drone_pos()
        curr_vel = self.__get_drone_vel()
        curr_att = self.__get_drone_att()

        # --- 位置环 ---
        err_pos_x = target[0] - curr_pos[:,0]
        err_pos_y = target[1] - curr_pos[:,1]
        err_pos_z = target[2] - curr_pos[:,2]

        vel_des_x = self.__pid_pos_x.update(err_pos_x, self.__dt)
        vel_des_y = self.__pid_pos_y.update(err_pos_y, self.__dt)
        vel_des_z = self.__pid_pos_z.update(err_pos_z, self.__dt)

        # --- 速度环 ---
        error_vel_x = vel_des_x - curr_vel[:,0]
        error_vel_y = vel_des_y - curr_vel[:,1]
        error_vel_z = vel_des_z - curr_vel[:,2]

        x_acc_del = self.__pid_vel_x.update(error_vel_x, self.__dt)
        y_acc_del = self.__pid_vel_y.update(error_vel_y, self.__dt)
        z_acc_des = self.__pid_vel_z.update(error_vel_z, self.__dt)
        
        # --- 补上重力，加 hover thrust ---
        mass = 0.027  # 来自 URDF
        g = 9.81
        total_thrust_div = torch.clamp(mass * z_acc_des, min=0.0, max=1.0) # 1.5倍基础转速推导而得， T_max = 4 * thrust_max, thrust_max = kf * (omega_max ** 2), omega_max = 2 * math.pi / 60 * max_rpm
        
        # --- 姿态角环 ---
        # 目标roll和pitch，假设期望姿态小，直接用x_vel_del, y_vel_del转成角度, 小角度值 30 deg / 0.5 rad
        roll_target = torch.clamp(-y_acc_del / g, -0.5, 0.5) # 通常横向速度误差对应roll
        pitch_target = torch.clamp(x_acc_del / g, -0.5, 0.5)  # 通常前向速度误差对应pitch
        yaw_target = 0.0          # 默认维持初始朝向
        
        # 注意上面已经转为弧度
        err_roll = roll_target - curr_att[:, 0] * math.pi / 180  # roll误差
        err_pitch = pitch_target - curr_att[:, 1] * math.pi / 180 # pitch误差
        err_yaw = yaw_target - curr_att[:, 2] * math.pi / 180     # yaw误差

        roll_torque = self.__pid_att_roll.update(err_roll, self.__dt) * 0
        pitch_torque = self.__pid_att_pitch.update(err_pitch, self.__dt) * 0
        yaw_torque = self.__pid_att_yaw.update(err_yaw, self.__dt) * 0

        # 再加一个PID?计算rate
        
        # prop_rpms = self.__mixer(total_thrust, roll_del, pitch_del, yaw_del, x_vel_del, y_vel_del)
        prop_rpms = self.improved_mixer(total_thrust_div, roll_torque, pitch_torque, yaw_torque)

        # prop_rpms = prop_rpms - prop_rpms.mean() + self.__base_rpm
        # prop_rpms = torch.clamp(prop_rpms, min=0.0)  # 防止负数
        
        return prop_rpms.cpu().numpy()

class PIDRunner:
    def __init__(self, env, drone_cfg, pid_params, obs_cfg=None, dt=0.01, ctrl_loop="pos", device="cpu"):
        self.env = env # BallDodgingEnv
        self.pid_params = pid_params
        self.device = device
        self.ctrl_loop = ctrl_loop # "pos" or "vel"
        self.obs_scale = obs_cfg["obs_scales"]
        
        self.__base_rpm = drone_cfg["base_rpm"]
        self.__dt = dt
        self.__rpm_scale = drone_cfg["rpm_scale"]
        
        self.err_pos, self.curr_vel, self.curr_att = -1, -1, -1
        
        # pid controllers param setup, not optimized
        self.__pid_pos_x = PIDController(kp=pid_params[0][0], ki=pid_params[0][1], kd=pid_params[0][2])
        self.__pid_pos_y = PIDController(kp=pid_params[1][0], ki=pid_params[1][1], kd=pid_params[1][2])
        self.__pid_pos_z = PIDController(kp=pid_params[2][0], ki=pid_params[2][1], kd=pid_params[2][2])

        self.__pid_vel_x = PIDController(kp=pid_params[3][0], ki=pid_params[3][1], kd=pid_params[3][2])
        self.__pid_vel_y = PIDController(kp=pid_params[4][0], ki=pid_params[4][1], kd=pid_params[4][2])
        self.__pid_vel_z = PIDController(kp=pid_params[5][0], ki=pid_params[5][1], kd=pid_params[5][2])

        self.__pid_att_roll = PIDController(kp=pid_params[6][0], ki=pid_params[6][1], kd=pid_params[6][2])
        self.__pid_att_pitch = PIDController(kp=pid_params[7][0], ki=pid_params[7][1], kd=pid_params[7][2])
        self.__pid_att_yaw = PIDController(kp=pid_params[8][0], ki=pid_params[8][1], kd=pid_params[8][2])
        
    
    def __mixer(self, thrust, roll, pitch, yaw, x_vel, y_vel) -> torch.Tensor:
        M1 = self.__base_rpm + (thrust - roll - pitch - yaw - x_vel + y_vel)
        M2 = self.__base_rpm + (thrust - roll + pitch + yaw + x_vel + y_vel)
        M3 = self.__base_rpm + (thrust + roll + pitch - yaw + x_vel - y_vel)
        M4 = self.__base_rpm + (thrust + roll - pitch + yaw - x_vel - y_vel)
        return torch.stack([M1, M2, M3, M4], dim=2)
    def obs_converter(self, obs):
        # obs = tensorND [rel_pos[3], base_quat[4], lin_vel[3], ang_vel[3], last_actions[4]]
        if self.obs_scale is not None:
            err_pos, quat, curr_vel= obs[:, :, :3] / self.obs_scale["rel_pos"], obs[:, :, 3:7], obs[:, :, 7:10] / self.obs_scale["lin_vel"]
        else:
            err_pos, quat, curr_vel= obs[:, :, :3], obs[:, :, 3:7], obs[:, :, 7:10]
        curr_att = quat_to_xyz(quat, rpy=True, degrees=True)
        return err_pos, curr_vel, curr_att
    
    def act(self, obs):
        self.err_pos, self.curr_vel, self.curr_att = self.obs_converter(obs)
        
        if self.ctrl_loop == "pos":
            actions = self.update_pos_ctrl()
        elif self.ctrl_loop == "vel": # Not finished yet
            actions = self.update_vel_ctrl()
        return actions
    
    def update_pos_ctrl(self, target=None): # 转换为 torch.Tensor 佳矣, 改为批量计算
        err_pos_x = self.err_pos[:,:,0]
        err_pos_y = self.err_pos[:,:,1]
        err_pos_z = self.err_pos[:,:,2]

        vel_des_x = self.__pid_pos_x.update(err_pos_x, self.__dt)
        vel_des_y = self.__pid_pos_y.update(err_pos_y, self.__dt)
        vel_des_z = self.__pid_pos_z.update(err_pos_z, self.__dt)

        error_vel_x = vel_des_x - self.curr_vel[:,:,0]
        error_vel_y = vel_des_y - self.curr_vel[:,:,1]
        error_vel_z = vel_des_z - self.curr_vel[:,:,2]

        x_vel_del = self.__pid_vel_x.update(error_vel_x, self.__dt)
        y_vel_del = self.__pid_vel_y.update(error_vel_y, self.__dt)
        thrust_des = self.__pid_vel_z.update(error_vel_z, self.__dt)

        err_roll = 0.0 - self.curr_att[:,:,0]
        err_pitch = 0.0 - self.curr_att[:,:,1]
        err_yaw = 0.0 - self.curr_att[:,:,2]

        roll_del = self.__pid_att_roll.update(err_roll, self.__dt)
        pitch_del = self.__pid_att_pitch.update(err_pitch, self.__dt)
        yaw_del = self.__pid_att_yaw.update(err_yaw, self.__dt)

        prop_rpms = self.__mixer(thrust_des, roll_del, pitch_del, yaw_del, x_vel_del, y_vel_del)
        # prop_rpms = prop_rpms.cpu()
        # prop_rpms - prop_rpms.numpy()
        # prop_rpms_factors = prop_rpms / self.__base_rpm - 1

        prop_rpms_clip = torch.clamp(prop_rpms, self.__rpm_scale[0] * self.__base_rpm, self.__rpm_scale[1] * self.__base_rpm)
        
        return prop_rpms_clip
    
    def update_vel_ctrl(self, target=None):
        pass
    
    def get_policy(self, device=None):
        policy = lambda x: self.act(x)
        return policy