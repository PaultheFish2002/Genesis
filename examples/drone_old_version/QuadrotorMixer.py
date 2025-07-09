# import numpy as np
import time
import xml.etree.ElementTree as ET
import os
import torch

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def expo(value, e):
    x = clamp(value, -1.0, 1.0)
    ec = clamp(e, 0.0, 1.0)
    return (1.0 - ec) * x + ec * x * x * x

def superexpo(value, e, g):
    x = clamp(value, -1.0, 1.0)
    gc = clamp(g, 0.0, 0.99)
    return expo(x, e) * (1.0 - gc) / (1.0 - abs(x) * gc)

class QuadrotorMixer:
    def __init__(self, aircraft_name, num_batch=1, device="cuda"):
        self.num_batch = num_batch
        self.device = torch.device(device)
        
        self.set_aircraft_param(aircraft_name)
        self._rotor_config = torch.tensor([
            [-0.707107,  0.707107,  1.000000,  1.000000],
            [ 0.707107, -0.707107,  1.000000,  1.000000],
            [ 0.707107,  0.707107, -1.000000,  1.000000],
            [-0.707107, -0.707107, -1.000000,  1.000000]
        ], device=self.device)
        self._rotor_count = 4
        self._thrust_factor = 0.3
        self._outputs_prev = torch.full((4,), -1.0).tile((self.num_batch, 1)).to(self.device)
        self._rate_int = torch.zeros(3).tile((self.num_batch, 1)).to(self.device)
        self._last_rate_error = torch.zeros(3).tile((self.num_batch, 1)).to(self.device)
        self._last_ctrl_time = time.time()
        self._tmp_array = torch.zeros(4).to(self.device)
        self._delta_out_max = 0.0
        self.is_acro_mode = False # 禁用 airmode 模式

    def set_aircraft_param(self, aircraft_name):
        
        # 解析 tmp.xml 文件
        tree = ET.parse(os.path.join(os.path.dirname(__file__), 'tmp.xml'))
        root = tree.getroot()

        # 找到对应的 aircraft 节点
        aircraft = None
        for ac in root.findall('aircraft'):
            if ac.get('name') == aircraft_name:
                aircraft = ac
                break

        if aircraft is None:
            raise ValueError(f"Aircraft {aircraft_name} not found in tmp.xml")

        # 找到 bodyrate 控制环
        loop = None
        for lp in aircraft.findall('loop'):
            if lp.get('name') == 'bodyrate':
                loop = lp
                break

        if loop is None:
            raise ValueError(f"'bodyrate' loop not found for aircraft {aircraft_name}")

        # 读取参数
        def get_float(tag): return float(loop.find(tag).text.strip())

        self._rate_k = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        self._rate_gain_p = self._rate_k * torch.tensor([get_float("p_roll"), get_float("p_pitch"), get_float("p_yaw")], device=self.device)
        self._rate_gain_i = self._rate_k * torch.tensor([get_float("i_roll"), get_float("i_pitch"), get_float("i_yaw")], device=self.device)
        self._rate_gain_d = self._rate_k * torch.tensor([get_float("d_roll"), get_float("d_pitch"), get_float("d_yaw")], device=self.device)
        self._rate_gain_ff = self._rate_k * torch.tensor([get_float("ff_roll"), get_float("ff_pitch"), get_float("ff_yaw")], device=self.device)
        self._rate_lim_int = self._rate_k * torch.tensor([get_float("lim_int_roll"), get_float("lim_int_pitch"), get_float("lim_int_yaw")], device=self.device)

        self.mc_acro_r_max = get_float("roll_max") / 57.3
        self.mc_acro_p_max = get_float("pitch_max") / 57.3
        self.mc_acro_y_max = get_float("yaw_max") / 57.3
        self.mc_acro_expo_r = get_float("roll_expo")
        self.mc_acro_expo_p = get_float("pitch_expo")
        self.mc_acro_expo_y = get_float("yaw_expo")
        self.mc_acro_supexpo_r = get_float("roll_superexpo")
        self.mc_acro_supexpo_p = get_float("pitch_superexpo")
        self.mc_acro_supexpo_y = get_float("yaw_superexpo")

    def GetMixerControls(self, rate, rate_sp, thrust_sp):
        """
        批量版本的控制器输入处理：接受 B x 3 的向量。
        :param rate_batch: shape = (B, 3)
        :param rate_sp_batch: shape = (B, 3)
        :param thrust_sp_batch: shape = (B,) 或 (B, 1)
        :return: actuator_controls_batch: shape = (B, 16)
        """        
        # TODO: 查找初始化之后先前变量被错误保留的bug
        now = time.time()
        dt = clamp(now - self._last_ctrl_time, 0.001, 0.025)
        self._last_ctrl_time = now

        man_rate_sp = self.GetAcroRateSetpoint(rate_sp) if self.is_acro_mode else rate_sp
        self._torque = self.UpdateAngularRateCtrl(rate, man_rate_sp, dt)

        mixer_outputs = torch.zeros(4).tile((self.num_batch, 1)).to(self.device)
        mixer_outputs = self.Mix(self._torque[:,0].reshape(-1,1), 
                                 self._torque[:,1].reshape(-1,1),
                                 self._torque[:,2].reshape(-1,1), thrust_sp, mixer_outputs)

        # mixer_outputs 限制在 [-1,1] 之间, 拟直接输出
        
        # effective_output = torch.zeros(4).tile((self.num_batch, 1)).to(self.device)
        # effective_output = self.output_limit_calc(1000, 2000, mixer_outputs, effective_output)

        # control_output = torch.zeros(4).tile((self.num_batch, 1)).to(self.device)
        # control_output = self.command_conversion(effective_output, control_output)

        actuator_controls = torch.zeros(16).tile((self.num_batch, 1)).to(self.device)
        actuator_controls[:, :4] = mixer_outputs 
        
        # 转成转速输出即可
        
        return actuator_controls

    def GetAcroRateSetpoint(self, rate):
        sp = torch.tensor([
            superexpo(rate[0].item(), self.mc_acro_expo_r, self.mc_acro_supexpo_r),
            superexpo(rate[1].item(), self.mc_acro_expo_p, self.mc_acro_supexpo_p),
            superexpo(rate[2].item(), self.mc_acro_expo_y, self.mc_acro_supexpo_y)
        ])
        return sp * torch.tensor([self.mc_acro_r_max, self.mc_acro_p_max, self.mc_acro_y_max])

    def UpdateAngularRateCtrl(self, rate, rate_sp, dt):
        rate_error = rate_sp - rate
        angular_accel = rate_error - self._last_rate_error
        torque = (
            self._rate_gain_p * rate_error +
            self._rate_gain_i * self._rate_int -
            self._rate_gain_d * angular_accel +
            self._rate_gain_ff * rate_sp
        )
        self._rate_int += rate_error * dt
        self._last_rate_error = rate_error
        return torque

    def Mix(self, roll, pitch, yaw, thrust, outputs_init):
        
        outputs = self.MixAirmodeDisabled(roll, pitch, yaw, thrust, outputs_init)
        
        if self._thrust_factor > 0:
            # 计算常数项
            a = (1.0 - self._thrust_factor) / (2.0 * self._thrust_factor)
            b = ((1.0 - self._thrust_factor) ** 2) / (4.0 * self._thrust_factor ** 2)

            # 修正推力：outputs ← -a + sqrt(b + max(0, outputs / tf))
            safe_outputs = torch.clamp(outputs / self._thrust_factor, min=0.0)
            outputs = -a + torch.sqrt(b + safe_outputs)
        
        # 归一化输出
        outputs = 2.0 * outputs - 1.0
        outputs = torch.clamp(outputs, -1.0, 1.0)

        # 差分限幅（逐通道）
        if self._delta_out_max > 0:
            prev = self._outputs_prev.unsqueeze(0)  # shape [B, 4]
            delta = outputs - prev
            over = delta > self._delta_out_max
            under = delta < -self._delta_out_max

            outputs = torch.where(over, prev + self._delta_out_max, outputs)
            outputs = torch.where(under, prev - self._delta_out_max, outputs)

        # 更新历史值（只保留最后一个样本的输出作为 "prev"）
        self._outputs_prev = outputs.detach()  # shape [B, 4]
        self._delta_out_max = 0.0
        
        return outputs
        
        # for i in range(self._rotor_count):
        #     if self._thrust_factor > 0:
        #         raw = outputs[i]
        #         outputs[i] = -(1.0 - self._thrust_factor) / (2.0 * self._thrust_factor) + torch.sqrt(
        #             ((1.0 - self._thrust_factor) ** 2) / (4.0 * self._thrust_factor ** 2) +
        #             max(0.0, raw / self._thrust_factor)
        #         )
        #     outputs[i] = clamp(2.0 * outputs[i] - 1.0, -1.0, 1.0)

        # for i in range(self._rotor_count):
        #     delta_out = outputs[i] - self._outputs_prev[i]
        #     if self._delta_out_max > 0:
        #         if delta_out > self._delta_out_max:
        #             outputs[i] = self._outputs_prev[i] + self._delta_out_max
        #         elif delta_out < -self._delta_out_max:
        #             outputs[i] = self._outputs_prev[i] - self._delta_out_max
        #     self._outputs_prev[i] = outputs[i]
        

    # def minimize_saturation(self, desaturation_vector, outputs, min_output=0.0, max_output=1.0, reduce_only=False):
    #     """
    #     This method minimizes the saturation of the outputs.
    #     Adjusts outputs to avoid saturation based on the desaturation_vector.
    #     """
    #     # Compute a gain 'k' to adjust the outputs for saturation reduction
    #     k1 = self.compute_desaturation_gain(desaturation_vector, outputs, min_output, max_output)

    #     if reduce_only and k1 > 0.0:
    #         return outputs

    #     # Apply proportional adjustment to outputs
    #     for i in range(self._rotor_count):
    #         outputs[i] += k1 * desaturation_vector[i]

    #     # Compute another gain 'k' based on updated outputs for further desaturation
    #     k2 = 0.5 * self.compute_desaturation_gain(desaturation_vector, outputs, min_output, max_output)

    #     for i in range(self._rotor_count):
    #         outputs[i] += k2 * desaturation_vector[i]
        
    #     return outputs
    
    def minimize_saturation(self, desaturation_vector, outputs_init, min_output=0.0, max_output=1.0, reduce_only=False):
        """
        This method minimizes the saturation of the outputs.
        Adjusts outputs to avoid saturation based on the desaturation_vector.
        """
        outputs = outputs_init
        desat_vec = desaturation_vector.unsqueeze(0)
        # Compute a gain 'k' to adjust the outputs for saturation reduction
        k1 = self.compute_desaturation_gain(desaturation_vector, outputs_init, min_output, max_output)

        if reduce_only:
            reduce_mask = (k1 <= 0.0).squeeze(1)  # shape [B]
            if not torch.any(reduce_mask):
                return outputs  # 无需进一步处理

            # 只更新需要减少的样本
            outputs[reduce_mask] += k1[reduce_mask] * desat_vec  # broadcasting [N, 1] * [1, 4]

            # Step 2: 重新计算 k2 并更新
            k2 = 0.5 * self.compute_desaturation_gain(desaturation_vector, outputs, min_output, max_output)
            outputs[reduce_mask] += k2[reduce_mask] * desat_vec

        else:
            # 所有样本都进行输出调整
            outputs += k1 * desat_vec
            k2 = 0.5 * self.compute_desaturation_gain(desaturation_vector, outputs, min_output, max_output)
            outputs += k2 * desat_vec
        
        return outputs

    # def compute_desaturation_gain(self, desaturation_vector, outputs, min_output, max_output):
    #     """
    #     This method computes the gain needed to desaturate the outputs.
    #     """
    #     k_min = 0.0
    #     k_max = 0.0

    #     for i in range(self._rotor_count):
    #         if abs(desaturation_vector[i]) < torch.finfo(torch.float).eps:
    #             continue

    #         if outputs[i] < min_output:
    #             k = (min_output - outputs[:,i]) / desaturation_vector[i]
    #             k_min = min(k_min, k)
    #             k_max = max(k_max, k)

    #         if outputs[i] > max_output:
    #             k = (max_output - outputs[i]) / desaturation_vector[i]
    #             k_min = min(k_min, k)
    #             k_max = max(k_max, k)

    #     return k_min + k_max
    
    def compute_desaturation_gain(self, desaturation_vector, outputs, min_output, max_output):
        """
        This method computes the gain needed to desaturate the outputs.
        """
        k_min = torch.zeros(1).tile((self.num_batch, 1)).to(self.device)
        k_max = torch.zeros(1).tile((self.num_batch, 1)).to(self.device)

        for i in range(self._rotor_count):
            if abs(desaturation_vector[i]) < torch.finfo(torch.float).eps:
                continue

            under_mask = outputs[:,i].unsqueeze(1) < min_output
            k_under = (min_output - outputs[:,i].unsqueeze(1)) / desaturation_vector[i]
            k_min = torch.where(under_mask, torch.minimum(k_min, k_under), k_min)
            k_max = torch.where(under_mask, torch.maximum(k_max, k_under), k_max)


            over_mask = outputs[:,i].unsqueeze(1) > max_output
            k_over = (max_output - outputs[:,i].unsqueeze(1)) / desaturation_vector[i]
            k_min = torch.where(over_mask, torch.minimum(k_min, k_over), k_min)
            k_max = torch.where(over_mask, torch.maximum(k_max, k_over), k_max)

        return k_min + k_max

    def mix_yaw(self, yaw, outputs):
        """
        Adjusts the yaw for the rotor outputs.
        """
        outputs += yaw * self._rotor_config[:, 2].unsqueeze(0)
        
        self._tmp_array = self._rotor_config[:, 2]
        outputs = self.minimize_saturation(self._tmp_array, outputs, 0.0, 1.15)
        
        self._tmp_array= self._rotor_config[:, 3]
        outputs = self.minimize_saturation(self._tmp_array, outputs, 0.0, 1.0, True)
        return outputs
    
    # def mix_yaw(self, yaw, outputs):
    #     """
    #     Adjusts the yaw for the rotor outputs.
    #     """
    #     for i in range(self._rotor_count):
    #         outputs[i] += yaw * self._rotor_config[i, 2]
    #         self._tmp_array[i] = self._rotor_config[i, 2]
    #     outputs = self.minimize_saturation(self._tmp_array, outputs, 0.0, 1.15)
        
    #     for i in range(self._rotor_count):
    #         self._tmp_array[i] = self._rotor_config[i, 3]
    #     outputs = self.minimize_saturation(self._tmp_array, outputs, 0.0, 1.0, True)
    #     return outputs

    def MixAirmodeDisabled(self, roll, pitch, yaw, thrust, outputs_init):
        """
        The 'Airmode Disabled' version of the mixing function, adjusting roll, pitch, and thrust.
        """
        for i in range(self._rotor_count):
            outputs_init[:,i] = (roll * self._rotor_config[i, 0] + pitch * self._rotor_config[i, 1] + thrust * self._rotor_config[i, 3]).reshape(1,-1)
            self._tmp_array[i] = self._rotor_config[i, 3]

        # Apply saturation minimization
        outputs0 = self.minimize_saturation(self._tmp_array, outputs_init, 0.0, 1.0, reduce_only=True)

        for i in range(self._rotor_count):
            self._tmp_array[i] = self._rotor_config[i, 0]

        outputs1 = self.minimize_saturation(self._tmp_array, outputs0, 0.0, 1.0, reduce_only=False)

        for i in range(self._rotor_count):
            self._tmp_array[i] = self._rotor_config[i, 1]

        outputs2 = self.minimize_saturation(self._tmp_array, outputs1, 0.0, 1.0, reduce_only=False)

        # Apply yaw mixing
        outputs_final = self.mix_yaw(yaw, outputs2)
        
        return outputs_final

    def MixAirmode_RPY(self, roll, pitch, yaw, thrust, outputs):
        """
        The 'Airmode RPY' version of the mixing function, adjusting roll, pitch, yaw, and thrust.
        """
        for i in range(self._rotor_count):
            outputs[i] = roll * self._rotor_config[i, 0] + pitch * self._rotor_config[i, 1] + yaw * self._rotor_config[i, 2] + thrust * self._rotor_config[i, 3]
            self._tmp_array[i] = self._rotor_config[i, 3]

        # Apply saturation minimization for the thrust
        outputs = self.minimize_saturation(self._tmp_array, outputs, 0.1)

        for i in range(self._rotor_count):
            self._tmp_array[i] = self._rotor_config[i, 2]

        # Apply saturation minimization for yaw
        outputs = self.minimize_saturation(self._tmp_array, outputs, 0.1)
        
        return outputs

    def MixAirmode_RP(self, roll, pitch, yaw, thrust, outputs):
        """
        The 'Airmode RP' version of the mixing function, adjusting roll, pitch, and thrust.
        """
        for i in range(self._rotor_count):
            outputs[i] = roll * self._rotor_config[i, 0] + pitch * self._rotor_config[i, 1] + thrust * self._rotor_config[i, 3]
            self._tmp_array[i] = self._rotor_config[i, 3]

        # Apply saturation minimization for the thrust
        outputs = self.minimize_saturation(self._tmp_array, outputs)

        # Apply yaw mixing
        self.mix_yaw(yaw, outputs)
        
        return outputs
    
    def output_limit_calc(self, min_output, max_output, output, effective_output):
        for i in range(4):
            val = output[i]
            effective_output[i] = val * (max_output - min_output) / 2 + (max_output + min_output) / 2
            effective_output[i] = clamp(effective_output[i], min_output, max_output)
        return effective_output

    def command_conversion(self, output, control_output):
        PWM_DEFAULT_MIN, PWM_DEFAULT_MAX = 1000, 2000
        pwm_center = (PWM_DEFAULT_MAX + PWM_DEFAULT_MIN) / 2
        for i in range(4):
            if output[i] > PWM_DEFAULT_MIN / 2:
                if i < 4:
                    control_output[i] = (output[i] - PWM_DEFAULT_MIN) / (PWM_DEFAULT_MAX - PWM_DEFAULT_MIN)
                else:
                    control_output[i] = (output[i] - pwm_center) / ((PWM_DEFAULT_MAX - PWM_DEFAULT_MIN) / 2)
        return control_output

if __name__ == "__main__":
    # 初始化混控器（使用 tmp.xml 中的 DX141 配置）
    mixer = QuadrotorMixer("DX141", 4)

    # 定义当前角速度（rate）和期望角速度（rate_sp）
    current_rate = torch.tensor([[ 0.7604,  0.9009,  0.0450],
                                [ 0.0992, -0.8224, -0.4512],
                                [ 0.5185, -1.0000, -0.4483],
                                [ 0.6604, -1.0000, -0.4321]], device=torch.device("cuda"))
    target_rate = torch.tensor([[0.6166, 1.0000, 0.5618],
                                [1.0000, 0.8178, 1.0000],
                                [0.6983, 1.0000, 1.0000],
                                [0.8528, 1.0000, 1.0000]], device=torch.device("cuda"))  # 模拟目标角速度
    thrust = torch.tensor([[0.3409],
                        [1.0000],
                        [1.0000],
                        [1.0000]], device=torch.device("cuda"))  # 模拟推力输入 

    # 获取控制输出
    control_outputs = mixer.GetMixerControls(current_rate, target_rate, thrust)

    # 打印前四路电机控制信号
    print("控制输出 (前4个电机):")
    for i in range(4):
        print(f"Motor {i+1}: {control_outputs[1, i].item():.4f}")