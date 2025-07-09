#include "QuadrotorMixer.h"

QuadrotorMixer::QuadrotorMixer(FString aircraft_name)
{
    set_aircraft_param(aircraft_name);
    
    // 每一行对应一个电机，每一列对应roll pitch yaw thrust
    _rotor_config << -0.707107,  0.707107,  1.000000,  1.000000,
                     0.707107, -0.707107,  1.000000,  1.000000,
                     0.707107,  0.707107, -1.000000,  1.000000,
                     -0.707107, -0.707107, -1.000000,  1.000000;

    _rotor_count = 4;
    _thrust_factor = 0.3;

    for (int i = 0; i < _rotor_count; ++i) {
        _outputs_prev[i] = -1.0;
    }

    _last_rate_error = Eigen::Vector3d(0, 0, 0);

    _last_ctrl_time = std::chrono::system_clock::now();

    // 特技飞行部分
    is_acro_mode = true;
}

void QuadrotorMixer::set_aircraft_param(FString aircraft_name)
{
    _param_manager = xmlParamManager::GetInstance();
    std::map<FString, double>* controller_params =
        _param_manager->GetParams(aircraft_name, TEXT("loop"), TEXT("bodyrate"));

    _rate_k << 1.0, 1.0, 1.0;

    _rate_gain_p = _rate_k.cwiseProduct(Eigen::Vector3d(
        (*controller_params)[TEXT("p_roll")],
        (*controller_params)[TEXT("p_pitch")],
        (*controller_params)[TEXT("p_yaw")]));
    // _rate_gain_i = _rate_k.cwiseProduct(Eigen::Vector3d(0.2, 0.2, 0.1));
    _rate_gain_i = _rate_k.cwiseProduct(Eigen::Vector3d(
        (*controller_params)[TEXT("i_roll")],
        (*controller_params)[TEXT("i_pitch")],
        (*controller_params)[TEXT("i_yaw")]));
    _rate_gain_d = _rate_k.cwiseProduct(Eigen::Vector3d(
        (*controller_params)[TEXT("d_roll")],
        (*controller_params)[TEXT("d_pitch")],
        (*controller_params)[TEXT("d_yaw")]));
    _rate_gain_ff = _rate_k.cwiseProduct(Eigen::Vector3d(
        (*controller_params)[TEXT("ff_roll")],
        (*controller_params)[TEXT("ff_pitch")],
        (*controller_params)[TEXT("ff_yaw")]));
    _rate_lim_int = _rate_k.cwiseProduct(Eigen::Vector3d(
        (*controller_params)[TEXT("lim_int_roll")],
        (*controller_params)[TEXT("lim_int_pitch")],
        (*controller_params)[TEXT("lim_int_yaw")]));

    mc_acro_r_max = (*controller_params)[TEXT("roll_max")] / 57.3;
    mc_acro_p_max = (*controller_params)[TEXT("pitch_max")] / 57.3;
    mc_acro_y_max = (*controller_params)[TEXT("yaw_max")] / 57.3;
    mc_acro_expo_r = (*controller_params)[TEXT("roll_expo")];
    mc_acro_expo_p = (*controller_params)[TEXT("pitch_expo")];
    mc_acro_expo_y = (*controller_params)[TEXT("yaw_expo")];
    mc_acro_supexpo_r = (*controller_params)[TEXT("roll_superexpo")];
    mc_acro_supexpo_p = (*controller_params)[TEXT("pitch_superexpo")];
    mc_acro_supexpo_y = (*controller_params)[TEXT("yaw_superexpo")];

    delete controller_params;
}

QuadrotorMixer::~QuadrotorMixer()
{
    xmlParamManager::deleteInstance();
    UE_LOG(LogTemp, Warning, TEXT("bodyrate config node not found"));

}

Eigen::VectorXd QuadrotorMixer::GetMixerControls(const Eigen::Vector3d& rate, const Eigen::Vector3d& rate_sp, double thrust_sp)
{
    Eigen::Vector3d man_rate_sp = {0, 0, 0};
    if(is_acro_mode == true)
    {
        man_rate_sp = GetAcroRateSetpoint(rate_sp);
    }

    double mixer_outputs[4];
    Eigen::VectorXd actuator_controls{};
    actuator_controls.resize(16);

    auto ctrl_time = std::chrono::system_clock::now();

    std::chrono::duration<double> ctrl_duration = ctrl_time - _last_ctrl_time;
    double dt = ctrl_duration.count();
    _last_ctrl_time = ctrl_time;
    dt = clamp(dt, 0.001, 0.025);

    if(is_acro_mode == true)
    {
        _torque = UpdateAngularRateCtrl(rate, man_rate_sp, dt);
    }else
    {
        _torque = UpdateAngularRateCtrl(rate, rate_sp, dt);
    }
    // _torque = UpdateAngularRateCtrl(rate, rate_sp, dt);

    Mix(_torque[0], _torque[1], _torque[2], thrust_sp, mixer_outputs);
    
    // MixAirmodeDisabled(_torque[0], _torque[1], _torque[2], thrust_sp, mixer_outputs);

    double effective_output[4];
    output_limit_calc(1000, 2000, mixer_outputs, effective_output);

    double control_output[4];

    command_conversion(effective_output, control_output);

    actuator_controls[0] = control_output[0];
    actuator_controls[1] = control_output[1];
    actuator_controls[2] = control_output[2];
    actuator_controls[3] = control_output[3];

    return actuator_controls;
}

Eigen::Vector3d QuadrotorMixer::GetAcroRateSetpoint(const Eigen::Vector3d& rate)
{
    Eigen::Vector3d man_rate_sp{superexpo(rate[0], mc_acro_expo_r, mc_acro_supexpo_r),
                                superexpo(rate[1], mc_acro_expo_p, mc_acro_supexpo_p),
                                superexpo(rate[2], mc_acro_expo_y, mc_acro_supexpo_y)};

    Eigen::Vector3d acro_rate_max{mc_acro_r_max, mc_acro_p_max, mc_acro_y_max};
    
    man_rate_sp = man_rate_sp.cwiseProduct(acro_rate_max);

    return man_rate_sp;
}

Eigen::Vector3d QuadrotorMixer::UpdateAngularRateCtrl(const Eigen::Vector3d& rate, const Eigen::Vector3d& rate_sp, double dt)
{
    // 角速度误差
    Eigen::Vector3d rate_error = rate_sp - rate;
    Eigen::Vector3d angular_accel = rate_error - _last_rate_error;
    // PID 控制 + 前馈
    Eigen::Vector3d torque = _rate_gain_p.cwiseProduct(rate_error)
                           + _rate_gain_i.cwiseProduct(_rate_int)
                           // + _rate_int
                           - _rate_gain_d.cwiseProduct(angular_accel)
                           + _rate_gain_ff.cwiseProduct(rate_sp);
    updateIntegral(rate_error, dt);
    _last_rate_error = rate_error;
    return torque;
}

void QuadrotorMixer::Mix(double roll, double pitch, double yaw, double thrust, double *outputs)
{
    // MixAirmodeDisabled(roll, pitch, yaw, thrust, outputs);
    MixAirmode_RPY(roll, pitch, yaw, thrust, outputs);
    // MixAirmode_RP(roll, pitch, yaw, thrust, outputs);

    for (int i = 0; i < _rotor_count; i++) {
        if (_thrust_factor > 0.0) {
            // 简单的静态推力模型
            outputs[i] = -(1.0 - _thrust_factor) / (2.0 * _thrust_factor) + sqrtf((1.0 - _thrust_factor) *
                    (1.0 - _thrust_factor) / (4.0 * _thrust_factor * _thrust_factor) + (outputs[i] < 0.0 ? 0.0 : outputs[i] /
                            _thrust_factor));
        }
        outputs[i] = clamp((2.0 * outputs[i] - 1.0), -1.0, 1.0);
    }

    // Slew rate limiting and saturation checking   目测没什么用
    for (int i = 0; i < _rotor_count; i++) {
        bool clipping_high = false;
        bool clipping_low_roll_pitch = false;
        bool clipping_low_yaw = false;

        if (outputs[i] < -0.99f) {
            clipping_low_yaw = true;
        }

        if (_delta_out_max > 0.0f) {
            double delta_out = outputs[i] - _outputs_prev[i];

            if (delta_out > _delta_out_max) {
                outputs[i] = _outputs_prev[i] + _delta_out_max;
                clipping_high = true;

            } else if (delta_out < -_delta_out_max) {
                outputs[i] = _outputs_prev[i] - _delta_out_max;
                clipping_low_roll_pitch = true;
                clipping_low_yaw = true;
            }
        }
        _outputs_prev[i] = outputs[i];
    }
    _delta_out_max = 0.0;    
}

void QuadrotorMixer::MixAirmodeDisabled(double roll, double pitch, double yaw, double thrust, double *outputs)
{
    for (int i = 0; i < _rotor_count; i++) {
        // outputs[i] = roll * _rotor_config(i, 0) + pitch  * _rotor_config(i, 1) + yaw * _rotor_config(i, 2) + thrust * _rotor_config(i, 3);
        outputs[i] = roll * _rotor_config(i, 0) + pitch  * _rotor_config(i, 1) + thrust * _rotor_config(i, 3);
        _tmp_array[i] = _rotor_config(i, 3);
    }
    minimize_saturation(_tmp_array, outputs,  0.0, 1.0, true);

    for (int i = 0; i < _rotor_count; i++) {
        _tmp_array[i] = _rotor_config(i, 0);
    }
    minimize_saturation(_tmp_array, outputs, 0.0, 1.0, false);

    for (int i = 0; i < _rotor_count; i++) {
        _tmp_array[i] = _rotor_config(i, 1);
    }
    minimize_saturation(_tmp_array, outputs, 0.0, 1.0, false);

    mix_yaw(yaw, outputs);
}

void QuadrotorMixer::MixAirmode_RPY(double roll, double pitch, double yaw, double thrust, double *outputs)
{
    for(int i = 0; i < _rotor_count; i++)
    {
        outputs[i] = roll * _rotor_config(i, 0) + pitch  * _rotor_config(i, 1) +
                    yaw * _rotor_config(i, 2) + thrust * _rotor_config(i, 3);
        _tmp_array[i] = _rotor_config(i, 3);
    }
    minimize_saturation(_tmp_array, outputs, 0.1);

    for(int i = 0; i < _rotor_count; i++)
    {
        _tmp_array[i] = _rotor_config(i, 2);
    }
    minimize_saturation(_tmp_array, outputs, 0.1);
}

void QuadrotorMixer::MixAirmode_RP(double roll, double pitch, double yaw, double thrust, double *outputs)
{
    for(int i = 0; i < _rotor_count; i++)
    {
        outputs[i] = roll * _rotor_config(i, 0) + pitch  * _rotor_config(i, 1) + thrust * _rotor_config(i, 3);
        _tmp_array[i] = _rotor_config(i, 3);
    }
    minimize_saturation(_tmp_array, outputs);

    mix_yaw(yaw, outputs);
}

void QuadrotorMixer::minimize_saturation(const double *desaturation_vector, double *outputs, double min_output, double max_output, bool reduce_only) {
    /*desaturation_vector: 影响电机输出调整的权重向量
        outputs: 存储电机的当前推力输出值
        sat_status: 用于存储当前电机的饱和状态信息
        reduce_only: 若为 true，则仅减少推力来消除饱和，不会增加推力
    */

    // 计算一个增益 k1，用于调整 outputs 以减少饱和
    double k1 = compute_desaturation_gain(desaturation_vector, outputs, min_output, max_output);

    if (reduce_only && k1 > 0.0) {
        return;
    }

    // 对所有电机推力 outputs[i] 进行等比例调整，使其趋向非饱和状态
    for (int i = 0; i < _rotor_count; i++) {
        outputs[i] += k1 * desaturation_vector[i];
    }

    // Compute the desaturation gain again based on the updated outputs.
    // In most cases it will be zero. It won't be if max(outputs) - min(outputs) > max_output - min_output.
    // In that case adding 0.5 of the gain will equilibrate saturations.
    double k2 = 0.5f * compute_desaturation_gain(desaturation_vector, outputs, min_output, max_output);

    for (int i = 0; i < _rotor_count; i++) {
        outputs[i] += k2 * desaturation_vector[i];
    }
}

double QuadrotorMixer::compute_desaturation_gain(const double *desaturation_vector, const double *outputs, double min_output, double max_output) const
{
    double k_min = 0.0;
    double k_max = 0.0;

    for (int i = 0; i < _rotor_count; i++) {
        // Avoid division by zero. If desaturation_vector[i] is zero, there's nothing we can do to unsaturate anyway
        if (fabsf(desaturation_vector[i]) < FLT_EPSILON) {
            continue;
        }

        if (outputs[i] < min_output) {
            float k = (min_output - outputs[i]) / desaturation_vector[i];

            if (k < k_min) { k_min = k; }

            if (k > k_max) { k_max = k; }

        }

        if (outputs[i] > max_output) {
            float k = (max_output - outputs[i]) / desaturation_vector[i];

            if (k < k_min) { k_min = k; }

            if (k > k_max) { k_max = k; }

        }
    }

    // Reduce the saturation as much as possible
    return k_min + k_max;
}

void QuadrotorMixer::mix_yaw(double yaw, double *outputs)
{
    // Add yaw to outputs
    for (int i = 0; i < _rotor_count; i++) {
        outputs[i] += yaw * _rotor_config(i, 2);
        // Yaw will be used to unsaturate if needed
        _tmp_array[i] = _rotor_config(i, 2);
    }
    minimize_saturation(_tmp_array, outputs, 0.0, 1.15);

    for (int i = 0; i < _rotor_count; i++) {
        _tmp_array[i] = _rotor_config(i, 3);
    }

    // reduce thrust only
    minimize_saturation(_tmp_array, outputs, 0.f, 1.f, true);

}

void QuadrotorMixer::output_limit_calc(double min_output, double max_output, const double *output, double *effective_output) {
    for (int i = 0; i < 4; i++) {
        double control_value = output[i];

        // 计算有效输出
        effective_output[i] = control_value * (max_output - min_output) / 2 + (max_output + min_output) / 2;

        // 限制输出在范围内
        if (effective_output[i] < min_output) {
            effective_output[i] = min_output;
            // std::cout << "[DEBUG] Channel " << i << ": Clamped to Min Output = " << effective_output[i] << std::endl;

        } else if (effective_output[i] > max_output) {
            effective_output[i] = max_output;
            // std::cout << "[DEBUG] Channel " << i << ": Clamped to Max Output = " << effective_output[i] << std::endl;
        }
    }
}

void QuadrotorMixer::command_conversion(const double *output, double *control_output){
    double PWM_DEFAULT_MIN = 1000, PWM_DEFAULT_MAX = 2000;
    double pwm_center = (PWM_DEFAULT_MAX + PWM_DEFAULT_MIN) / 2;

    for (unsigned i = 0; i < 4; i++){
        if (output[i] > PWM_DEFAULT_MIN / 2){
            if (i < 4){
                control_output[i] = (output[i] - PWM_DEFAULT_MIN) / (PWM_DEFAULT_MAX - PWM_DEFAULT_MIN);
            }else{
                control_output[i] = (output[i] - pwm_center) / ((PWM_DEFAULT_MAX - PWM_DEFAULT_MIN) / 2);
            }
        }
    }
}
