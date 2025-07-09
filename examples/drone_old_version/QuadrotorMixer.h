#pragma once

#include "QuadRotorControlLibrary/Include/Eigen/Eigen"
#include "QuadRotorControlLibrary/Include/Eigen/Dense"
#include <chrono>
#include <cfloat>
#include <cmath>
#include "XmlParser/Public/XmlParser.h"
#include "xmlParamManager.h"
#include <map>


class QuadrotorMixer {

public:
	QuadrotorMixer(FString aircraft_name);

	void set_aircraft_param(FString aircraft_name);
	
	~QuadrotorMixer();

	Eigen::VectorXd GetMixerControls(const Eigen::Vector3d& rate, const Eigen::Vector3d& rate_sp, double thrust_sp);

	void output_limit_calc(double min_output, double max_output, const double *output, double *effective_output);

	void command_conversion(const double *output, double *control_output);

private:
	Eigen::Vector3d GetAcroRateSetpoint(const Eigen::Vector3d& rate);

	Eigen::Vector3d UpdateAngularRateCtrl(const Eigen::Vector3d& rate, const Eigen::Vector3d& rate_sp, double dt);

	void Mix(double roll, double pitch, double yaw, double thrust, double *outputs);
	
	void MixAirmodeDisabled(double roll, double pitch, double yaw, double thrust, double *outputs);

	void MixAirmode_RPY(double roll, double pitch, double yaw, double thrust, double *outputs);

	void MixAirmode_RP(double roll, double pitch, double yaw, double thrust, double *outputs);
	
	void updateIntegral(const Eigen::Vector3d& rate_error, float dt){
		_rate_int += rate_error * dt;
	}

	double clamp(double value, double min_value, double max_value) {
		if (value < min_value) {
			return min_value;
		} else if (value > max_value) {
			return max_value;
		}
		return value;
	}

	void minimize_saturation(const double *desaturation_vector, double *outputs,
					double min_output = 0.0, double max_output = 1.0, bool reduce_only = false);

	double compute_desaturation_gain(const double *desaturation_vector, const double *outputs, double min_output, double max_output) const;

	void mix_yaw(double yaw, double *outputs);

	double expo(double value, double e)
	{
		double x = clamp(value, -1.0, 1.0);
		double ec = clamp(e, 0.0, 1.0);
		return (1.0 - ec) * x + ec * x * x * x;
	}

	double superexpo(double value, double e, double g)
	{
		double x = clamp(value, -1.0, 1.0);
		double gc = clamp(g, 0.0, 0.99);
		return expo(x, e) * (1.0 - gc) / (1.0 - std::fabs(x) * gc);
	}

private:
	// 顺序为滚转-俯仰-偏航
	Eigen::Vector3d _rate_gain_p;  	// 比例增益
	Eigen::Vector3d _rate_gain_i;  	// 积分增益
	Eigen::Vector3d _rate_gain_d;  	// 微分增益
	Eigen::Vector3d _rate_gain_ff; 	// 前馈增益
	Eigen::Vector3d _rate_int;     	// 积分项
	Eigen::Vector3d _rate_lim_int; 	// 积分上限

	Eigen::Vector3d _rate_k;

	Eigen::Vector3d _last_rate_error;
	double _tmp_array[4] = {0};
	double _outputs_prev[4] = {0};

	Eigen::Matrix4d _rotor_config;
	int _rotor_count;
	double _thrust_factor{0.0};
	double _delta_out_max{0.0};

	// 特技飞行部分
	bool is_acro_mode;
	double mc_acro_r_max, mc_acro_p_max, mc_acro_y_max;
	double mc_acro_expo_r, mc_acro_expo_p, mc_acro_expo_y;
	double mc_acro_supexpo_r, mc_acro_supexpo_p, mc_acro_supexpo_y;
	
	Eigen::Vector3d _torque;
	std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000>>> _last_ctrl_time;
	xmlParamManager* _param_manager;
};