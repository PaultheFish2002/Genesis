import argparse
import os
import pickle
import shutil

from hover_env import HoverEnv
from playtime_env import PlaytimeEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.002,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,    # 1 second
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "resume_ckpt": 5600,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 200,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        # termination
        "termination_if_roll_greater_than": 180,  # degree
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,
        "termination_if_x_greater_than": 5.0,
        "termination_if_y_greater_than": 5.0,
        "termination_if_z_greater_than": 2.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "at_target_threshold": 0.1,
        "resampling_time_s": 3.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "num_obs": 17,
        "obs_scales": {
            "rel_pos": 1 / 3.0, # 疑似和边界有关 无关
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            # 悬停策略有效参数设定值1： 10.0, -1e-3, 0.02, -2e-4, -10.0, 0.1, 0.0 orin-test  悬停即计分 0.1
            # 悬停策略有效参数设定值2： 10.0, -1e-2, 0.02, -1e-2, -10.0, 0.5, 0.0 orin-test  悬停即计分，成功率更高 0.1
            # 悬停策略有效参数设定值3： 10.0, -1e-2, 0.02, -1e-2, -20.0, 1.0, -0.5 orin-tl-toward  训练悬停计分，rew_target速度矢量偏移惩罚 ckpt 200-600-1600 表现稳健
            # 悬停策略有效参数设定值4： 10.0, -1e-2, 0.02, -1e-2, -20.0, 0.5, 0.0 orin-time  训练悬停计分，rew_target速度矢量投影  
            
            "target": 10.0,      # 2.5 # 
            "smooth": -1e-2,
            "yaw": 0.02, 
            "angular": -1e-2,
            "crash": -20.0,
            "attarget": 1.0,    # 10.0 0.05
            "offset": -0.0,
            #//yaw//
            # "target": 2.0,      # 2.5
            # "smooth": -1e-4,
            # "yaw": 0.08, 
            # "angular": -2e-4,
            # "crash": -10.0,
            # "attarget": 10.0,    # 10.0
            # "timeout": 0.0,
        #//orig//
        # "yaw_lambda": -10.0,
        # "reward_scales": {
        #     "target": 10.0,
        #     "smooth": -1e-4,
        #     "yaw": 0.02, 
        #     "angular": -2e-4,
        #     "crash": -10.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-4.0, 4.0],
        "pos_y_range": [-4.0, 4.0],
        "pos_z_range": [0.8, 1.2],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="orin-with-vel-proj")
    parser.add_argument("-B", "--num_envs", type=int, default=16384) # 16384
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--resume_exp", type=str, default="attarget5-target2")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--ckpt", type=int, default=3000)
    args = parser.parse_args()

    gs.init(logging_level="warning", backend=gs.gpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    ckpt_dir = f"logs/{args.resume_exp}"
    
    train_cfg["runner"]["resume"] = args.resume
    train_cfg["runner"]["resume_path"] = os.path.join(ckpt_dir, f"model_{args.ckpt}.pt") # f"model_{args.ckpt}.pt"

    if os.path.exists(log_dir) and train_cfg["runner"]["resume"] == False:
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = HoverEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/drone/hover_train.py
"""
