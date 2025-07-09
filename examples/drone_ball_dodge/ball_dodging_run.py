import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner
from quadcopter_controller import PIDRunner

import genesis as gs

from ball_dodging_env import BallDodgingEnv

# pid_params = [
#         [2.0, 0.0, 0.0],
#         [2.0, 0.0, 0.0],
#         [2.0, 0.0, 0.0],
#         [20.0, 0.0, 20.0],
#         [20.0, 0.0, 20.0],
#         [25.0, 0.0, 20.0],
#         [10.0, 0.0, 1.0],
#         [10.0, 0.0, 1.0],
#         [2.0, 0.0, 0.2],
#     ]

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="ball_dodging")
    parser.add_argument("--ckpt", type=int, default=300)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--controller", type=str, default="PID") # PID, PPO
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, drone_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # visualize the target
    env_cfg["visualize_target"] = True
    # for video recording
    env_cfg["visualize_camera"] = args.record
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60

    env = BallDodgingEnv(
        num_envs=1,
        env_cfg=env_cfg,
        drone_cfg=drone_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    if args.controller == "PID":
        runner = PIDRunner(env, drone_cfg, pid_params, obs_cfg, dt=env_cfg["sim_dt"], device=gs.device)
        policy = runner.get_policy(device=gs.device)
    else:
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
        resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=gs.device)
        
    
    # 把 PID 参数传进来，修改观测传递方式
    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"] * 5)
    with torch.no_grad():
        if args.record:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                env.cam.render()
            env.cam.stop_recording(save_to_filename="video.mp4", fps=env_cfg["max_visualize_FPS"])
        else:
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/drone/hover_eval.py

# Note
If you experience slow performance or encounter other issues
during evaluation, try removing the --record option.
"""
