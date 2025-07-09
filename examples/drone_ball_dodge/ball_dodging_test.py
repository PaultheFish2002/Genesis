import genesis as gs
import math
from quadcopter_controller import *
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.vis.camera import Camera

base_rpm = 14468.429183500699
min_rpm = 0.9 * base_rpm
max_rpm = 1.5 * base_rpm


def hover(drone: DroneEntity):
    drone.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])


def clamp(rpm):
    return max(min_rpm, min(int(rpm), max_rpm))


def fly_to_point(target, controller: DronePIDController, scene: gs.Scene, cam: Camera, hover=False):
    drone = controller.drone
    step = 0
    x = target[0] - drone.get_pos()[:, 0]
    y = target[1] - drone.get_pos()[:, 1]
    z = target[2] - drone.get_pos()[:, 2]

    distance = math.sqrt(x**2 + y**2 + z**2)

    while distance > 0.1 and step < 1000:
        # [M1, M2, M3, M4] = controller.update(target)
        # M1 = clamp(M1)
        # M2 = clamp(M2)
        # M3 = clamp(M3)
        # M4 = clamp(M4)
        prop_rpms = controller.update(target)
        prop_rpms = torch.clamp(prop_rpms, min_rpm, max_rpm)
        drone.set_propellels_rpm(prop_rpms.unsqueeze(0))
        scene.step()
        cam.render()
        # print("point =", drone.get_pos())
        drone_pos = drone.get_pos()
        drone_pos = drone_pos.cpu().numpy()
        x = drone_pos[:, 0]
        y = drone_pos[:, 1]
        z = drone_pos[:, 2]
        cam.set_pose(lookat=(x.item(), y.item(), z.item()))
        x = target[0] - x.item()
        y = target[0] - y.item()
        z = target[0] - z.item()
        distance = math.sqrt(x**2 + y**2 + z**2)
        step += 1

def main():
    gs.init(backend=gs.gpu)

    ##### scene #####
    scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=0.01))

    ##### entities #####
    plane = scene.add_entity(morph=gs.morphs.Plane())

    # TODO: add a ball & swarm 
    drone = scene.add_entity(morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 1.0))) # 0,0,0.2

    # parameters are tuned such that the
    # drone can fly, not optimized
    pid_params = [
        [2.0, 0.05, 0.0],
        [2.0, 0.05, 0.0],
        [3.0, 0.15, 0.0],
        [20.0, 0.0, 20.0],
        [20.0, 0.0, 20.0],
        [25.0, 0.0, 20.0],
        [10.0, 0.0, 1.0],
        [10.0, 0.0, 1.0],
        [2.0, 0.0, 0.2],
    ]

    # 调参前的初始参数
    # pid_params = [
    #     (2.0, 0.0, 0.0),   # pos_x
    #     (2.0, 0.0, 0.0),   # pos_y
    #     (3.0, 0.05, 0.0),  # pos_z
    #     (4.0, 0.0, 0.2),   # vel_x
    #     (4.0, 0.0, 0.2),   # vel_y
    #     (5.0, 0.05, 0.0),  # vel_z
    #     (10.0, 0.0, 1.5),  # roll
    #     (10.0, 0.0, 1.5),  # pitch
    #     (5.0, 0.0, 0.8),  # yaw
    # ]
    
    controller = DronePIDController(drone=drone, dt=0.01, base_rpm=base_rpm, pid_params=pid_params)

    cam = scene.add_camera(pos=(1, 1, 1), lookat=drone.morph.pos, GUI=False, res=(640, 480), fov=30)

    ##### build #####

    scene.build(n_envs=1)
    
    # 测试 小球的运动学 

    # cam.start_recording()

    points = [(1, 1, 2), (-1, 2, 1), (0, 0, 1.5), (-1, -1, 2), (0, 0, 0.5)]

    for point in points:
        fly_to_point(point, controller, scene, cam)
    # point = (0, 0, 2)
    
    # fly_to_point(point, controller, scene, cam)

    # cam.stop_recording(save_to_filename="../../videos/fly_route.mp4")


if __name__ == "__main__":
    main()
