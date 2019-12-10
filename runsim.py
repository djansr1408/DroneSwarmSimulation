from utils import *
from Quadcopter import Quadcopter
import controller
from traj_generator import *

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


if __name__ == "__main__":
    state_0 = init_state(np.array([0.5, 0, 0]), yaw=0.0)
    quad = Quadcopter(state_0, params.L, params.H)
    waypoints = calculate_helix_waypoints(0, 9)
    coeff_x, coeff_y, coeff_z = calculate_MST_coeffs(waypoints)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 7)
    for i,_ in enumerate(waypoints[:, 0]):
        ax.plot3D([waypoints[i, 0]], [waypoints[i, 1]], [waypoints[i, 2]], '*')

    time = 0
    yaws = []
    yaws_d = []
    for i in range(300):
        for _ in range(params.num_of_iters_in_batch):
            des_state = generate_trajectory(time, 1.2, waypoints, coeff_x, coeff_y, coeff_z)
            F, M = controller.run(quad, des_state)
            quad.update(params.dt, F, M)
            time += params.dt
        quad_frame = quad.get_quad_motor_xyz_pos()
        res = quad_frame[5, :]
        a = ax.plot3D([res[0]], [res[1]], [res[2]], 'r*')
        # ax.plot3D([quad_frame[0, 0], quad_frame[2, 0]], [quad_frame[0, 1], quad_frame[2, 1]], [quad_frame[0, 2], quad_frame[2, 2]])
        # ax.plot3D([quad_frame[1, 0], quad_frame[3, 0]], [quad_frame[1, 1], quad_frame[3, 1]], [quad_frame[1, 2], quad_frame[3, 2]])
        plt.pause(params.dt)

    plt.savefig('simulation.png')
