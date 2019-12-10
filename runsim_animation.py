from utils import *
from Quadcopter import Quadcopter
import controller
from traj_generator import *
import params

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == "__main__":
    state_0 = init_state(np.array([0.5, 0, 0]), yaw=0.0)
    quad = Quadcopter(state_0, params.L, params.H)
    waypoints = calculate_helix_waypoints(0, 9)
    coeff_x, coeff_y, coeff_z = calculate_MST_coeffs(waypoints)

    fig = plt.figure(figsize=(25, 25))
    ax = plt.axes(projection='3d')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 7)
    for i,_ in enumerate(waypoints[:, 0]):
        ax.plot3D([waypoints[i, 0]], [waypoints[i, 1]], [waypoints[i, 2]], "*")

    time = [0]
    yaws = []
    yaws_d = []
    des_path = []

    t = 0
    for _ in range(1500):
        des_state = generate_trajectory(t, 1.2, waypoints, coeff_x, coeff_y, coeff_z)
        des_path.append(des_state.pos)
        t += params.dt
    des_path = np.array(des_path)


    def run_update(i):
        for _ in range(params.num_of_iters_in_batch):
            des_state = generate_trajectory(time[0], 1.2, waypoints, coeff_x, coeff_y, coeff_z)
            F, M = controller.run(quad, des_state)
            quad.update(params.dt, F, M)
            time[0] += params.dt
        frame = quad.get_quad_motor_xyz_pos()
        lines = ax.get_lines()
        lines_data = [frame[[0, 2], :], frame[[1, 3], :], frame[[4, 5], :]]
        lines = ax.get_lines()
        for line, line_data in zip(lines[:3], lines_data):
            line_data = np.array(line_data)
            x, y, z = line_data.T
            line.set_data(x, y)
            line.set_3d_properties(z)
        return lines

    ani = animation.FuncAnimation(fig, run_update, frames=20, init_func=None,
                                  interval=25, blit=False)

    # for i,_ in enumerate(waypoints[:, 0]):
    #     ax.plot3D([waypoints[i, 0]], [waypoints[i, 1]], [waypoints[i, 2]], 'r*')
    ax.plot3D(des_path[:, 0], des_path[:, 1], des_path[:, 2], 'y.', linewidth=1)
    plt.show()

    ani.save('sim.gif', dpi=80, writer='imagemagick', fps=60)

