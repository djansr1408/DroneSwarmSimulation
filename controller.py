from Quadcopter import Quadcopter
import params
import numpy as np
from math import sin, cos


def run(quad, des_state):
    """
        Controller is used to calculate output of a PD controller (F and M) based on the
        error between current state and desired state.
    """
    x, y, z = quad.get_position()
    x_d, y_d, z_d = quad.get_velocity()
    phi, theta, psi = quad.get_attitude()
    p, q, r = quad.get_pqr()

    des_x, des_y, des_z = des_state.pos
    des_x_d, des_y_d, des_z_d = des_state.vel
    des_x_dd, des_y_dd, des_z_dd = des_state.acc

    des_psi = des_state.yaw
    des_psi_d = des_state.yaw_d

    # Commanded accelerations in x, y, z
    commanded_x_dd = des_x_dd + params.kd_x * (des_x_d - x_d) + params.kp_x * (des_x - x)
    commanded_y_dd = des_y_dd + params.kd_y * (des_y_d - y_d) + params.kp_y * (des_y - y)
    commanded_z_dd = des_z_dd + params.kd_z * (des_z_d - z_d) + params.kp_z * (des_z - z)

    # Thrust
    F = params.mass * (params.g + commanded_z_dd)

    # Moment
    des_p = 0
    des_q = 0
    des_r = des_psi_d
    des_phi = 1 / params.g * (commanded_x_dd * sin(des_psi) - commanded_y_dd * cos(des_psi))
    des_theta = 1 / params.g * (commanded_x_dd * cos(des_psi) + commanded_y_dd * sin(des_psi))

    M = np.array([[params.kp_phi * (des_phi - phi) + params.kd_phi * (des_p - p), \
                  params.kp_theta * (des_theta - theta) + params.kd_theta * (des_q - q), \
                  params.kp_psi * (des_psi - psi) + params.kd_psi * (des_r - r)]]).T  # transpose to get 3x1 shape

    return F, M

