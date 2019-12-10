import numpy as np
import scipy.integrate as integrate
from utils import *
import params


class Quadcopter(object):
    """
        Quadcopter class which contains description of a drone, with current states and velocities.
    """
    def __init__(self, state, L, H):
        self.state = state

        self.L = L
        self.H = H

        # Body frame contain 6 important dots on a drone, it is 6x4 matrix
        self.body_frame = np.array([[self.L, 0, 0, 1], [0, self.L, 0, 1], [-self.L, 0, 0, 1], [0, -self.L, 0, 1],\
                                    [0, 0, 0, 1], [0, 0, self.H, 1]])

    def get_position(self):
        return self.state[0:3]

    def get_velocity(self):
        return self.state[3:6]

    def get_attitude(self):
        R = Quat2Rot(self.state[6:10])  # bRw
        phi, theta, psi = Rot2RPY(R)
        return phi, theta, psi

    def get_pqr(self):
        return self.state[10:13]

    def get_quad_motor_xyz_pos(self):
        """
            Calculates position of drone's crucial (motor) points in xyz (world) coord. system.
            Result is 6x3 matrix.
        """
        wRb = Quat2Rot(self.state[6:10]).T

        wHb = np.zeros((4, 4))
        wHb[:3, :3] = wRb
        wHb[:3, 3] = self.state[0:3]
        wHb[3, 3] = 1

        quad_world_frame = wHb.dot(self.body_frame.T)
        return quad_world_frame[:3, :].T

    def state_d(self, state, t, F, M):
        x, y, z, x_d, y_d, z_d, qw, qx, qy, qz, p, q, r = state
        quat = np.array([qw, qx, qy, qz])
        bRw = Quat2Rot(quat)
        wRb = bRw.T

        accel = 1 / params.mass * (wRb.dot(np.array([[0, 0, F]]).T) - np.array([[0, 0, params.mass * params.g]]).T)

        K_quat = 2
        quaterror = 1 - (qw**2 + qx**2 + qy**2 + qz**2)
        qdot = (-1.0 / 2) * np.array([[0, -p, -q, -r],\
                                [p, 0, -r, q],\
                                [q, r, 0, -p],\
                                [r, -q, p, 0]]).dot(quat) + K_quat * quaterror * quat
        omega = np.array([p, q, r])
        pqrdot = params.invI.dot(M.flatten() - np.cross(omega, params.I.dot(omega)))  # (3, )

        s_d = np.zeros(13)
        s_d[0] = x_d
        s_d[1] = y_d
        s_d[2] = z_d
        s_d[3] = accel[0]
        s_d[4] = accel[1]
        s_d[5] = accel[2]
        s_d[6] = qdot[0]
        s_d[7] = qdot[1]
        s_d[8] = qdot[2]
        s_d[9] = qdot[3]
        s_d[10] = pqrdot[0]
        s_d[11] = pqrdot[1]
        s_d[12] = pqrdot[2]

        return s_d

    def update(self, dt, F, M):
        # Limit propellers thrusts
        prop_thrusts = params.invA.dot(np.r_[[[F]], M])
        prop_thrusts_clumped = np.maximum(np.minimum(prop_thrusts, params.maxF/4), params.minF/4)
        F = np.sum(prop_thrusts_clumped)
        M = params.A[1:, :].dot(prop_thrusts_clumped)
        self.state = integrate.odeint(self.state_d, self.state, [0, dt], args=(F, M))[1]










