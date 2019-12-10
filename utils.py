import numpy as np
from math import sin, cos, asin, atan2, sqrt


def init_state(start_pos, yaw):
    """
        Set initial state of a drone.
        Args: start_pos is array of len 3
    """
    state = np.zeros(13)
    phi0 = 0
    theta0 = 0
    psi0 = yaw
    bRw = RPY2Rot(phi0, theta0, psi0)
    quat0 = Rot2Quat(bRw)
    state[0] = start_pos[0]  # x
    state[1] = start_pos[1]  # y
    state[2] = start_pos[2]  # z
    state[3] = 0  # xd
    state[4] = 0  # yd
    state[5] = 0  # zd
    state[6] = quat0[0]  # qw
    state[7] = quat0[1]  # qx
    state[8] = quat0[2]  # qy
    state[9] = quat0[3]  # qz
    state[10] = 0  # p
    state[11] = 0  # q
    state[12] = 0  # r
    return state


def RPY2Rot(phi, theta, psi):
    """Converts roll, pitch, yaw to a world-to-body (bRw) rotation matrix.
        In case you need body-to-world rotation matrix, just transpose this one.
    """
    R = [[np.cos(psi)*np.cos(theta) - np.sin(phi)*np.sin(psi)*np.sin(theta),
          np.cos(theta)*np.sin(psi) + np.cos(psi)*np.sin(phi)*np.sin(theta),
          -np.cos(phi)*np.sin(theta)],
         [-np.cos(phi)*np.sin(psi), np.cos(phi) * np.cos(psi), np.sin(phi)],
         [np.cos(psi)*np.sin(theta) + np.cos(theta)*np.sin(phi)*np.sin(psi),
          np.sin(psi)*np.sin(theta) - np.cos(psi)*np.cos(theta)*np.sin(phi),
          np.cos(phi)*np.cos(theta)]]
    return np.array(R)


def Rot2RPY(R):
    """
        Extracts roll, pitch, yaw from world-to-body (bRw) rotation matrix.
    """
    phi = asin(R[1, 2])
    psi = atan2(-R[1, 0] / cos(phi), R[1, 1] / cos(phi))
    theta = atan2(-R[0, 2] / cos(phi), R[2, 2] / cos(phi))
    return phi, theta, psi


def Rot2Quat(R):
    """
        Convert rotation matrix R(3x3) to quaternion.
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = sqrt(tr+1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 1] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.sign(qw) * np.array([qw, qx, qy, qz])
    return q


def Quat2Rot(quat):
    """
        Returns rotation matrix from quaternion.
    """
    q = quat / np.sqrt(np.sum(quat**2))
    qahat = np.zeros((3, 3))
    qahat[0, 1] = -q[3]
    qahat[0, 2] = q[2]
    qahat[1, 2] = -q[1]
    qahat[1, 0] = q[3]
    qahat[2, 0] = -q[2]
    qahat[2, 1] = q[1]

    R = np.eye(3, 3) + 2 * np.dot(qahat, qahat) + 2 * q[0] * qahat

    return R
