import numpy as np
import matplotlib.pyplot as plt
import params
from collections import namedtuple

DesiredState = namedtuple("DesiredState", "pos vel acc yaw yaw_d")
current_direction = np.zeros(2)
yaw = 0


def calculate_helix_waypoints(t, n):
    waypoints = np.linspace(t, t + 2*np.pi, n)
    x = 0.5 * np.cos(waypoints)
    y = 0.5 * np.sin(waypoints)
    z = waypoints

    return np.stack((x, y, z), axis=-1)


def get_poly_cc(n, k, t):
    cc = np.ones(n)
    D = np.linspace(0, n-1, n)

    for i in range(n):
        for j in range(k):
            cc[i] = cc[i] * D[i]
            D[i] -= 1
            if D[i] == -1:
                D[i] = 0

    for i in range(n):
        cc[i] = cc[i] * np.power(t, D[i])

    return cc


def generate_trajectory(t, vel_avg, waypoints, coeff_x, coeff_y, coeff_z):
    global yaw
    global current_direction

    yaw_d = 0.0
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)

    distance_vec = waypoints[0:-1] - waypoints[1:]
    T = (1.0 / vel_avg) * np.sqrt(distance_vec[:, 0]**2 + distance_vec[:, 1]**2 + distance_vec[:, 2]**2)
    T_cum = np.zeros(len(T) + 1)
    T_cum[1:] = np.cumsum(T)
    t_segment = np.where(t >= T_cum)[0][-1]

    if t == 0:
        # initialize
        pos = waypoints[0]
        t0 = get_poly_cc(8, 1, 0)
        current_direction = np.array([coeff_x[0:8].dot(t0), coeff_y[0:8].dot(t0)]) * (1.0 / T[0])
        # current_direction = np.array([-0.00016518,  0.00051137])

    elif t > T_cum[-1]:
        # end
        pos = waypoints[-1]
    else:
        t_scaled = (t - T_cum[t_segment]) / T[t_segment]
        start = t_segment * 8
        end = (t_segment + 1) * 8

        pol_pos = get_poly_cc(8, 0, t_scaled)
        pos = np.array([coeff_x[start:end].dot(pol_pos), coeff_y[start:end].dot(pol_pos), coeff_z[start:end].dot(pol_pos)])

        pol_vel = get_poly_cc(8, 1, t_scaled)
        vel = np.array([coeff_x[start:end].dot(pol_vel), coeff_y[start:end].dot(pol_vel), coeff_z[start:end].dot(pol_vel)]) * (1.0 / T[t_segment])

        pol_acc = get_poly_cc(8, 2, t_scaled)
        acc = np.array([coeff_x[start:end].dot(pol_acc), coeff_y[start:end].dot(pol_acc), coeff_z[start:end].dot(pol_acc)]) * (1.0 / T[t_segment]**2)

        next_direction = vel[0:2]
        delta_psi = np.arccos(current_direction.dot(next_direction) / (np.linalg.norm(current_direction) * np.linalg.norm(next_direction)))
        norm_v = np.cross(current_direction, next_direction)
        prev_yaw = yaw
        if norm_v > 0:
            yaw += delta_psi
        else:
            yaw -= delta_psi

        if yaw > np.pi:
            yaw -= 2 * np.pi
        current_direction = next_direction
        yaw_d = delta_psi / params.dt

    return DesiredState(pos, vel, acc, yaw, yaw_d)


def calculate_MST_coeffs(waypoints):
    """
        Calculates polynomial coefficients for segments between waypoints in x, y, z dimension (3D space).
        For each segment in a single dimension there are 8 coefficients, which means there are 8*N coefficients in total
        for a single dimension. These coefficients are saved in 1D arrays of len 8*N.
    """
    coeff_x = MST(waypoints[:, 0]).flatten()
    coeff_y = MST(waypoints[:, 1]).flatten()
    coeff_z = MST(waypoints[:, 2]).flatten()

    return (coeff_x, coeff_y, coeff_z)


# Minimum Snap Trajectory
def MST(waypoints):
    """ This function takes a list of desired waypoint i.e. [x0, x1, x2...xN] and
    time, returns a [8N,1] coeffitients matrix for the N+1 waypoints.
    1.The Problem
    Generate a full trajectory across N+1 waypoint made of N polynomial line segment.
    Each segment is defined as 7 order polynomial defined as follow:
    Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7
    Each polynomial has 8 unknown coefficients, thus we will have 8*N unknown to
    solve in total, so we need to come up with 8*N constraints.
    2.The constraints
    In general, the constraints is a set of condition which define the initial
    and final state, continuity between each piecewise function. This includes
    specifying continuity in higher derivatives of the trajectory at the
    intermediate waypoints.
    3.Matrix Design
    Since we have 8*N unknown coefficients to solve, and if we are given 8*N
    equations(constraints), then the problem becomes solving a linear equation.
    A * Coeff = B
    Let's look at B matrix first, B matrix is simple because it is just some constants
    on the right hand side of the equation. There are 8xN constraints,
    so B matrix will be [8N, 1].
    Now, how do we determine the dimension of Coeff matrix? Coeff is the final
    output matrix consists of 8*N elements. Since B matrix is only one column,
    thus Coeff matrix must be [8N, 1].
    Coeff.transpose = [a10 a11..a17...aN0 aN1..aN7]
    A matrix is tricky, we then can think of A matrix as a coeffient-coeffient matrix.
    We are no longer looking at a particular polynomial Pi, but rather P1, P2...PN
    as a whole. Since now our Coeff matrix is [8N, 1], and B is [8N, 8N], thus
    A matrix must have the form [8N, 8N].
    A = [A10 A12 ... A17 ... AN0 AN1 ...AN7
         ...
        ]
    Each element in a row represents the coefficient of coeffient aij under
    a certain constraint, where aij is the jth coeffient of Pi with i = 1...N, j = 0...7.
    """

    n = len(waypoints) - 1

    # initialize A, and B matrix
    A = np.zeros((8*n, 8*n))
    B = np.zeros((8*n, 1))

    # populate B matrix.
    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i+1]

    # Constraint 1
    for i in range(n):
        A[i][8*i:8*(i+1)] = get_poly_cc(8, 0, 0)

    # Constraint 2
    for i in range(n):
        A[i+n][8*i:8*(i+1)] = get_poly_cc(8, 0, 1)

    # Constraint 3
    for k in range(1, 4):
        A[2*n+k-1][:8] = get_poly_cc(8, k, 0)

    # Constraint 4
    for k in range(1, 4):
        A[2*n+3+k-1][-8:] = get_poly_cc(8, k, 1)

    # Constraint 5
    for i in range(n-1):
        for k in range(1, 7):
            A[2*n+6 + i*6+k-1][i*8 : (i*8+16)] = np.concatenate((get_poly_cc(8, k, 1), -get_poly_cc(8, k, 0)))

    # solve for the coefficients
    Coeff = np.linalg.solve(A, B)

    return Coeff


