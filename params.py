import numpy as np



CONTROL_FREQUENCY = 200  # Hz
ANIMATION_FERQUENCY = 50
num_of_iters_in_batch = int(CONTROL_FREQUENCY / ANIMATION_FERQUENCY)
dt = 1.0 / CONTROL_FREQUENCY

Kf = 6.11 * 1e-8
Km = 1.5 * 1e-9

kd_x = 20
kp_x = 80

kd_y = 20
kp_y = 80

kd_z = 25
kp_z = 100


kd_phi = 5
kp_phi = 40

kd_theta = 5
kp_theta = 40

kd_psi = 5
kp_psi = 40


mass = 0.18
g = 9.81
minF = 0.0
maxF = 2.0 * mass * g

L = 0.086
H = 0.05
km = 1.5e-9
kf = 6.11e-8
r = km / kf
A = np.array([[1, 1, 1, 1],
              [0, L, 0, -L],
              [-L, 0, L, 0],
              [r, -r, r, -r]])

# F                 F1
# M1   =    A *     F2
# M2                F3
# M3                F4

invA = np.linalg.inv(A)


I = np.array([(0.00025, 0, 2.55e-6),
              (0, 0.000232, 0),
              (2.55e-6, 0, 0.0003738)])

invI = np.linalg.inv(I)

# A = np.array([[0.25, 0, -0.5/L], [0.25, 0.5/L, 0], [0.25, 0, 0.5/L], [0.25, -0.5/L, 0]])

B = np.array([[1, 1, 1, 1], [0, L, 0, -L], [-L, 0, L, 0]])

