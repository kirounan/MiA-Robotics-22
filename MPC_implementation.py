import casadi as ca
from casadi import sin, cos, pi

# Setting weights
Q_X = 9
Q_Y = 7
Q_theta = 1
R_w1 = 0.05
R_w2 = 0.05
R_w3 = 0.05
R_w4 = 0.05

h = 0.2  # Sampling time[s]
N = 20  # prediction horizon

rob_diam = 0.3  # Robot diameter
Rw = 0.1  # Wheel radious
Lx = 0.2  # Distance from wheel to the centre of robot on X axis
Ly = 0.1  # Distance from wheel to the centre of robot on Y axis

omega_max = pi
omega_min = -omega_max

x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.numel()

omega_w1 = ca.SX.sym('omega_w1')
omega_w2 = ca.SX.sym('omega_w2')
omega_w3 = ca.SX.sym('omega_w3')
omega_w4 = ca.SX.sym('omega_w4')
controls = ca.vertcat(omega_w1, omega_w2, omega_w3, omega_w4)
n_controls = controls.numel()

# Discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)

j0_plus = (Rw / 4) * ca.DM([
    [1, 1, 1, 1],
    [-1, 1, 1, -1],
    [-1 / (Lx + Ly), 1 / (Lx + Ly), -1 / (Lx + Ly), 1 / (Lx + Ly)]
])
rot_3d_z = ca.vertcat(ca.horzcat(cos(theta), -sin(theta), 0),
            ca.horzcat(sin(theta), cos(theta), 0),
            ca.horzcat(0, 0, 1))
# rot_3d_z = ca.vertcat(
#    ca.horzcat(cos(theta), -sin(theta), 0),
#   ca.horzcat(sin(theta),  cos(theta), 0),
#   ca.horzcat(0, 0, 1)
# )
# RHS = rot_3d_z @ j0_plus @ controls
rhs = rot_3d_z @ j0_plus @ controls

f = ca.Function('f', [states, controls], [rhs])
X = ca.SX.sym('X', n_states, (N + 1))  # States vector
U = ca.SX.sym('U', n_controls, N)  # Decision variables (controls)
P = ca.SX.sym('P', n_states + n_states)  # Parameters
#Q = ca.zeros(n_states, n_states)
#Q[1, 1] = Q_X, Q[2, 2] = Q_Y, Q[3, 3] = Q_theta  # Weighing matrices (states)
#R = ca.zeros(n_controls, n_controls)
#R[1, 1] = R_w1, R[2, 2] = R_w2, R[3, 3] = R_w3, R[4, 4] = R_w4  # Weighing matrices (controls)
Q = ca.diagcat(Q_X, Q_Y, Q_theta)
R = ca.diagcat(R_w1, R_w2, R_w3, R_w4)
obj = 0  # Objective function
#g = []  # Constraints vector
#st = X[:, 0]  # Initial state
# g = [g, X[:, 0] - P[:n_states]]  # Initial condition constraints
# g = X[:, 0] - P[:n_states]
g = X[:, 0] - P[:n_states] # Constraints vector
# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    # obj = obj + (st - P[4:6]) @ Q @ (st - P[4:6]) + con @ R @ con
    # obj = obj + (st - P[4:6]).T @ Q @ (st - P[4:6]) + con.T @ R @ con
    obj = obj + (st - P[n_states:]).T @ Q @ (st - P[n_states:]) + con.T @ R @ con
    st_next = X[:, k + 1]
    k1 = f(st, con)
    k2 = f(st + h / 2 * k1, con)
    k3 = f(st + h / 2 * k2, con)
    k4 = f(st + h * k3, con)
    st_next_RK4 = st + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    # g = [g, st_next - st_next_RK4]
    g = ca.vertcat(g, st_next - st_next_RK4)

# OPT_variables = [X.reshape((n_states * (N + 1), 1)), U.reshape((n_controls * N, 1))]
OPT_variables = ca.vertcat(X.reshape((n_states * (N + 1), 1)), U.reshape((n_controls * N, 1)))
# OPT_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
opts = {'ipopt': {'max_iter': 2000, 'print_level': 0, 'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6},
        'print_time': 0}  # print level = 0.3
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbg = ca.DM.zeros((n_states * (N + 1), 1))
ubg = ca.DM.zeros((n_states * (N + 1), 1))
lbx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))
ubx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))

lbx[0: n_states * (N + 1): n_states] = -ca.inf  # X lower bound
ubx[0: n_states * (N + 1): n_states] = ca.inf  # X upper bound
lbx[1: n_states * (N + 1): n_states] = -ca.inf  # Y lower bound
ubx[1: n_states * (N + 1): n_states] = ca.inf  # Y upper bound
lbx[2: n_states * (N + 1): n_states] = -ca.inf  # theta lower bound
ubx[2: n_states * (N + 1): n_states] = ca.inf  # theta upper bound

lbx[n_states * (N + 1):] = omega_min  # v lower bound for all V
ubx[n_states * (N + 1):] = omega_max  # v upper bound for all V
args = {
    'lbg': lbg,  # constraints lower bound
    'ubg': ubg,  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}
# Notes
# to create matrices in python casadi we can do it with multiple ways :
# using vertcat , diagcat , hortzcat or using ca.DM and writing the matrix
# like matlab in this part there is no much big difference between matlab and
# python we first declared our objects and weights , created states and control
# matrices , initial parameters for robot and target matrix , tuning matrix with
# Q and R containing the tuning weights declared before
# Then we formed the codt function using the runge kutta method and we created
# constrains matrix g , we put all our states and controls in one big 1D matrix
# (OPT) then finally declaring the optimizers , initializing the constrains
# and solve the problem with casadi function
