from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
from simulation_code import simulate

# Setting weights
Q_X = 7
Q_Y = 9
Q_theta = 4
R_w1 = 0.1
R_w2 = 0.1
R_w3 = 0.1
R_w4 = 0.1

h = 0.1  # Sampling time[s]
N = 25  # prediction horizon

rob_diam = 0.3  # Robot diameter
Rw = 0.1  # Wheel radious
Lx = 0.2  # Distance from wheel to the centre of robot on X axis
Ly = 0.1  # Distance from wheel to the centre of robot on Y axis

omega_max = pi
omega_min = -omega_max
# Initialization
# Robot
x_init = 0
y_init = 0
theta_init = 0

# Target

x_target = 10
y_target = 10
theta_target = pi / 2

# Obstcale1

x_obstcale1 = 5
y_obstcale1 = 3
r_obstacle1 = 1

# Obstcale2

x_obstcale2 = 3
y_obstcale2 = 5
r_obstacle2 = 1

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
# Q = ca.zeros(n_states, n_states)
# Q[1, 1] = Q_X, Q[2, 2] = Q_Y, Q[3, 3] = Q_theta  # Weighing matrices (states)
# R = ca.zeros(n_controls, n_controls)
# R[1, 1] = R_w1, R[2, 2] = R_w2, R[3, 3] = R_w3, R[4, 4] = R_w4  # Weighing matrices (controls)
Q = ca.diagcat(Q_X, Q_Y, Q_theta)
R = ca.diagcat(R_w1, R_w2, R_w3, R_w4)
obj = 0  # Objective function
# g = []  # Constraints vector
# st = X[:, 0]  # Initial state
# g = [g, X[:, 0] - P[:n_states]]  # Initial condition constraints
# g = X[:, 0] - P[:n_states]
g = X[:, 0] - P[:n_states]  # Constraints vector
print(X)
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

for M in range(N + 1):
    g = ca.vertcat(g, -np.sqrt((X[0:1, M] - x_obstcale1) ** 2 + ((X[1:2, M] - y_obstcale1) ** 2)) + (
                rob_diam / 2) + r_obstacle1)
for M in range(N + 1):
    g = ca.vertcat(g, -np.sqrt((X[0:1, M] - x_obstcale2) ** 2 + ((X[1:2, M] - y_obstcale2) ** 2)) + (
                rob_diam / 2) + r_obstacle2)
    # Implementing the constrains of the obstacle by the equation -sqrt(y-y0)^2+(x-x0)^2)+robot_radius+obstcale_radius <= 0

# OPT_variables = [X.reshape((n_states * (N + 1), 1)), U.reshape((n_controls * N, 1))]
OPT_variables = ca.vertcat(X.reshape((n_states * (N + 1), 1)), U.reshape((n_controls * N, 1)))
# OPT_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
opts = {'ipopt': {'max_iter': 2000, 'print_level': 0, 'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6},
        'print_time': 0}  # print level = 0.3
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbg = ca.DM.zeros((n_states * (N + 1) + 2*(N + 1), 1))
ubg = ca.DM.zeros((n_states * (N + 1) + 2*(N + 1), 1))
lbg[n_states * (N + 1):] = -ca.inf
# setting the obstacle constrains to be lower than 0 taking the last 21 constrain from g
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

# Simulation loop
# Initializing
t0 = 0  # setting time to zero
x0 = ca.DM([x_init, y_init, theta_init])  # initial state
# Setting the robot initial coordinates
xs = ca.DM([x_target, y_target, theta_target])  # target state
# Setting the target initial (constant) coordinates
# xx = ca.DM(x0)
t = ca.DM(t0)

# u0 = ca.DM.zeros((n_controls, N))  # initial control
u0 = ca.DM.zeros((N, n_controls))  # initial control
X0 = ca.repmat(x0, 1, N + 1)  # initial state full
sim_time = 60  # Maximum simulation time
# Start MPC
mpciter = 0
# xx1 = []
# u_cl = []
cat_states = np.array(X0.full())
cat_controls = np.array(u0[:, 0].full())
times = np.array([[0]])


def shift(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0


if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (ca.norm_2(x0 - xs) > 1e-2) and (mpciter < (
            sim_time / h)):  # condition for the loop to end when the robot approach the target or when the simulation time reach its maximum
        t1 = time()
        # args['p'] = [x0, xs]
        args['p'] = ca.vertcat(x0, xs)
        # put the 6 initial states in the paramters list
        # set the values of the parameters vector
        # initial value of the optimization variables
        args['x0'] = ca.vertcat(ca.reshape(X0, n_states * (N + 1), 1), ca.reshape(u0, n_controls * N, 1))
        # filling the values that will be optimized
        sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'], p=args['p'])
        # calling the solver with the constrains , the optimize list and the parameters
        # u = ca.reshape(sol['x'][n_states * (N + 1)+1:], n_controls, N)
        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N + 1)], n_states, N + 1)
        # updating the u and X0 matricies and reshaping it to be valid in optimizing
        cat_states = np.dstack((cat_states, np.array(X0.full())))
        # xx1[:, 1: 3, mpciter + 1] = ca.reshape(sol['x'][: n_states * (N + 1)], n_states, N + 1)
        # u_cl = [u_cl, u[:, 0]]
        # t[mpciter + 1] = t0
        cat_controls = np.vstack((cat_controls, np.array((u[:, 0]).full())))
        t = np.vstack((t, t0))
        #getting the solution from the solver for states and control
        # Apply the control and shift the solution
        t0, x0, u0 = shift(h, t0, x0, u, f)
        # shifting to the new values for next iteration
        X0 = ca.horzcat(X0[:, 1:], ca.reshape(X0[:, -1], -1, 1))
        # xx[:, mpciter + 2] = x0
        t2 = time()
        print(mpciter)
        #counter
        print(t2 - t1)
        #printing the time for one iteration
        times = np.vstack((times, t2 - t1))
        #calculating the time
        # X0 = ca.reshape(sol['x'][: n_states * (N + 1)], n_states, N + 1)
        # X0 = ca.horzcat(X0[:, 1:],ca.reshape(X0[:, -1], -1, 1))
        # Shift trajectory to initialize the next step

        # X0 = X0[:, 1:], ca.reshape(X0[:, -1], -1, 1)
        mpciter = mpciter + 1
        #updating the counter
    main_loop_time = time()
    ss_error = ca.norm_2(x0 - xs)
    #calculating the remaining distance to reach the target
    average_mpc_time = main_loop_time / (mpciter + 1)
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

    # simulate
    simulate(cat_states, cat_controls, times, h, N,
             np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), x_obstcale1, y_obstcale1,
             r_obstacle1, x_obstcale2, y_obstcale2,
             r_obstacle2, save=False)
