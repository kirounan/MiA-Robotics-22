from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
from simulation_code import simulate

# Setting weights
Q_X = 10
Q_Y = 7
Q_theta = 1
R_W = 0.05
R_V = 0.05

h = 0.1  # Sampling time[s]
N = 20  # prediction horizon

rob_diam = 0.3  # Robot diameter
Rw = 0.1  # Wheel radius
Lx = 0.2  # Distance from wheel to the centre of robot on X axis
Ly = 0.1  # Distance from wheel to the centre of robot on Y axis

omega_max = 0.3
omega_min = -omega_max
v_max = 1
v_min = -v_max
# Initialization
# Robot
x_init = 0
y_init = 0
theta_init = 0

# Target

x_target = 15
y_target = 15
theta_target = pi
# States
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.numel()
# Controls
omega = ca.SX.sym('omega')
v = ca.SX.sym('v')
controls = ca.vertcat(v, omega)
n_controls = controls.numel()

# Differential model (transfer function)
rhs = ca.vertcat(v*cos(theta), v*sin(theta), omega)

f = ca.Function('f', [states, controls], [rhs])  # Mapping function implementation
X = ca.SX.sym('X', n_states, (N + 1))  # States vector
U = ca.SX.sym('U', n_controls, N)  # Decision variables (controls)
P = ca.SX.sym('P', n_states + n_states)  # Parameters
# Weights for control and states
Q = ca.diagcat(Q_X, Q_Y, Q_theta)
R = ca.diagcat(R_W, R_V)

obj = 0  # Objective function

g = X[:, 0] - P[:n_states]  # Constraints vector
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    obj = obj + (st - P[n_states:]).T @ Q @ (st - P[n_states:]) + con.T @ R @ con
    st_next = X[:, k + 1]
    k1 = f(st, con)
    k2 = f(st + h / 2 * k1, con)
    k3 = f(st + h / 2 * k2, con)
    k4 = f(st + h * k3, con)
    st_next_RK4 = st + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - st_next_RK4)
OPT_variables = ca.vertcat(X.reshape((n_states * (N + 1), 1)), U.reshape((n_controls * N, 1)))

nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
opts = {'ipopt': {'max_iter': 2000, 'print_level': 0, 'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6},
        'print_time': 0}  # print level = 0.3
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbg = ca.DM.zeros((n_states * (N + 1), 1))
ubg = ca.DM.zeros((n_states * (N + 1), 1))

lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

lbx[0: n_states*(N+1): n_states] = -ca.inf     # X lower bound
lbx[1: n_states*(N+1): n_states] = -ca.inf     # Y lower bound
lbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound

ubx[0: n_states*(N+1): n_states] = ca.inf      # X upper bound
ubx[1: n_states*(N+1): n_states] = ca.inf      # Y upper bound
ubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound
for i in range (N):
    lbx[n_states * (N + 1)+2*i] = v_min  # v lower bound
    ubx[n_states * (N + 1)+2*i] = v_max  # v upper bound
    lbx[n_states * (N + 1)+2*i+1] = omega_min  # w lower bound
    ubx[n_states * (N + 1)+2*i+1] = omega_max  # w upper bound

args = {
    'lbg': lbg,  # constraints lower bound
    'ubg': ubg,  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}

# Simulation loop
def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())

t0 = 0
state_init = ca.DM([x_init, y_init, theta_init])        # initial state
state_target = ca.DM([x_target, y_target, theta_target])  # target state

# xx = DM(state_init)
t = ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(state_init, 1, N+1)         # initial state full


mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])
sim_time = 100

###############################################################################

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * h < sim_time):
        t1 = time()
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_target   # target state
        )
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))

        t0, state_init, u0 = shift_timestep(h, t0, state_init, u, f)

        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        # xx ...
        t2 = time()
        print(mpc_iter)
        print(t2-t1)
        times = np.vstack((
            times,
            t2-t1
        ))

        mpc_iter = mpc_iter + 1

    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_target)

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

    # simulate
    simulate(cat_states, cat_controls, times, h, N,
             np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), save=False)

