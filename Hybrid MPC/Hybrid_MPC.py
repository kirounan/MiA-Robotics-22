# Hybrid MPC model mix of Mecanum and differential MPC models for better performance
from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
from simulation_code_1obs import simulate

# Setting weights
Q_X = 10
Q_Y = 7
Q_theta = 1
# Mecanum weights
R_w1 = 0.05
R_w2 = 0.05
R_w3 = 0.05
R_w4 = 0.05
# Differential weights
R_W = 0.001
R_V = 0.005

h = 0.1  # Sampling time[s]
N = 20  # prediction horizon

rob_diam = 0.3  # Robot diameter
Rw = 0.1  # Wheel radius
Lx = 0.2  # Distance from wheel to the centre of robot on X axis
Ly = 0.1  # Distance from wheel to the centre of robot on Y axis
# Mecanum and differential speed limits
v_max = 1
v_min = -v_max
# differential rotational limits
omega_max = pi/4
omega_min = -omega_max
# Initialization
# Robot
x_init = 0
y_init = 0
theta_init = pi/2


# Obstacles
#obs =[6, 3, 1, 8, 4, 2, 5, 2, 1, 6, 6, 1]


# Obstcale1

x_obstcale = 14
y_obstcale = 14
r_obstacle = 1


# Target
if(x_obstcale>x_init):
    x_target = x_obstcale - r_obstacle
else:
    x_target = x_obstcale + r_obstacle
if(y_obstcale>y_init):
    y_target = y_obstcale - r_obstacle
else:
    y_target = y_obstcale + r_obstacle

theta_target = np.arctan((y_target-y_init)/(x_target-x_init))
if theta_target < 0:
    if y_target > y_init:
        theta_target=theta_target+pi
else:
    if y_init > y_target:
        theta_target=theta_target+pi
# Obstcale2

#x_obstcale2 = 3
#y_obstcale2 = 5
#r_obstacle2 = 1

x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.numel()

omega_w1 = ca.SX.sym('omega_w1')
omega_w2 = ca.SX.sym('omega_w2')
omega_w3 = ca.SX.sym('omega_w3')
omega_w4 = ca.SX.sym('omega_w4')
Mecanum_controls = ca.vertcat(omega_w1, omega_w2, omega_w3, omega_w4)
n_Mcontrols = Mecanum_controls.numel()

# Differential Model control
omega = ca.SX.sym('omega')
v = ca.SX.sym('v')
Differential_controls = ca.vertcat(v, omega)
n_Dcontrols = Differential_controls.numel()

# Discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
#Mecanum
j0_plus = (Rw / 4) * ca.DM([
    [1, 1, 1, 1],
    [-1, 1, 1, -1],
    [-1 / (Lx + Ly), 1 / (Lx + Ly), -1 / (Lx + Ly), 1 / (Lx + Ly)]
])
rot_3d_z = ca.vertcat(ca.horzcat(cos(theta), -sin(theta), 0),
                      ca.horzcat(sin(theta), cos(theta), 0),
                      ca.horzcat(0, 0, 1))
Mrhs = rot_3d_z @ j0_plus @ Mecanum_controls
#differential
Drhs = ca.vertcat(v*cos(theta), v*sin(theta), omega)
#########################################################################################################################
Mf = ca.Function('f', [states, Mecanum_controls], [Mrhs]) #function for Mecanum
Df = ca.Function('f', [states, Differential_controls], [Drhs]) #function for Differential
MX = ca.SX.sym('X', n_states, (N + 1))  # States vector for Mecanum
DX = ca.SX.sym('X', n_states, (N + 1))  # States vector for Differential
MU = ca.SX.sym('U', n_Mcontrols, N)  # Decision variables (controls) Mecanum
DU = ca.SX.sym('U', n_Dcontrols, N) # Decision variables (controls) Differential
MP = ca.SX.sym('P', n_states + n_states)  # Parameters for Mecanum
DP = ca.SX.sym('P', n_states + n_states)  # Parameters for differential
Q = ca.diagcat(Q_X, Q_Y, Q_theta)
MR = ca.diagcat(R_w1, R_w2, R_w3, R_w4)
DR = ca.diagcat(R_W, R_V)
Mobj = 0  # Objective function for Mecanum
Dobj = 0 # Objective function for Differential
Mg = MX[:, 0] - MP[:n_states]  # Constraints vector
Dg = DX[:, 0] - DP[:n_states]  # Constraints vector
# runge kutta for Mecanum
for k in range(N):
    Mst = MX[:, k]
    Mcon = MU[:, k]
    Mobj = Mobj + (Mst - MP[n_states:]).T @ Q @ (Mst - MP[n_states:]) + Mcon.T @ MR @ Mcon
    Mst_next = MX[:, k + 1]
    Mk1 = Mf(Mst, Mcon)
    Mk2 = Mf(Mst + h / 2 * Mk1, Mcon)
    Mk3 = Mf(Mst + h / 2 * Mk2, Mcon)
    Mk4 = Mf(Mst + h * Mk3, Mcon)
    Mst_next_RK4 = Mst + (h / 6) * (Mk1 + 2 * Mk2 + 2 * Mk3 + Mk4)
    Mg = ca.vertcat(Mg, Mst_next - Mst_next_RK4)
#obs_number = int(N_obs/3)
#for h in range(obs_number):
  #  for M in range(N + 1):
     #   Mg = ca.vertcat(Mg, -np.sqrt((MX[0:1, M] - obs[3*h]) ** 2 + ((MX[1:2, M] - obs[3*h+1]) ** 2)) + (
        #        rob_diam / 2) + obs[3*h+2])
# Implementing the constrains of the obstacle by the equation -sqrt(y-y0)^2+(x-x0)^2)+robot_radius+obstacle_radius <= 0
# Runge Kutta for differential
for p in range(N):
    Dst = DX[:, p]
    Dcon = DU[:, p]
    Dobj = Dobj + (Dst - DP[n_states:]).T @ Q @ (Dst - DP[n_states:]) + Dcon.T @ DR @ Dcon
    Dst_next = DX[:, p + 1]
    Dk1 = Df(Dst, Dcon)
    Dk2 = Df(Dst + h / 2 * Dk1, Dcon)
    Dk3 = Df(Dst + h / 2 * Dk2, Dcon)
    Dk4 = Df(Dst + h * Dk3, Dcon)
    Dst_next_RK4 = Dst + (h / 6) * (Dk1 + 2 * Dk2 + 2 * Dk3 + Dk4)
    Dg = ca.vertcat(Dg, Dst_next - Dst_next_RK4)



MOPT_variables = ca.vertcat(MX.reshape((n_states * (N + 1), 1)), MU.reshape((n_Mcontrols * N, 1)))
DOPT_variables = ca.vertcat(DX.reshape((n_states * (N + 1), 1)), DU.reshape((n_Dcontrols * N, 1)))
Mnlp_prob = {'f': Mobj, 'x': MOPT_variables, 'g': Mg, 'p': MP}
Dnlp_prob = {'f': Dobj, 'x': DOPT_variables, 'g': Dg, 'p': DP}
opts = {'ipopt': {'max_iter': 2000, 'print_level': 0, 'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6},
        'print_time': 0}  # print level = 0.3
Msolver = ca.nlpsol('solver', 'ipopt', Mnlp_prob, opts)
Dsolver = ca.nlpsol('solver', 'ipopt', Dnlp_prob, opts)

# Mecanum constrains

#Mlbg = ca.DM.zeros((n_states * (N + 1) + obs_number*(N + 1), 1))
#Mubg = ca.DM.zeros((n_states * (N + 1) + obs_number*(N + 1), 1))
Mlbg = ca.DM.zeros((n_states * (N + 1), 1))
Mubg = ca.DM.zeros((n_states * (N + 1), 1))
Mlbg[n_states * (N + 1):] = -ca.inf
# setting the obstacle constrains to be lower than 0 taking the last 21 constrain from g
Mlbx = ca.DM.zeros((n_states * (N + 1) + n_Mcontrols * N, 1))
Mubx = ca.DM.zeros((n_states * (N + 1) + n_Mcontrols * N, 1))

Mlbx[0: n_states * (N + 1): n_states] = -ca.inf  # X lower bound
Mubx[0: n_states * (N + 1): n_states] = ca.inf  # X upper bound
Mlbx[1: n_states * (N + 1): n_states] = -ca.inf  # Y lower bound
Mubx[1: n_states * (N + 1): n_states] = ca.inf  # Y upper bound
Mlbx[2: n_states * (N + 1): n_states] = -ca.inf  # theta lower bound
Mubx[2: n_states * (N + 1): n_states] = ca.inf  # theta upper bound

Mlbx[n_states * (N + 1):] = v_min  # v lower bound for all V
Mubx[n_states * (N + 1):] = v_max  # v upper bound for all V
Margs = {
    'lbg': Mlbg,  # constraints lower bound
    'ubg': Mubg,  # constraints upper bound
    'lbx': Mlbx,
    'ubx': Mubx
}
# Differential constrains
Dlbg = ca.DM.zeros((n_states * (N + 1), 1))
Dubg = ca.DM.zeros((n_states * (N + 1), 1))

Dlbx = ca.DM.zeros((n_states*(N+1) + n_Dcontrols*N, 1))
Dubx = ca.DM.zeros((n_states*(N+1) + n_Dcontrols*N, 1))

Dlbx[0: n_states*(N+1): n_states] = -ca.inf     # X lower bound
Dlbx[1: n_states*(N+1): n_states] = -ca.inf     # Y lower bound
Dlbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound

Dubx[0: n_states*(N+1): n_states] = ca.inf      # X upper bound
Dubx[1: n_states*(N+1): n_states] = ca.inf      # Y upper bound
Dubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound
for i in range (N):
    Dlbx[n_states * (N + 1)+2*i] = v_min  # v lower bound
    Dubx[n_states * (N + 1)+2*i] = v_max  # v upper bound
    Dlbx[n_states * (N + 1)+2*i+1] = omega_min  # w lower bound
    Dubx[n_states * (N + 1)+2*i+1] = omega_max  # w upper bound

Dlbx[n_states * (N + 1):] = v_min  # v lower bound for all V
Dubx[n_states * (N + 1):] = v_max  # v upper bound for all V
Dargs = {
    'lbg': Dlbg,  # constraints lower bound
    'ubg': Dubg,  # constraints upper bound
    'lbx': Dlbx,
    'ubx': Dubx
}
# Simulation loop
# Initializing
t0 = 0  # setting time to zero
x0 = ca.DM([x_init, y_init, theta_init])  # initial state
# Setting the robot initial coordinates
xs = ca.DM([x_target, y_target, theta_target])  # target state
t = ca.DM(t0)
# Mecanum
Mu0 = ca.DM.zeros((N, n_Mcontrols))  # initial control
Du0 = ca.DM.zeros((N, n_Dcontrols))  # initial control
MX0 = ca.repmat(x0, 1, N + 1)  # initial state full

sim_time = 200  # Maximum simulation time
# Start MPC
mpciter = 0
cat_states = np.array(MX0.full())
cat_controls = np.array(Mu0[:, 0].full())
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
            sim_time / h)):
        if (abs(x0[0]-x_init)<5):

            # condition for the loop to end when the robot approach the target or when the simulation time reach its maximum
            t1 = time()
            Margs['p'] = ca.vertcat(x0, xs)
            # put the 6 initial states in the paramters list
            # set the values of the parameters vector
            # initial value of the optimization variables
            Margs['x0'] = ca.vertcat(ca.reshape(MX0, n_states * (N + 1), 1), ca.reshape(Mu0, n_Mcontrols * N, 1))
            # filling the values that will be optimized
            sol = Msolver(x0=Margs['x0'], lbx=Margs['lbx'], ubx=Margs['ubx'], lbg=Margs['lbg'], ubg=Margs['ubg'], p=Margs['p'])
            # calling the solver with the constrains , the optimize list and the parameters
            u = ca.reshape(sol['x'][n_states * (N + 1):], n_Mcontrols, N)
            MX0 = ca.reshape(sol['x'][: n_states * (N + 1)], n_states, N + 1)
            # updating the u and X0 matricies and reshaping it to be valid in optimizing
            cat_states = np.dstack((cat_states, np.array(MX0.full())))
            cat_controls = np.vstack((cat_controls, np.array((u[:, 0]).full())))
            t = np.vstack((t, t0))
            # getting the solution from the solver for states and control
            # Apply the control and shift the solution
            t0, x0, Mu0 = shift(h, t0, x0, u, Mf)
            # shifting to the new values for next iteration
            MX0 = ca.horzcat(MX0[:, 1:], ca.reshape(MX0[:, -1], -1, 1))
            t2 = time()
            print(mpciter)
            # counter
            print(t2 - t1)
            # printing the time for one iteration
            times = np.vstack((times, t2 - t1))
            # calculating the time
            # Shift trajectory to initialize the next step
            mpciter = mpciter + 1
            # updating the counter
        else:
            # condition for the loop to end when the robot approach the target or when the simulation time reach its maximum
            t1 = time()
            Dargs['p'] = ca.vertcat(x0, xs)
            # put the 6 initial states in the paramters list
            # set the values of the parameters vector
            # initial value of the optimization variables
            Dargs['x0'] = ca.vertcat(ca.reshape(MX0, n_states * (N + 1), 1), ca.reshape(Du0, n_Dcontrols * N, 1))
            # filling the values that will be optimized
            sol = Dsolver(x0=Dargs['x0'], lbx=Dargs['lbx'], ubx=Dargs['ubx'], lbg=Dargs['lbg'], ubg=Dargs['ubg'],
                          p=Dargs['p'])
            # calling the solver with the constrains , the optimize list and the parameters
            u = ca.reshape(sol['x'][n_states * (N + 1):], n_Dcontrols, N)
            MX0 = ca.reshape(sol['x'][: n_states * (N + 1)], n_states, N + 1)
            # updating the u and X0 matricies and reshaping it to be valid in optimizing
            cat_states = np.dstack((cat_states, np.array(MX0.full())))
            cat_controls = np.vstack((cat_controls, np.array((u[:, 0]).full())))
            t = np.vstack((t, t0))
            # getting the solution from the solver for states and control
            # Apply the control and shift the solution
            t0, x0, Du0 = shift(h, t0, x0, u, Df)
            # shifting to the new values for next iteration
            MX0 = ca.horzcat(MX0[:, 1:], ca.reshape(MX0[:, -1], -1, 1))
            t2 = time()
            print(mpciter)
            # counter
            print(t2 - t1)
            # printing the time for one iteration
            times = np.vstack((times, t2 - t1))
            # calculating the time
            # Shift trajectory to initialize the next step
            mpciter = mpciter + 1
            # updating the counter
    main_loop_time = time()
    ss_error = ca.norm_2(x0 - xs)
    # calculating the remaining distance to reach the target
    average_mpc_time = main_loop_time / (mpciter + 1)
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

    # simulate
    simulate(cat_states, cat_controls, times, h, N,
             np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]),x_obstcale,y_obstcale,r_obstacle, save=False)
