import casadi as ca
from casadi import sin, cos, pi, sqrt

''' Constants '''
# Costs
Q_x = Q_y = 800 / 2
Q_theta = 260 / pi
# Q_theta = 60 makes the base rotates while moving to the target
# Decrasing it makes the base rotates after reaching the target
R1 = R2 = R3 = R4 = 6 / 5     # speed cost
A1 = A2 = A3 = A4 = 0         # No Acceleration

# MPC parameters
sampling_time = 0.3  # time between steps in seconds
N = 100              # number of look ahead steps
sim_time = 200       # simulation time

# Robot parameters
rob_radius = 0.68   # 0.4243 # diameter of the robot
wheel_radius = 0.07 # wheel radius
Lx = 0.24           # L in J Matrix (half robot x-axis length)
Ly = 0.24           # l in J Matrix (half robot y-axis length)
R = rob_radius/2
x_min = 0.4
x_max = 3
y_min = 0.4
y_max = 7
theta_min = -ca.inf
theta_max =  ca.inf
v_max = 1 / wheel_radius  # rad/s    (speed_m_per_sec / wheel_radius)
a_max = 0.15 / wheel_radius    # rad/s^2  (acceleration_m_per_sec2 / wheel_radius)

# Obstacles to be avoided
#    x   y   radius
obstacles = ca.DM([
    [10,  10, 1],
])
# obstacles = ca.DM([[]])  # uncomment for no obstacles
n_obstacles = obstacles.shape[0]

''' Symbols '''
# state symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
theta_ref = ca.SX.sym('theta_ref')
states = ca.vertcat(
    x,
    y,
    theta
)
n_states = states.numel()

# control symbolic variables
V_a = ca.SX.sym('V_a')
V_b = ca.SX.sym('V_b')
V_c = ca.SX.sym('V_c')
controls = ca.vertcat(
    V_a,
    V_b,
    V_c,
)
n_controls = controls.numel()

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_theta)

# controls weights matrix
R = ca.diagcat(R1, R2, R3, R4)

# acceleration weights matrix
A = ca.diagcat(A1, A2, A3, A4)

'''J = (wheel_radius/4) * ca.DM([
    [0,         R/sqrt(3),          R/sqrt(3)],
    [2*R/3,         -R/3,          -R/3],
    [-R/3*Lx, -R/3*Lx, -R/3*Lx]
])'''
J = ca.DM([
    [0    ,    0.34/sqrt(3),          0.34/sqrt(3)],
    [((2*0.34)/3)  ,      (-0.34/3),         (-0.34/3)],
    [(-0.34/(3*Lx)), (-0.34/(3*Lx))   , (-0.34/(3*Lx))]
])


rot_3d_z = ca.vertcat(
    ca.horzcat(cos(theta+theta_ref), -sin(theta+theta_ref), 0),
    ca.horzcat(sin(theta+theta_ref),  cos(theta+theta_ref), 0),
    ca.horzcat(         0,           0, 1)
)

state_change_rate = rot_3d_z @ J @ controls
derivatives = ca.Function('derivatives', [states, controls, theta_ref], [state_change_rate])

''' Defining Upper and Lower Bounds '''
lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

lbx[0: n_states*(N+1): n_states] = x_min      # X lower bound
lbx[1: n_states*(N+1): n_states] = y_min      # Y lower bound
lbx[2: n_states*(N+1): n_states] = theta_min  # theta lower bound

ubx[0: n_states*(N+1): n_states] = x_max      # X upper bound
ubx[1: n_states*(N+1): n_states] = y_max      # Y upper bound
ubx[2: n_states*(N+1): n_states] = theta_max  # theta upper bound

lbx[n_states*(N+1)+0::n_controls] = -v_max                 # v lower bound for all V
lbx[n_states*(N+1)+1::n_controls] = -v_max                 # v lower bound for all V
lbx[n_states*(N+1)+2::n_controls] = -v_max                 # v lower bound for all V
lbx[n_states*(N+1)+3::n_controls] = -v_max                 # v lower bound for all V

ubx[n_states*(N+1)+0::n_controls] = v_max                  # v upper bound for all V
ubx[n_states*(N+1)+1::n_controls] = v_max                  # v upper bound for all V
ubx[n_states*(N+1)+2::n_controls] = v_max                  # v upper bound for all V
ubx[n_states*(N+1)+3::n_controls] = v_max                  # v upper bound for all V