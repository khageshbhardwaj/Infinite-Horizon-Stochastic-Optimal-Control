import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time
from tqdm import tqdm
from scipy.stats import multivariate_normal
from utils import visualize
import ray
import utils


#################################################################################################

# initialization

# intial state of the car
x_init = 1.5
y_init = 0.0
theta_init = np.pi / 2

# control limits
v_min, v_max = 0, 1 # lin vel 
w_min, w_max = -1, 1 # ang vel

# horizon
time_step = 0.5  # time between steps in seconds
T = 100
sim_time = 120  # simulation time

# noise characteristics
sigma = np.array([0.04, 0.04, 0.004])

# # cost metrices
# Q = np.eye(2) * 0.4
# R = np.eye(2) * 1.5
# q = 10

# gamma = 0.9    #discount factor

# cost metrices
Q = np.eye(2) * 8
R = np.eye(2) * 1.2
q = 1.05

gamma = 0.88    #discount factor


#################################################################################################

# discretization
x_min, x_max = -3, 3
y_min, y_max = -3, 3
theta_min, theta_max = -np.pi, np.pi
num_element = 10

t_space = np.arange(0,T)
ex_space = np.linspace(x_min, x_max, num_element) # discretizing the x error state-space
ey_space = np.linspace(y_min, y_max, num_element) # discretizing the y error state-space
eth_space = np.linspace(theta_min, theta_max, (num_element))  # discretizing the theta error state-space

v_space = np.linspace(v_min, v_max, 10)
w_space = np.linspace(w_min, w_max, 10)

# Initialize an empty list to hold all possible combinations
state_space_list = []

for t in t_space:
    for ex in ex_space:
        for ey in ey_space:
            for eth in eth_space:
                state_space_list.append([t, ex, ey, eth])

state_space = np.array(state_space_list)

# Initialize an empty list to hold all possible combinations
control_space_list = []

for v in v_space:
    for w in w_space:
        control_space_list.append([v, w])

control_space = np.array(control_space_list)


len_state_space = state_space.shape[0]
len_control_space = control_space.shape[0]


#################################################################################################

# enforcing wrap around for the theta
def theta_wrap_around(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

#################################################################################################
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2 * np.pi / (T * time_step)
    b = 3 * a
    k = k % T
    delta = np.pi / 2
    xref = xref_start + A * np.sin(a * k * time_step + delta)
    yref = yref_start + B * np.sin(b * k * time_step)
    v = [A * a * np.cos(a * k * time_step + delta), B * b * np.cos(b * k * time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]


#################################################################################################

# compute the stage cost for all possible state_space and control_space
def compute_stage_costs(lissajous):
    print('Calculating Stage Cost')

    try:
        stage_cost = np.load('stage_cost.npy')
    except:
        # initialize the stage cost
        stage_cost = np.zeros((len_state_space, len_control_space))
        OBSTACLE_COST = 1e9
        OUT_OF_BOUNDS_COST = 1e9

        obstacle_centers = [(1, 2), (-2, -2)]
        obstacle_radius = 0.5

        for i, state in tqdm(enumerate(state_space)):
            t = state[0]
            ref_traj = lissajous(t)
            cur_state = state[1:] + ref_traj

            p_tilda = state[1:3]
            theta_tilda = state[3]

            for j, control in enumerate(control_space):
                U = control

                stage_cost[i, j] = (p_tilda.T @ Q @ p_tilda) + (q * (1 - np.cos(theta_tilda)) ** 2) + (U.T @ R @ U)
                
                x, y = cur_state[0], cur_state[1]
                in_obstacle = any((x - xc)**2 + (y - yc)**2 < obstacle_radius**2 for xc, yc in obstacle_centers)
                out_of_bounds = not (x_min <= x <= x_max and y_min <= y <= y_max)

                if in_obstacle:
                    stage_cost[i, j] = OBSTACLE_COST
                elif out_of_bounds:
                    stage_cost[i, j] = OUT_OF_BOUNDS_COST

        np.save('stage_cost.npy', stage_cost)
    
    return stage_cost

stage_cost = compute_stage_costs(lissajous)

#################################################################################################

# Convert the metric state to grid indices based on discretization grids
    
def state_metric_to_index(metric_state):
    # Calculate indices for each dimension
    t_index = np.digitize([metric_state[0]], t_space, right=True)[0] - 1
    x_index = np.digitize([metric_state[1]], ex_space, right=True)[0] - 1
    y_index = np.digitize([metric_state[2]], ey_space, right=True)[0] - 1
    theta_index = np.digitize([metric_state[3]], eth_space, right=True)[0] - 1

    # Clamp the indices to ensure they fall within the valid range of the spaces
    t_index = max(0, min(t_index, len(t_space) - 1))
    x_index = max(0, min(x_index, len(ex_space) - 1))
    y_index = max(0, min(y_index, len(ey_space) - 1))
    theta_index = max(0, min(theta_index, len(eth_space) - 1))

    return [t_index, x_index, y_index, theta_index]

# Convert grid indices back to the metric state based on discretization grids
    
def state_index_to_metric(indices):
    t = t_space[indices[0]]
    x = ex_space[indices[1]]
    y = ey_space[indices[2]]
    theta = eth_space[indices[3]]
    return np.array([t, x, y, theta])


#################################################################################################

# Convert control metrics to indices in the control space

def control_metric_to_index(control_metric):
    v_indices = np.digitize(control_metric[0], v_space, right=True) - 1
    w_indices = np.digitize(control_metric[1], w_space, right=True) - 1
    return v_indices, w_indices

# Convert indices in the control space to metric values
def control_index_to_metric(v_indices, w_indices):
    v_metrics = v_space[v_indices]
    w_metrics = w_space[w_indices]
    return v_metrics, w_metrics

#################################################################################################

# Find the index of the grid point closest to the given value."""
def find_index_in_grid(value, grid):
    return np.argmin(np.abs(grid - value))

def calculate_state_index(t, indices, num_element):
    """Calculate a unique state index given time and spatial indices."""
    # This uses a weighted sum where the weights are powers of number of elements,
    # similar to calculating a number in a mixed radix numeral system
    return int(t * num_element**3 + indices[0] * num_element**2 + indices[1] * num_element + indices[2])

def state_to_index(state):
    """
    Given a state consisting of time t, and spatial coordinates x, y, and theta,
    find the closest matching state in a discretized space and return its index.
    """
    t, x, y, theta = state
    # Calculate closest indices in each spatial dimension
    spatial_indices = [find_index_in_grid(coord, grid) for coord, grid in zip((x, y, theta), (ex_space, ey_space, eth_space))]
    # Calculate and return the unique index of the state
    return calculate_state_index(t, spatial_indices, num_element)

#################################################################################################

# Generates a series of states perturbed around the given state based on specified noise standard deviations.

def compute_possible_next_states(cur_state, noise_std_dev):
    # Create a matrix where each row is the original state
    possible_next_states = np.tile(cur_state, (7, 1))
    
    # Create an array of adjustments
    purturbations = np.array([noise_std_dev[0], -noise_std_dev[0], noise_std_dev[1], -noise_std_dev[1], noise_std_dev[2], -noise_std_dev[2]])
    
    # Apply adjustments to the respective columns
    possible_next_states[[1, 2], 1] += purturbations[:2]
    possible_next_states[[3, 4], 2] += purturbations[2:4]
    possible_next_states[[5, 6], 3] += purturbations[4:]
    
    return possible_next_states


#################################################################################################

# Compute the next state for all control inputs given the current state and a motion model.
    
def compute_next_state(state, control_space):
    time_step = 0.5
    t = state[0]
    theta_tilda = state[3]
    
    ref_traj = np.array(lissajous(t))
    next_ref_traj = np.array(lissajous(t + 1))
    alpha = ref_traj[2]
    
    theta = theta_tilda + alpha
    diff_ref_traj = ref_traj - next_ref_traj

    G_tilda = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])

    # Compute next state E_next for all control inputs
    f = time_step * (G_tilda @ control_space.T)
    E_next = state[1:, None] + f + diff_ref_traj[:, None]
    E_next[2, :] = theta_wrap_around(E_next[2, :])  # Adjusting theta to be within [-pi, pi]

    return E_next

# computing transition matrix for the robot 
#  
def compute_transition_matrix(lissajous):
    print('Calculating transition matrix')
    try:
        transition_matrix = np.load('transition_matrix.npy')
    except:
        transition_matrix = np.zeros((len_state_space, len_control_space, 7, 2))  # Combined indices and probabilities matrix

        for index, state in tqdm(enumerate(state_space)):
            t = state[0]
            E_next = compute_next_state(state, control_space)
            time_component = np.full((control_space.shape[0],), (t + 1) % T)  # Time component for all controls

            for j, control in enumerate(control_space):
                # Construct the complete state for the next time step
                next_state = np.concatenate(([time_component[j]], E_next[:, j]))
                next_state_index = state_to_index(next_state)

                # Generate possible next states based on the current state and noise
                possible_next_states = compute_possible_next_states(next_state, noise_std_dev=[0.04, 0.04, 0.004])
                
                # Calculate probabilities of these states
                state_mean = state_space[next_state_index, 1:]  # Mean based on the current next state index
                covariance = np.diag([0.04, 0.04, 0.004])  # Covariance matrix for the multivariate normal distribution
                probabilities = multivariate_normal.pdf(possible_next_states[:, 1:], state_mean, covariance)
                normalized_probabilities = probabilities / probabilities.sum()  # Normalize probabilities

                # Calculate indices for these possible states
                possible_indices = np.array([state_to_index(state) for state in possible_next_states])  # Use list comprehension for clarity

                # Store results in the transition matrix
                transition_matrix[index, j, :, 0] = possible_indices
                transition_matrix[index, j, :, 1] = normalized_probabilities

        np.save('transition_matrix.npy', transition_matrix)
    return transition_matrix

transition_matrix = compute_transition_matrix(lissajous)


#################################################################################################

# Initialize the value function as a zero array based on the number of states.
def init_value_function(len_state_space):
    return np.zeros(len_state_space)

# Initialize the Q-value function as a zero array based on the number of states, number of controls
def init_QValue_function(len_state_space, len_control_space):
    return np.zeros((len_state_space, len_control_space))

#################################################################################################

# Policy evaluation step of the GPI algorithm
# Evaluate the value function using the Bellman optimality equation until convergence.
    
def policy_evaluation(V_func, Q_func, L, gamma, transition_matrix, epsilon=1e-3, max_iterations=1000):
    iteration = 0

    while True:
        V_prev = np.copy(V_func)
        Q_func = L + gamma * np.sum(transition_matrix[:,:,:,1] * V_func[transition_matrix[:,:,:,0].astype(int)], axis=2)
        V_func = np.min(Q_func, axis=1)
        max_change = np.max(np.abs(V_func - V_prev))
        print("Iteration:", iteration, "Value change:", max_change)

        if max_change < epsilon or iteration >= max_iterations:
            break
        iteration += 1

    return V_func, Q_func


#################################################################################################


# Compute next error state based on the dynamics with added noise, using NumPy for matrix operations """

def error_next_state(t, cur_error, control):
    # Retrieve the current and next reference states using Lissajous function
    time_step = 0.5
    ref_traj = np.array(lissajous(t))
    next_ref_traj = np.array(lissajous(t + 1))
    diff_ref_traj = ref_traj - next_ref_traj

    theta = cur_error[2] + ref_traj[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])

    f = rot_3d_z @ control.T

    # Define the noise mean and covariance
    mean = [0, 0, 0]  # Mean of the noise
    cov = np.diag([0.04, 0.04, 0.004])  # Covariance matrix of the noise
    noise = np.random.multivariate_normal(mean, cov)  # Sample noise

    # Compute the next error state incorporating the dynamics and the noise
    next_error = cur_error + time_step * f + diff_ref_traj + noise

    return next_error

#################################################################################################

# main code loop
cur_traj = []  # initialize an empty list to store the current car trajectory
ref_traj = []  # Initialize an empty list to store reference trajectory points

for k in t_space:
    ref_point = lissajous(k)
    ref_traj.append(ref_point)

ref_traj = np.array(ref_traj)

# initial car state
x_init, y_init, theta_init = 1.5, 0.0, np.pi/2
cur_state = np.array([x_init, y_init, theta_init])
cur_error = cur_state - ref_traj[0,:]
cur_traj.append(cur_state)

# initialise value function
V_function = init_value_function(len_state_space)
Q_value_function = init_QValue_function(len_state_space, len_control_space)

V_function, Q_value_function = policy_evaluation(V_function, Q_value_function, stage_cost, gamma, transition_matrix, epsilon=0.001, max_iterations=1000)

times = []
cum_error_trans = 0
cum_error_rot = 0

for t in tqdm(t_space):
    if t + 1 >= ref_traj.shape[0]:
        break
    t1 = time()

    # Get the optimal control trajectory
    err_state = np.zeros(4)
    err_state[0] = t
    err_state[1:] = cur_error
    err_state_idx = state_to_index(err_state)
    U_improved = control_space[np.argmin(Q_value_function[err_state_idx,:])]

    # Apply the first control input to the system
    err_next = error_next_state(t, cur_error, U_improved)
    err_next[2] = theta_wrap_around(err_next[2])

    cur_state = err_next + ref_traj[t+1, :]
    # cur_state[2] = wrap_to_pi(cur_state[2])
    
    cur_error = err_next
    t2 = time()
    times.append(t2-t1)
    cur_traj.append(cur_state)
    cum_error_trans = cum_error_trans + np.linalg.norm(cur_state[:2] - ref_traj[t, :2])    
    cum_error_trans = cum_error_rot + np.linalg.norm(cur_state[2] - ref_traj[t, 2])

cur_traj = np.array(cur_traj)
execution_time = np.sum(times) * 1000
average_time = execution_time / len(times) * 1000

obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
visualize(cur_traj, ref_traj, obstacles, times, time_step, save=True)
print(f"Execution Time: {execution_time} s")
print(f"Average Execution time: {average_time} ms")
print(f"Final Translation Tracking Error: {cum_error_trans}")
print(f"Final Rotational Tracking Error: {cum_error_rot}")

#################################################################################################
