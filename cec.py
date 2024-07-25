import casadi as ca
import numpy as np
import utils


class CEC:
    def __init__(self, obstacle_avoidance = False):
        self.N = 10  # look ahead stesp
        self.time_step = 0.5
        self.v_min, self.v_max = 0, 1       # control (lin vel bounds)
        self.w_min, self.w_max = -1, 1      # control (ang vel bounds)
        self.x_min, self.x_max = -3, 3      # x - workspace 
        self.y_min, self.y_max = -3, 3      # y - workspace
        self.theta_min, self.theta_max = -np.pi, np.pi     # rotation - workspace
        self.obstacle_avoidance = obstacle_avoidance        # whether to avoid obstacle or not?

    def __call__(self, cur_iter: int, cur_state: np.ndarray, cur_ref: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        ######################################################################################################
        # TODO: define optimization variables

        opti = ca.Opti()    # import a new optimization problem

        # define variables
        U = opti.variable(2, self.N)  # control input variables over the Horizon (v,w)
        E = opti.variable(3, self.N + 1) # State input variables over the Horizon (x, y, theta)

        # parameters for cost calculation
        if self.obstacle_avoidance:
            Q = ca.DM.eye(2) * 0.8       # state (x, y) weighting in the cost function
            R = ca.DM.eye(2) * 1.4     # control weighting in the cost function
            q = 10                 # rotation (theta) weighting in the cost function
            gamma = 0.85

        else:
            Q = ca.DM.eye(2) * 1.2       # state (x, y) weighting in the cost function
            R = ca.DM.eye(2) * 0.1     # control weighting in the cost function
            q = 10                # rotation (theta) weighting in the cost function
            gamma = 0.9

        ######################################################################################################
        # TODO: define optimization constraints and optimization objective

        # objective function
        objective_fun = 0
        for i in range(self.N):
            p_tilda_t = E[:2, i]
            theta_tilda_t = E[2, i]
            control = U[:, i]

            # ground truth state
            cur_ref_state = utils.lissajous(cur_iter + i)
            x_true = E[0, i] + cur_ref_state[0]
            y_true = E[1, i] + cur_ref_state[1]

            # free workspace constraints
            # boundary constraints
            opti.subject_to(opti.bounded(self.x_min, x_true, self.x_max))
            opti.subject_to(opti.bounded(self.y_min, y_true, self.y_max))

            # collision checking with circles (with some tolerance)
            if self.obstacle_avoidance:
                # obstacle 1
                opti.subject_to((x_true - 1) ** 2 + (y_true - 2) ** 2 >= (0.5) ** 2)
                # obstacle 2
                opti.subject_to((x_true + 2) ** 2 + (y_true + 2) ** 2 >= (0.5) ** 2)
            
            # model dynamics
            opti.subject_to(E[:, i+1] == self.Error_NextState(cur_iter + i, E[:, i], U[:, i]))  # Dynamics

            objective_fun += (gamma ** i) * (ca.mtimes([p_tilda_t.T, Q, p_tilda_t]) + q * (1 - ca.cos(theta_tilda_t))** 2 + ca.mtimes([control.T, R, control]))

        # terminal cost
        q_T = (gamma ** self.N) * (ca.mtimes([E[:2, -1].T, Q, E[:2, -1]]) + q * (1 - ca.cos(E[2, -1]))** 2)
        objective_fun += q_T        # overall cost

        # constraints          
        opti.subject_to(opti.bounded(self.v_min, U[0, :], self.v_max))      #control bounds on velocity
        opti.subject_to(opti.bounded(self.w_min, U[1, :], self.w_max))      # control bounds on angular velocity

        # intial conditions
        cur_error = cur_state - cur_ref
        opti.subject_to(E[:, 0] == cur_error)
        
        # Set initial guesses and bounds for the solver
        # Here you would set all initial guesses for U and E
        opti.set_initial(U[:,0], [0.5, -0.25])
        opti.set_initial(E[:,0], cur_error)

        # minimise the objective function
        opti.minimize(objective_fun)

        # Create the solver instance
        opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0})
        
        # Set the tolerance for constraints
        # opti.solver_options = {'constr_viol_tol': 1e-6}

        # Solve the problem
        sol = opti.solve()

        # Extract the control inputs from the solution
        u = sol.value(U[:, 0])
        return u
    
    
    def Error_NextState(self, i, cur_error, control):
        """ Compute next error state based on the dynamics """
        cur_ref_state = np.array(utils.lissajous(i))
        next_ref_state = np.array(utils.lissajous(i+1))
        diff_ref_state = cur_ref_state - next_ref_state

        theta = cur_error[2] + cur_ref_state[2]
        rot_3d_z = ca.vertcat(
            ca.hcat([ca.cos(theta), 0]),
            ca.hcat([ca.sin(theta), 0]),
            ca.hcat([0, 1])
        )
        f = ca.mtimes(rot_3d_z, control)
        
        next_error = cur_error + self.time_step * f + diff_ref_state  # compute next error state
        return next_error
