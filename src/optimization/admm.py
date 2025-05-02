import numpy as np
import cvxpy as cp
from tqdm import tqdm

class ADMMSolver:
    """
    ADMM solver for compositional safety verification.
    
    This class implements the Alternating Direction Method of Multipliers (ADMM)
    algorithm for solving the global optimization problem in the compositional
    safety verification framework.
    """
    
    def __init__(self, network, rho=1.0, max_iter=100, tol=1e-4, verbose=False):
        """
        Initialize the ADMM solver.
        
        Args:
            network: Network object.
            rho (float, optional): ADMM penalty parameter. Defaults to 1.0.
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.
            tol (float, optional): Convergence tolerance. Defaults to 1e-4.
            verbose (bool, optional): Whether to print progress. Defaults to False.
        """
        self.network = network
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        # Initialize subsystem optimizers
        self.subsystem_optimizers = []
        for subsystem in network.subsystems:
            self.subsystem_optimizers.append(SubsystemOptimizer(subsystem, rho))
        
        # Initialize global optimizer
        self.global_optimizer = GlobalOptimizer(network)
        
        # Initialize variables
        self.X_list = None  # X matrices for each subsystem
        self.Z_list = None  # Auxiliary variables for each subsystem
        self.Lambda_list = None  # Dual variables for each subsystem
        
        # Initialize results
        self.status = None
        self.objective_values = []
        self.residuals = []
    
    def solve(self, barriers, data_list, verbose=None):
        """
        Solve the compositional safety verification problem.
        
        Args:
            barriers (list): List of barrier function representations.
            data_list (list): List of data tuples (states, inputs, next_states) for each subsystem.
            verbose (bool, optional): Whether to print progress. If None, uses self.verbose.
            
        Returns:
            dict: Results dictionary.
        """
        if verbose is None:
            verbose = self.verbose
            
        n_subsystems = len(self.network.subsystems)
        
        # Initialize X, Z, Lambda
        self.X_list = [None] * n_subsystems
        self.Z_list = [None] * n_subsystems
        self.Lambda_list = [None] * n_subsystems
        
        # Initialize with random values
        for i in range(n_subsystems):
            n_w = self.network.subsystems[i].input_dim
            n_x = self.network.subsystems[i].state_dim
            
            # Initialize X and Z randomly
            self.X_list[i] = np.random.randn(n_w + n_x, n_w + n_x)
            # Make X symmetric
            self.X_list[i] = (self.X_list[i] + self.X_list[i].T) / 2
            
            self.Z_list[i] = self.X_list[i].copy()
            
            # Initialize Lambda to zeros
            self.Lambda_list[i] = np.zeros((n_w + n_x, n_w + n_x))
        
        # Main ADMM loop
        converged = False
        iter_count = 0
        
        # Initialize progress bar
        pbar = tqdm(total=self.max_iter) if verbose else None
        
        while not converged and iter_count < self.max_iter:
            # Step 1: Update X_i (local step)
            for i in range(n_subsystems):
                self.X_list[i] = self.subsystem_optimizers[i].update_X(
                    barriers[i], data_list[i], self.Z_list[i], self.Lambda_list[i]
                )
            
            # Step 2: Update Z (global step)
            Z_list_prev = [Z.copy() for Z in self.Z_list]
            self.Z_list = self.global_optimizer.update_Z(self.X_list, self.Lambda_list)
            
            # Step 3: Update Lambda (dual update)
            for i in range(n_subsystems):
                self.Lambda_list[i] += self.rho * (self.X_list[i] - self.Z_list[i])
            
            # Check convergence
            res_primal = 0
            res_dual = 0
            
            for i in range(n_subsystems):
                # Primal residual: ||X_i - Z_i||_F
                res_primal += np.linalg.norm(self.X_list[i] - self.Z_list[i], 'fro') ** 2
                
                # Dual residual: ||Z_i - Z_i^prev||_F
                res_dual += np.linalg.norm(self.Z_list[i] - Z_list_prev[i], 'fro') ** 2
            
            res_primal = np.sqrt(res_primal)
            res_dual = self.rho * np.sqrt(res_dual)
            
            # Store residuals
            self.residuals.append((res_primal, res_dual))
            
            # Check convergence
            converged = res_primal < self.tol and res_dual < self.tol
            
            # Update progress bar
            if verbose:
                pbar.update(1)
                pbar.set_description(f"Res Primal: {res_primal:.6f}, Res Dual: {res_dual:.6f}")
            
            iter_count += 1
        
        # Close progress bar
        if verbose:
            pbar.close()
        
        # Check LMI condition
        lmi_satisfied, max_eigenvalue = self.network.check_LMI_condition()
        
        # Set status
        if converged:
            self.status = "converged"
        elif iter_count >= self.max_iter:
            self.status = "max_iter_reached"
        else:
            self.status = "unknown"
        
        # Set the optimized X matrices in the subsystems
        for i in range(n_subsystems):
            self.network.subsystems[i].X_matrix = self.Z_list[i]
        
        # Return results
        return {
            'status': self.status,
            'iter_count': iter_count,
            'residuals': self.residuals,
            'lmi_satisfied': lmi_satisfied,
            'max_eigenvalue': max_eigenvalue,
            'X_matrices': self.Z_list
        }


class SubsystemOptimizer:
    """
    Optimizer for individual subsystems in the ADMM algorithm.
    """
    
    def __init__(self, subsystem, rho=1.0):
        """
        Initialize the subsystem optimizer.
        
        Args:
            subsystem: Subsystem object.
            rho (float, optional): ADMM penalty parameter. Defaults to 1.0.
        """
        self.subsystem = subsystem
        self.rho = rho
    
    def update_X(self, barrier, data, Z, Lambda):
        """
        Update the X matrix for a subsystem in the ADMM algorithm.
        
        Args:
            barrier: Barrier function representation.
            data: Data tuple (states, inputs, next_states).
            Z: Auxiliary variable.
            Lambda: Dual variable.
            
        Returns:
            numpy.ndarray: Updated X matrix.
        """
        # Extract data
        states, inputs, next_states = data
        n_samples = states.shape[0]
        
        # Extract dimensions
        n_x = self.subsystem.state_dim
        n_w = self.subsystem.input_dim
        
        # Define variables
        X = cp.Variable((n_w + n_x, n_w + n_x), symmetric=True)
        eta = cp.Variable()  # Slack variable
        
        # Compute barrier values
        B_values = np.zeros(n_samples)
        B_next_values = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Handle different return types from barrier.evaluate
            if hasattr(barrier, 'model'):
                # Neural network barrier
                B_val = barrier.evaluate(states[i].reshape(1, -1))
                B_next_val = barrier.evaluate(next_states[i].reshape(1, -1))
            else:
                # Linear barrier
                B_val = barrier.evaluate(states[i].reshape(1, -1), barrier.parameters)
                B_next_val = barrier.evaluate(next_states[i].reshape(1, -1), barrier.parameters)
            
            # Handle different return types
            if np.isscalar(B_val):
                B_values[i] = B_val
            elif hasattr(B_val, "shape") and B_val.shape == ():
                # Handle 0-dim arrays (numpy scalars)
                B_values[i] = B_val.item()
            elif hasattr(B_val, "__getitem__") and len(B_val) > 0:
                # Handle array-like objects with at least one element
                B_values[i] = B_val[0]
            else:
                raise ValueError(f"Unexpected output from barrier.evaluate: {B_val}")
            
            if np.isscalar(B_next_val):
                B_next_values[i] = B_next_val
            elif hasattr(B_next_val, "shape") and B_next_val.shape == ():
                # Handle 0-dim arrays (numpy scalars)
                B_next_values[i] = B_next_val.item()
            elif hasattr(B_next_val, "__getitem__") and len(B_next_val) > 0:
                # Handle array-like objects with at least one element
                B_next_values[i] = B_next_val[0]
            else:
                raise ValueError(f"Unexpected output from barrier.evaluate: {B_next_val}")
                
        # Add constraints
        constraints = [eta >= 0]  # Non-negative slack
        
        # Sample constraints for decay: B(x^+) <= B(x) + z^T X z for all x, w
        for i in range(n_samples):
            z_i = np.concatenate((inputs[i], states[i]))
            constraints.append(B_next_values[i] - B_values[i] - cp.quad_form(z_i, X) <= eta)
        
        # ADMM regularization term: (rho/2) * ||X - Z + Lambda/rho||_F^2
        F_norm = cp.norm(X - Z + Lambda / self.rho, 'fro')
        regularization = (self.rho / 2) * F_norm ** 2
        
        # Define objective
        objective = cp.Minimize(eta + regularization)
        
        # Solve problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except:
            # If optimization fails, return the previous value
            return Z
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # If optimization fails, return the previous value
            return Z
        
        return X.value


class GlobalOptimizer:
    """
    Optimizer for the global step in the ADMM algorithm.
    """
    
    def __init__(self, network):
        """
        Initialize the global optimizer.
        
        Args:
            network: Network object.
        """
        self.network = network
    
    def update_Z(self, X_list, Lambda_list):
        """
        Update the Z matrices in the ADMM algorithm.
        
        Args:
            X_list: List of X matrices for each subsystem.
            Lambda_list: List of dual variables for each subsystem.
            
        Returns:
            list: Updated Z matrices.
        """
        # Define variables for all subsystems
        n_subsystems = len(self.network.subsystems)
        
        # Extract dimensions for each subsystem
        state_dims = self.network.state_dims
        input_dims = self.network.input_dims
        
        # Define variables
        Z_vars = []
        for i in range(n_subsystems):
            n_w = input_dims[i]
            n_x = state_dims[i]
            Z_vars.append(cp.Variable((n_w + n_x, n_w + n_x), symmetric=True))
        
        # Formulate the LMI constraint: M_ext^T * Z_diag * M_ext <= 0
        # where Z_diag is the block diagonal matrix of Z_i
        
        # Construct the extended interconnection matrix
        M = self.network.M
        
        # Fix: Check that the dimensions match before stacking
        # M should have shape (total_input_dim, total_state_dim)
        # Make sure the identity matrix has appropriate dimensions
        if M.shape[0] != self.network.total_input_dim:
            # Add appropriate padding or reshape M to ensure compatibility
            padded_M = np.zeros((self.network.total_input_dim, M.shape[1]))
            padded_M[:M.shape[0], :] = M
            M = padded_M
            
        M_ext = np.vstack((M, np.eye(M.shape[1])))
        
        # Construct the block diagonal of Z_i
        Z_diag_blocks = []
        
        # Input blocks come first
        input_idx = 0
        for i in range(n_subsystems):
            n_w = input_dims[i]
            n_x = state_dims[i]
            Z_i = Z_vars[i]
            
            # Extract Z_i^{11} (input-input block)
            Z_diag_blocks.append(('input', input_idx, Z_i[:n_w, :n_w]))
            input_idx += n_w
        
        # State blocks come second
        state_idx = 0
        for i in range(n_subsystems):
            n_w = input_dims[i]
            n_x = state_dims[i]
            Z_i = Z_vars[i]
            
            # Extract Z_i^{22} (state-state block)
            Z_diag_blocks.append(('state', state_idx, Z_i[n_w:, n_w:]))
            state_idx += n_x
        
        # Cross blocks come last
        input_idx = 0
        state_idx = 0
        for i in range(n_subsystems):
            n_w = input_dims[i]
            n_x = state_dims[i]
            Z_i = Z_vars[i]
            
            # Extract Z_i^{12} (input-state block)
            Z_diag_blocks.append(('cross', (input_idx, state_idx), Z_i[:n_w, n_w:]))
            input_idx += n_w
            state_idx += n_x
        
        # Construct the LMI constraint
        # We need to express the LMI as a sum of scaled semidefinite terms
        
        # Set up linear terms
        terms = []
        
        # Input-input blocks
        for type_i, idx_i, Z_block in Z_diag_blocks:
            if type_i == 'input':
                # Extract the relevant slice of M_ext
                M_i = M_ext[idx_i:idx_i+Z_block.shape[0], :]
                terms.append(M_i.T @ Z_block @ M_i)
        
        # State-state blocks
        for type_i, idx_i, Z_block in Z_diag_blocks:
            if type_i == 'state':
                # Extract the relevant slice of M_ext
                total_input_dim = self.network.total_input_dim
                M_i = M_ext[total_input_dim+idx_i:total_input_dim+idx_i+Z_block.shape[0], :]
                terms.append(M_i.T @ Z_block @ M_i)
        
        # Cross blocks
        for type_i, idx_i, Z_block in Z_diag_blocks:
            if type_i == 'cross':
                input_idx, state_idx = idx_i
                # Extract the relevant slices of M_ext
                M_w = M_ext[input_idx:input_idx+Z_block.shape[0], :]
                total_input_dim = self.network.total_input_dim
                M_x = M_ext[total_input_dim+state_idx:total_input_dim+state_idx+Z_block.shape[1], :]
                terms.append(M_w.T @ Z_block @ M_x + M_x.T @ Z_block.T @ M_w)
        
        # Sum the terms to get the LMI
        lmi = sum(terms)
        
        # Add the LMI constraint
        constraints = [lmi <= 0]
        
        # Add augmented Lagrangian terms to the objective
        objective_terms = []
        for i in range(n_subsystems):
            # Augmented Lagrangian: ||Z_i - (X_i + Lambda_i/rho)||_F^2
            objective_terms.append(cp.norm(Z_vars[i] - (X_list[i] + Lambda_list[i]), 'fro') ** 2)
        
        # Define objective
        objective = cp.Minimize(sum(objective_terms))
        
        # Solve problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except:
            # If optimization fails, return the previous values
            return X_list
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # If optimization fails, return the previous values
            return X_list
        
        # Return the updated Z values
        return [Z_i.value for Z_i in Z_vars] 