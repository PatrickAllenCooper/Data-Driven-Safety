import numpy as np
from src.systems.subsystem import Subsystem

class Network:
    """
    Network of interconnected discrete-time subsystems.
    
    This class represents a network of subsystems with a given interconnection structure.
    """
    
    def __init__(self, subsystems, interconnection_matrix=None, name=None):
        """
        Initialize the network.
        
        Args:
            subsystems (list): List of Subsystem objects.
            interconnection_matrix (numpy.ndarray, optional): Interconnection matrix, shape (p, n).
                If None, an identity matrix is used.
            name (str, optional): Name of the network.
        """
        self.subsystems = subsystems
        self.name = name if name is not None else f"Network_{id(self)}"
        
        # Compute dimensions
        self.n_subsystems = len(subsystems)
        self.state_dims = [sys.state_dim for sys in subsystems]
        self.input_dims = [sys.input_dim for sys in subsystems]
        
        self.total_state_dim = sum(self.state_dims)
        self.total_input_dim = sum(self.input_dims)
        
        # Set up interconnection matrix if not provided
        if interconnection_matrix is None:
            # Default: identity matrix
            self.M = np.eye(self.total_state_dim, self.total_input_dim)
        else:
            self.M = interconnection_matrix
            
        # Check dimensions of M
        assert self.M.shape == (self.total_input_dim, self.total_state_dim), \
            f"M should have shape ({self.total_input_dim}, {self.total_state_dim}), got {self.M.shape}"
    
    def step(self, state):
        """
        Compute one step of the network dynamics.
        
        Args:
            state (numpy.ndarray): Current state, shape (total_state_dim,).
            
        Returns:
            numpy.ndarray: Next state, shape (total_state_dim,).
        """
        # Split state into subsystem states
        subsystem_states = self._split_state(state)
        
        # Compute inputs for each subsystem
        inputs = self.M @ state
        subsystem_inputs = self._split_input(inputs)
        
        # Compute next state for each subsystem
        next_subsystem_states = []
        for i, sys in enumerate(self.subsystems):
            next_state = sys.step(subsystem_states[i], subsystem_inputs[i])
            next_subsystem_states.append(next_state)
            
        # Combine next states
        next_state = np.concatenate(next_subsystem_states)
        
        return next_state
    
    def is_safe(self, state):
        """
        Check if a state is safe (not unsafe).
        
        Args:
            state (numpy.ndarray): State to check, shape (total_state_dim,).
            
        Returns:
            bool: True if the state is safe, False otherwise.
        """
        # Split state into subsystem states
        subsystem_states = self._split_state(state)
        
        # Check if any subsystem state is in the unsafe set
        for i, sys in enumerate(self.subsystems):
            if sys.is_in_unsafe_set(subsystem_states[i]):
                return False
                
        return True
    
    def is_in_initial_set(self, state):
        """
        Check if a state is in the initial set.
        
        Args:
            state (numpy.ndarray): State to check, shape (total_state_dim,).
            
        Returns:
            bool: True if the state is in the initial set, False otherwise.
        """
        # Split state into subsystem states
        subsystem_states = self._split_state(state)
        
        # Check if all subsystem states are in the initial set
        for i, sys in enumerate(self.subsystems):
            if not sys.is_in_initial_set(subsystem_states[i]):
                return False
                
        return True
    
    def set_initial_sets(self, bounds_list):
        """
        Set the initial sets for all subsystems.
        
        Args:
            bounds_list (list): List of initial set bounds for each subsystem.
        """
        assert len(bounds_list) == self.n_subsystems, \
            f"Expected {self.n_subsystems} sets of bounds, got {len(bounds_list)}"
            
        for i, bounds in enumerate(bounds_list):
            self.subsystems[i].set_initial_set(bounds)
    
    def set_unsafe_sets(self, bounds_list):
        """
        Set the unsafe sets for all subsystems.
        
        Args:
            bounds_list (list): List of unsafe set bounds for each subsystem.
        """
        assert len(bounds_list) == self.n_subsystems, \
            f"Expected {self.n_subsystems} sets of bounds, got {len(bounds_list)}"
            
        for i, bounds in enumerate(bounds_list):
            self.subsystems[i].set_unsafe_set(bounds)
    
    def set_safety_parameters(self, X_matrices, gammas):
        """
        Set the safety parameters for all subsystems.
        
        Args:
            X_matrices (list): List of dissipation matrices for each subsystem.
            gammas (list): List of safety thresholds for each subsystem.
        """
        assert len(X_matrices) == len(gammas) == self.n_subsystems, \
            f"Expected {self.n_subsystems} matrices and thresholds, got {len(X_matrices)} and {len(gammas)}"
            
        for i, (X, gamma) in enumerate(zip(X_matrices, gammas)):
            self.subsystems[i].set_safety_parameters(X, gamma)
    
    def check_LMI_condition(self):
        """
        Check if the global dissipativity condition (LMI) holds.
        
        Returns:
            tuple: (bool, float) indicating whether the condition holds and the largest eigenvalue.
        """
        # Construct the block diagonal matrix of X matrices
        X_diag = self._construct_X_diagonal()
        
        # Extended interconnection matrix
        M = self.M
        
        # M should have shape (total_input_dim, total_state_dim)
        if M.shape[0] != self.total_input_dim:
            # Add appropriate padding or reshape M to ensure compatibility
            padded_M = np.zeros((self.total_input_dim, M.shape[1]))
            padded_M[:M.shape[0], :] = M
            M = padded_M
            
        M_ext = np.vstack((M, np.eye(M.shape[1])))
        
        # Compute the LMI Delta matrix
        Delta = M_ext.T @ X_diag @ M_ext
        
        # Check if Delta <= 0 (all eigenvalues are non-positive)
        eigenvalues = np.linalg.eigvals(Delta)
        max_eigenvalue = np.max(eigenvalues).real
        
        return max_eigenvalue <= 0, max_eigenvalue
    
    def generate_data(self, n_samples, from_initial=False):
        """
        Generate data points from all subsystems.
        
        Args:
            n_samples (int): Number of samples to generate per subsystem.
            from_initial (bool, optional): Whether to sample from the initial set.
                If False, samples from the entire state space. Defaults to False.
                
        Returns:
            list: List of data tuples (states, inputs, next_states) for each subsystem.
        """
        data = []
        
        for sys in self.subsystems:
            sys_data = sys.generate_data(n_samples, from_initial)
            data.append(sys_data)
            
        return data
    
    def simulate(self, initial_state, n_steps):
        """
        Simulate the network for a given number of steps.
        
        Args:
            initial_state (numpy.ndarray): Initial state, shape (total_state_dim,).
            n_steps (int): Number of steps to simulate.
            
        Returns:
            numpy.ndarray: Trajectory of states, shape (n_steps+1, total_state_dim).
        """
        # Initialize trajectory
        trajectory = np.zeros((n_steps + 1, self.total_state_dim))
        trajectory[0] = initial_state
        
        # Simulate
        for i in range(n_steps):
            trajectory[i + 1] = self.step(trajectory[i])
            
        return trajectory
    
    def _split_state(self, state):
        """
        Split a network state into subsystem states.
        
        Args:
            state (numpy.ndarray): Network state, shape (total_state_dim,).
            
        Returns:
            list: List of subsystem states.
        """
        subsystem_states = []
        idx = 0
        
        for dim in self.state_dims:
            subsystem_states.append(state[idx:idx+dim])
            idx += dim
            
        return subsystem_states
    
    def _split_input(self, input_vec):
        """
        Split a network input into subsystem inputs.
        
        Args:
            input_vec (numpy.ndarray): Network input, shape (total_input_dim,).
            
        Returns:
            list: List of subsystem inputs.
        """
        subsystem_inputs = []
        idx = 0
        
        for dim in self.input_dims:
            subsystem_inputs.append(input_vec[idx:idx+dim])
            idx += dim
            
        return subsystem_inputs
    
    def _combine_states(self, subsystem_states):
        """
        Combine subsystem states into a network state.
        
        Args:
            subsystem_states (list): List of subsystem states.
            
        Returns:
            numpy.ndarray: Network state, shape (total_state_dim,).
        """
        return np.concatenate(subsystem_states)
    
    def _construct_X_diagonal(self):
        """
        Construct the block diagonal matrix of X matrices.
        
        Returns:
            numpy.ndarray: Block diagonal matrix of X matrices.
        """
        # Initialize with zeros
        X_diag = np.zeros((self.total_input_dim + self.total_state_dim, 
                           self.total_input_dim + self.total_state_dim))
        
        # Get the X matrices for each subsystem
        X_matrices = [sys.X_matrix for sys in self.subsystems]
        
        # Check if all X matrices are defined
        if any(X is None for X in X_matrices):
            raise ValueError("X matrices are not defined for all subsystems. Call set_safety_parameters first.")
        
        # Fill the diagonal blocks
        input_idx = 0
        state_idx = self.total_input_dim
        
        for i, X in enumerate(X_matrices):
            input_dim = self.input_dims[i]
            state_dim = self.state_dims[i]
            
            # Get partitions of X
            X11 = X[:input_dim, :input_dim]
            X12 = X[:input_dim, input_dim:]
            X21 = X[input_dim:, :input_dim]
            X22 = X[input_dim:, input_dim:]
            
            # Place in the block diagonal matrix
            X_diag[input_idx:input_idx+input_dim, input_idx:input_idx+input_dim] = X11
            X_diag[input_idx:input_idx+input_dim, state_idx:state_idx+state_dim] = X12
            X_diag[state_idx:state_idx+state_dim, input_idx:input_idx+input_dim] = X21
            X_diag[state_idx:state_idx+state_dim, state_idx:state_idx+state_dim] = X22
            
            input_idx += input_dim
            state_idx += state_dim
            
        return X_diag
    
    def __str__(self):
        subsystem_str = ", ".join(str(sys.name) for sys in self.subsystems)
        return f"{self.name} (subsystems: {subsystem_str})" 