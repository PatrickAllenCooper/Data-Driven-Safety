import numpy as np
from abc import ABC, abstractmethod

class Subsystem(ABC):
    """
    Abstract base class for discrete-time subsystems.
    
    This class defines the interface for subsystems in the compositional
    safety verification framework.
    """
    
    def __init__(self, state_dim, input_dim, state_bounds, input_bounds, name=None):
        """
        Initialize the subsystem.
        
        Args:
            state_dim (int): Dimension of the state space.
            input_dim (int): Dimension of the input space.
            state_bounds (numpy.ndarray): Bounds of the state space, shape (state_dim, 2).
            input_bounds (numpy.ndarray): Bounds of the input space, shape (input_dim, 2).
            name (str, optional): Name of the subsystem.
        """
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.state_bounds = state_bounds
        self.input_bounds = input_bounds
        self.name = name if name is not None else f"Subsystem_{id(self)}"
        
        # Initialize state and safety sets
        self.initial_set = None
        self.unsafe_set = None
        
        # Sub-barrier function and parameters
        self.X_matrix = None  # Dissipation matrix
        self.gamma = None     # Safety threshold
    
    @abstractmethod
    def step(self, state, input_vec):
        """
        Compute one step of the subsystem dynamics.
        
        Args:
            state (numpy.ndarray): Current state, shape (state_dim,).
            input_vec (numpy.ndarray): Current input, shape (input_dim,).
            
        Returns:
            numpy.ndarray: Next state, shape (state_dim,).
        """
        pass
    
    def generate_data(self, n_samples, from_initial=False):
        """
        Generate data points from the subsystem.
        
        Args:
            n_samples (int): Number of samples to generate.
            from_initial (bool, optional): Whether to sample from the initial set.
                If False, samples from the entire state space. Defaults to False.
                
        Returns:
            tuple: Tuple of (states, inputs, next_states), where:
                - states: States, shape (n_samples, state_dim).
                - inputs: Inputs, shape (n_samples, input_dim).
                - next_states: Next states, shape (n_samples, state_dim).
        """
        # Sample states
        if from_initial and self.initial_set is not None:
            # Sample from initial set
            states = self._sample_from_set(self.initial_set, n_samples)
        else:
            # Sample from entire state space
            states = self._sample_uniform(n_samples, self.state_bounds)
        
        # Sample inputs
        inputs = self._sample_uniform(n_samples, self.input_bounds)
        
        # Compute next states
        next_states = np.zeros((n_samples, self.state_dim))
        for i in range(n_samples):
            next_states[i] = self.step(states[i], inputs[i])
            
        return states, inputs, next_states
    
    def is_in_initial_set(self, state):
        """
        Check if a state is in the initial set.
        
        Args:
            state (numpy.ndarray): State to check, shape (state_dim,).
            
        Returns:
            bool: True if the state is in the initial set, False otherwise.
        """
        if self.initial_set is None:
            raise ValueError("Initial set is not defined.")
            
        # Check if state is in the initial set
        return np.all(state >= self.initial_set[:, 0]) and np.all(state <= self.initial_set[:, 1])
    
    def is_in_unsafe_set(self, state):
        """
        Check if a state is in the unsafe set.
        
        Args:
            state (numpy.ndarray): State to check, shape (state_dim,).
            
        Returns:
            bool: True if the state is in the unsafe set, False otherwise.
        """
        if self.unsafe_set is None:
            raise ValueError("Unsafe set is not defined.")
            
        # Check if state is in the unsafe set
        return np.all(state >= self.unsafe_set[:, 0]) and np.all(state <= self.unsafe_set[:, 1])
    
    def set_initial_set(self, bounds):
        """
        Set the initial set.
        
        Args:
            bounds (numpy.ndarray): Bounds of the initial set, shape (state_dim, 2).
        """
        self.initial_set = bounds
    
    def set_unsafe_set(self, bounds):
        """
        Set the unsafe set.
        
        Args:
            bounds (numpy.ndarray): Bounds of the unsafe set, shape (state_dim, 2).
        """
        self.unsafe_set = bounds
    
    def set_safety_parameters(self, X_matrix, gamma):
        """
        Set the safety parameters.
        
        Args:
            X_matrix (numpy.ndarray): Dissipation matrix, shape (input_dim+state_dim, input_dim+state_dim).
            gamma (float): Safety threshold.
        """
        self.X_matrix = X_matrix
        self.gamma = gamma
    
    def _sample_uniform(self, n_samples, bounds):
        """
        Sample uniformly from a hyperrectangle.
        
        Args:
            n_samples (int): Number of samples to generate.
            bounds (numpy.ndarray): Bounds of the hyperrectangle, shape (dim, 2).
            
        Returns:
            numpy.ndarray: Sampled points, shape (n_samples, dim).
        """
        dim = bounds.shape[0]
        samples = np.zeros((n_samples, dim))
        
        for i in range(dim):
            samples[:, i] = np.random.uniform(bounds[i, 0], bounds[i, 1], n_samples)
            
        return samples
    
    def _sample_from_set(self, bounds, n_samples):
        """
        Sample uniformly from a hyperrectangle.
        
        Args:
            bounds (numpy.ndarray): Bounds of the hyperrectangle, shape (dim, 2).
            n_samples (int): Number of samples to generate.
            
        Returns:
            numpy.ndarray: Sampled points, shape (n_samples, dim).
        """
        return self._sample_uniform(n_samples, bounds)
    
    def __str__(self):
        return f"{self.name} (state_dim={self.state_dim}, input_dim={self.input_dim})"


class LinearSubsystem(Subsystem):
    """
    Linear discrete-time subsystem.
    
    The dynamics are of the form:
    x_{k+1} = A x_k + B w_k + c
    
    where x_k is the state, w_k is the input, and c is a constant term.
    """
    
    def __init__(self, A, B, c=None, state_bounds=None, input_bounds=None, name=None):
        """
        Initialize the linear subsystem.
        
        Args:
            A (numpy.ndarray): State transition matrix, shape (state_dim, state_dim).
            B (numpy.ndarray): Input matrix, shape (state_dim, input_dim).
            c (numpy.ndarray, optional): Constant term, shape (state_dim,). Defaults to None.
            state_bounds (numpy.ndarray, optional): Bounds of the state space, shape (state_dim, 2).
                If None, uses [-1, 1] for each dimension.
            input_bounds (numpy.ndarray, optional): Bounds of the input space, shape (input_dim, 2).
                If None, uses [-1, 1] for each dimension.
            name (str, optional): Name of the subsystem.
        """
        self.A = A
        self.B = B
        
        state_dim = A.shape[0]
        input_dim = B.shape[1]
        
        # Set default bounds if not provided
        if state_bounds is None:
            state_bounds = np.array([[-1.0, 1.0]] * state_dim)
        if input_bounds is None:
            input_bounds = np.array([[-1.0, 1.0]] * input_dim)
        
        super().__init__(state_dim, input_dim, state_bounds, input_bounds, name)
        
        # Set constant term
        self.c = np.zeros(state_dim) if c is None else c
    
    def step(self, state, input_vec):
        """
        Compute one step of the linear subsystem dynamics.
        
        Args:
            state (numpy.ndarray): Current state, shape (state_dim,).
            input_vec (numpy.ndarray): Current input, shape (input_dim,).
            
        Returns:
            numpy.ndarray: Next state, shape (state_dim,).
        """
        return self.A @ state + self.B @ input_vec + self.c


class NonlinearSubsystem(Subsystem):
    """
    Nonlinear discrete-time subsystem.
    
    The dynamics are defined by a user-provided function.
    """
    
    def __init__(self, dynamics_fn, state_dim, input_dim, state_bounds=None, input_bounds=None, name=None):
        """
        Initialize the nonlinear subsystem.
        
        Args:
            dynamics_fn (callable): Dynamics function that takes (state, input) and returns next state.
            state_dim (int): Dimension of the state space.
            input_dim (int): Dimension of the input space.
            state_bounds (numpy.ndarray, optional): Bounds of the state space, shape (state_dim, 2).
                If None, uses [-1, 1] for each dimension.
            input_bounds (numpy.ndarray, optional): Bounds of the input space, shape (input_dim, 2).
                If None, uses [-1, 1] for each dimension.
            name (str, optional): Name of the subsystem.
        """
        # Set default bounds if not provided
        if state_bounds is None:
            state_bounds = np.array([[-1.0, 1.0]] * state_dim)
        if input_bounds is None:
            input_bounds = np.array([[-1.0, 1.0]] * input_dim)
        
        super().__init__(state_dim, input_dim, state_bounds, input_bounds, name)
        
        # Store dynamics function
        self.dynamics_fn = dynamics_fn
    
    def step(self, state, input_vec):
        """
        Compute one step of the nonlinear subsystem dynamics.
        
        Args:
            state (numpy.ndarray): Current state, shape (state_dim,).
            input_vec (numpy.ndarray): Current input, shape (input_dim,).
            
        Returns:
            numpy.ndarray: Next state, shape (state_dim,).
        """
        return self.dynamics_fn(state, input_vec)


class UnknownSubsystem(Subsystem):
    """
    Unknown discrete-time subsystem.
    
    For this type of subsystem, the dynamics are not explicitly known.
    We can only collect data points from the system, but we cannot simulate it directly.
    """
    
    def __init__(self, state_dim, input_dim, state_bounds=None, input_bounds=None, name=None):
        """
        Initialize the unknown subsystem.
        
        Args:
            state_dim (int): Dimension of the state space.
            input_dim (int): Dimension of the input space.
            state_bounds (numpy.ndarray, optional): Bounds of the state space, shape (state_dim, 2).
                If None, uses [-1, 1] for each dimension.
            input_bounds (numpy.ndarray, optional): Bounds of the input space, shape (input_dim, 2).
                If None, uses [-1, 1] for each dimension.
            name (str, optional): Name of the subsystem.
        """
        # Set default bounds if not provided
        if state_bounds is None:
            state_bounds = np.array([[-1.0, 1.0]] * state_dim)
        if input_bounds is None:
            input_bounds = np.array([[-1.0, 1.0]] * input_dim)
            
        super().__init__(state_dim, input_dim, state_bounds, input_bounds, name)
        
        # Store data samples
        self.data_states = None
        self.data_inputs = None
        self.data_next_states = None
    
    def step(self, state, input_vec):
        """
        This method is not available for unknown subsystems.
        
        Raises:
            NotImplementedError: Always raised for unknown subsystems.
        """
        raise NotImplementedError("Cannot step an unknown subsystem. Use generate_data with provided data instead.")
    
    def set_data(self, states, inputs, next_states):
        """
        Set the data for the unknown subsystem.
        
        Args:
            states (numpy.ndarray): States, shape (n_samples, state_dim).
            inputs (numpy.ndarray): Inputs, shape (n_samples, input_dim).
            next_states (numpy.ndarray): Next states, shape (n_samples, state_dim).
        """
        # Check dimensions
        assert states.shape[1] == self.state_dim, f"Expected states with {self.state_dim} dimensions, got {states.shape[1]}"
        assert inputs.shape[1] == self.input_dim, f"Expected inputs with {self.input_dim} dimensions, got {inputs.shape[1]}"
        assert next_states.shape[1] == self.state_dim, f"Expected next_states with {self.state_dim} dimensions, got {next_states.shape[1]}"
        assert states.shape[0] == inputs.shape[0] == next_states.shape[0], "Number of samples must be the same for states, inputs, and next_states"
        
        # Store data
        self.data_states = states
        self.data_inputs = inputs
        self.data_next_states = next_states
    
    def generate_data(self, n_samples, from_initial=False):
        """
        Return a subset of the stored data.
        
        Args:
            n_samples (int): Number of samples to return.
            from_initial (bool, optional): Not used for unknown subsystems.
                
        Returns:
            tuple: Tuple of (states, inputs, next_states).
        
        Raises:
            ValueError: If data has not been set or if n_samples is larger than the available data.
        """
        if self.data_states is None:
            raise ValueError("Data has not been set. Call set_data first.")
        
        # Check if we have enough data
        if n_samples > self.data_states.shape[0]:
            raise ValueError(f"Requested {n_samples} samples, but only {self.data_states.shape[0]} available.")
        
        # Randomly select n_samples from the data
        indices = np.random.choice(self.data_states.shape[0], n_samples, replace=False)
        
        return (
            self.data_states[indices],
            self.data_inputs[indices],
            self.data_next_states[indices]
        ) 