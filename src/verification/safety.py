import numpy as np
import matplotlib.pyplot as plt
from src.verification.barrier import BarrierVerifier

class SafetyVerifier:
    """
    High-level interface for safety verification.
    
    This class provides a user-friendly interface for safety verification
    with additional utilities for validation and visualization.
    """
    
    def __init__(self, network, barrier_representations, confidence=0.99, epsilon=1e-3, verbose=False):
        """
        Initialize the safety verifier.
        
        Args:
            network: Network object.
            barrier_representations (list): List of barrier function representations for each subsystem.
            confidence (float, optional): Desired confidence level (1 - beta). Defaults to 0.99.
            epsilon (float, optional): Tolerance parameter. Defaults to 1e-3.
            verbose (bool, optional): Whether to print progress. Defaults to False.
        """
        self.network = network
        self.barriers = barrier_representations
        self.confidence = confidence
        self.epsilon = epsilon
        self.verbose = verbose
        
        # Initialize barrier verifier
        self.barrier_verifier = BarrierVerifier(network, barrier_representations, confidence, epsilon, verbose)
        
        # Initialize results
        self.verification_result = None
        self.verification_confidence = None
        self.barrier_values = None
    
    def verify(self, **kwargs):
        """
        Verify the safety of the network.
        
        Args:
            **kwargs: Additional parameters for the barrier verifier.
            
        Returns:
            dict: Verification results.
        """
        result = self.barrier_verifier.verify(**kwargs)
        
        self.verification_result = result['verified']
        self.verification_confidence = result['confidence']
        
        return result
    
    def validate_simulation(self, n_simulations=100, n_steps=100, max_violations=0, true_dynamics=None):
        """
        Validate the verification result using simulations.
        
        Args:
            n_simulations (int, optional): Number of simulations to run. Defaults to 100.
            n_steps (int, optional): Number of steps per simulation. Defaults to 100.
            max_violations (int, optional): Maximum number of allowed safety violations. Defaults to 0.
            true_dynamics (callable, optional): True system dynamics for validation.
                If None, uses the network dynamics. Defaults to None.
                
        Returns:
            dict: Validation results.
        """
        if self.verification_result is None:
            raise ValueError("Run verify() first before validation.")
            
        # Run simulations
        violations = 0
        trajectories = []
        barrier_values = []
        
        for i in range(n_simulations):
            # Sample initial state from the initial set
            initial_state = self._sample_initial_state()
            
            # Simulate
            if true_dynamics is None:
                trajectory = self.network.simulate(initial_state, n_steps)
            else:
                trajectory = self._simulate_with_true_dynamics(initial_state, n_steps, true_dynamics)
                
            trajectories.append(trajectory)
            
            # Compute barrier values along the trajectory
            b_values = np.zeros(n_steps + 1)
            for t in range(n_steps + 1):
                b_values[t] = self.barrier_verifier.evaluate_barrier(trajectory[t])
                
            barrier_values.append(b_values)
            
            # Check for safety violations
            for t in range(n_steps + 1):
                if not self.network.is_safe(trajectory[t]):
                    violations += 1
                    break
        
        # Store barrier values
        self.barrier_values = barrier_values
        
        # Check if validation is successful
        validation_success = violations <= max_violations
        
        return {
            'success': validation_success,
            'violations': violations,
            'violation_rate': violations / n_simulations,
            'trajectories': trajectories,
            'barrier_values': barrier_values
        }
    
    def estimate_safe_region(self, n_samples=1000, grid_resolution=50):
        """
        Estimate the safe region certified by the barrier function.
        
        This method works for 1D or 2D systems by estimating the sublevel sets
        of the barrier function.
        
        Args:
            n_samples (int, optional): Number of samples to use. Defaults to 1000.
            grid_resolution (int, optional): Resolution of the grid for visualization. Defaults to 50.
            
        Returns:
            tuple: (grid, barrier_values) for visualization.
        """
        if self.verification_result is None:
            raise ValueError("Run verify() first before estimating safe region.")
            
        # For 1D or 2D, we can visualize the barrier function directly
        dim = self.network.total_state_dim
        
        if dim == 1:
            # 1D case
            x_min, x_max = self._get_state_bounds()[0]
            x_grid = np.linspace(x_min, x_max, grid_resolution)
            
            # Compute barrier values
            barrier_values = np.zeros(grid_resolution)
            for i in range(grid_resolution):
                barrier_values[i] = self.barrier_verifier.evaluate_barrier(np.array([x_grid[i]]))
                
            return x_grid, barrier_values
            
        elif dim == 2:
            # 2D case
            bounds = self._get_state_bounds()
            x_min, x_max = bounds[0]
            y_min, y_max = bounds[1]
            
            x_grid = np.linspace(x_min, x_max, grid_resolution)
            y_grid = np.linspace(y_min, y_max, grid_resolution)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Compute barrier values
            barrier_values = np.zeros((grid_resolution, grid_resolution))
            for i in range(grid_resolution):
                for j in range(grid_resolution):
                    barrier_values[i, j] = self.barrier_verifier.evaluate_barrier(np.array([X[i, j], Y[i, j]]))
                    
            return (X, Y), barrier_values
            
        else:
            # For higher dimensions, use sampling
            # Sample states from state space
            states = self._sample_states(n_samples)
            
            # Compute barrier values
            barrier_values = np.zeros(n_samples)
            for i in range(n_samples):
                barrier_values[i] = self.barrier_verifier.evaluate_barrier(states[i])
                
            return states, barrier_values
    
    def visualize_barrier(self, save_path=None):
        """
        Visualize the barrier function and safety regions.
        
        This method works for 1D or 2D systems.
        
        Args:
            save_path (str, optional): Path to save the figure. If None, shows the figure. Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: Figure object.
        """
        if self.verification_result is None:
            raise ValueError("Run verify() first before visualization.")
            
        # Estimate safe region
        grid, barrier_values = self.estimate_safe_region()
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        
        # For 1D, plot barrier function
        if self.network.total_state_dim == 1:
            plt.plot(grid, barrier_values, 'b-', linewidth=2)
            plt.axhline(y=1.0, color='r', linestyle='--', label=r'$B(x) = 1$')
            
            # Get initial and unsafe sets
            x_min, x_max = self._get_state_bounds()[0]
            initial_set = self.network.subsystems[0].initial_set[0]
            unsafe_set = self.network.subsystems[0].unsafe_set[0]
            
            # Plot sets
            plt.axvspan(initial_set[0], initial_set[1], alpha=0.2, color='g', label='Initial Set')
            plt.axvspan(unsafe_set[0], unsafe_set[1], alpha=0.2, color='r', label='Unsafe Set')
            
            # Plot safe region
            safe_mask = barrier_values < 1.0
            safe_x = grid[safe_mask]
            if len(safe_x) > 0:
                plt.axvspan(min(safe_x), max(safe_x), alpha=0.2, color='b', label='Safe Region')
            
            plt.xlabel('x')
            plt.ylabel('B(x)')
            plt.title('Barrier Function and Safety Regions')
            plt.legend()
            plt.grid(True)
            
        # For 2D, plot barrier level sets
        elif self.network.total_state_dim == 2:
            (X, Y), barrier_values = grid, barrier_values
            
            # Plot barrier level sets
            contour = plt.contourf(X, Y, barrier_values, 50, cmap='viridis', alpha=0.7)
            plt.colorbar(contour, label='B(x)')
            
            # Plot B(x) = 1 level set
            plt.contour(X, Y, barrier_values, levels=[1.0], colors='r', linewidths=2, linestyles='dashed')
            
            # Get initial and unsafe sets
            initial_sets = [sys.initial_set for sys in self.network.subsystems]
            unsafe_sets = [sys.unsafe_set for sys in self.network.subsystems]
            
            # Plot sets
            if len(self.network.subsystems) == 1:
                # Single 2D subsystem
                init_x = [initial_sets[0][0, 0], initial_sets[0][0, 1], initial_sets[0][0, 1], initial_sets[0][0, 0], initial_sets[0][0, 0]]
                init_y = [initial_sets[0][1, 0], initial_sets[0][1, 0], initial_sets[0][1, 1], initial_sets[0][1, 1], initial_sets[0][1, 0]]
                
                unsafe_x = [unsafe_sets[0][0, 0], unsafe_sets[0][0, 1], unsafe_sets[0][0, 1], unsafe_sets[0][0, 0], unsafe_sets[0][0, 0]]
                unsafe_y = [unsafe_sets[0][1, 0], unsafe_sets[0][1, 0], unsafe_sets[0][1, 1], unsafe_sets[0][1, 1], unsafe_sets[0][1, 0]]
                
                plt.plot(init_x, init_y, 'g-', linewidth=2, label='Initial Set')
                plt.plot(unsafe_x, unsafe_y, 'r-', linewidth=2, label='Unsafe Set')
            else:
                # Two 1D subsystems
                init_x = [initial_sets[0][0, 0], initial_sets[0][0, 1], initial_sets[0][0, 1], initial_sets[0][0, 0], initial_sets[0][0, 0]]
                init_y = [initial_sets[1][0, 0], initial_sets[1][0, 0], initial_sets[1][0, 1], initial_sets[1][0, 1], initial_sets[1][0, 0]]
                
                unsafe_x = [unsafe_sets[0][0, 0], unsafe_sets[0][0, 1], unsafe_sets[0][0, 1], unsafe_sets[0][0, 0], unsafe_sets[0][0, 0]]
                unsafe_y = [unsafe_sets[1][0, 0], unsafe_sets[1][0, 0], unsafe_sets[1][0, 1], unsafe_sets[1][0, 1], unsafe_sets[1][0, 0]]
                
                plt.plot(init_x, init_y, 'g-', linewidth=2, label='Initial Set')
                plt.plot(unsafe_x, unsafe_y, 'r-', linewidth=2, label='Unsafe Set')
            
            plt.xlabel('x₁')
            plt.ylabel('x₂')
            plt.title('Barrier Function Level Sets')
            plt.legend()
            plt.grid(True)
            
        else:
            # For higher dimensions, we can't visualize directly
            # Plot histogram of barrier values
            states, barrier_values = grid, barrier_values
            
            plt.hist(barrier_values, bins=50, alpha=0.7, color='b')
            plt.axvline(x=1.0, color='r', linestyle='--', label=r'$B(x) = 1$')
            
            plt.xlabel('B(x)')
            plt.ylabel('Frequency')
            plt.title('Histogram of Barrier Function Values')
            plt.legend()
            plt.grid(True)
        
        # Save or show figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_simulation(self, n_simulations=5, n_steps=100, save_path=None):
        """
        Visualize simulations and barrier function values.
        
        This method works for 1D or 2D systems.
        
        Args:
            n_simulations (int, optional): Number of simulations to run. Defaults to 5.
            n_steps (int, optional): Number of steps per simulation. Defaults to 100.
            save_path (str, optional): Path to save the figure. If None, shows the figure. Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: Figure object.
        """
        # Run simulations
        trajectories = []
        barrier_values = []
        
        for i in range(n_simulations):
            # Sample initial state from the initial set
            initial_state = self._sample_initial_state()
            
            # Simulate
            trajectory = self.network.simulate(initial_state, n_steps)
            trajectories.append(trajectory)
            
            # Compute barrier values along the trajectory
            b_values = np.zeros(n_steps + 1)
            for t in range(n_steps + 1):
                b_values[t] = self.barrier_verifier.evaluate_barrier(trajectory[t])
                
            barrier_values.append(b_values)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        
        # For 1D system
        if self.network.total_state_dim == 1:
            # Plot state trajectories
            plt.subplot(2, 1, 1)
            for i in range(n_simulations):
                plt.plot(np.arange(n_steps + 1), trajectories[i], alpha=0.7)
                
            # Get initial and unsafe sets
            initial_set = self.network.subsystems[0].initial_set[0]
            unsafe_set = self.network.subsystems[0].unsafe_set[0]
            
            # Plot sets
            plt.axhspan(initial_set[0], initial_set[1], alpha=0.2, color='g', label='Initial Set')
            plt.axhspan(unsafe_set[0], unsafe_set[1], alpha=0.2, color='r', label='Unsafe Set')
            
            plt.xlabel('Time Step')
            plt.ylabel('x')
            plt.title('State Trajectories')
            plt.legend()
            plt.grid(True)
            
            # Plot barrier values
            plt.subplot(2, 1, 2)
            for i in range(n_simulations):
                plt.plot(np.arange(n_steps + 1), barrier_values[i], alpha=0.7)
                
            plt.axhline(y=1.0, color='r', linestyle='--', label=r'$B(x) = 1$')
            
            plt.xlabel('Time Step')
            plt.ylabel('B(x)')
            plt.title('Barrier Function Values')
            plt.legend()
            plt.grid(True)
            
        # For 2D system
        elif self.network.total_state_dim == 2:
            # Plot state trajectories
            plt.subplot(2, 2, 1)
            for i in range(n_simulations):
                plt.plot(trajectories[i][:, 0], trajectories[i][:, 1], alpha=0.7)
                plt.plot(trajectories[i][0, 0], trajectories[i][0, 1], 'go', markersize=5)  # Initial point
                
            # Get initial and unsafe sets
            initial_sets = [sys.initial_set for sys in self.network.subsystems]
            unsafe_sets = [sys.unsafe_set for sys in self.network.subsystems]
            
            # Plot sets
            if len(self.network.subsystems) == 1:
                # Single 2D subsystem
                init_x = [initial_sets[0][0, 0], initial_sets[0][0, 1], initial_sets[0][0, 1], initial_sets[0][0, 0], initial_sets[0][0, 0]]
                init_y = [initial_sets[0][1, 0], initial_sets[0][1, 0], initial_sets[0][1, 1], initial_sets[0][1, 1], initial_sets[0][1, 0]]
                
                unsafe_x = [unsafe_sets[0][0, 0], unsafe_sets[0][0, 1], unsafe_sets[0][0, 1], unsafe_sets[0][0, 0], unsafe_sets[0][0, 0]]
                unsafe_y = [unsafe_sets[0][1, 0], unsafe_sets[0][1, 0], unsafe_sets[0][1, 1], unsafe_sets[0][1, 1], unsafe_sets[0][1, 0]]
                
                plt.plot(init_x, init_y, 'g-', linewidth=2, label='Initial Set')
                plt.plot(unsafe_x, unsafe_y, 'r-', linewidth=2, label='Unsafe Set')
            else:
                # Two 1D subsystems
                init_x = [initial_sets[0][0, 0], initial_sets[0][0, 1], initial_sets[0][0, 1], initial_sets[0][0, 0], initial_sets[0][0, 0]]
                init_y = [initial_sets[1][0, 0], initial_sets[1][0, 0], initial_sets[1][0, 1], initial_sets[1][0, 1], initial_sets[1][0, 0]]
                
                unsafe_x = [unsafe_sets[0][0, 0], unsafe_sets[0][0, 1], unsafe_sets[0][0, 1], unsafe_sets[0][0, 0], unsafe_sets[0][0, 0]]
                unsafe_y = [unsafe_sets[1][0, 0], unsafe_sets[1][0, 0], unsafe_sets[1][0, 1], unsafe_sets[1][0, 1], unsafe_sets[1][0, 0]]
                
                plt.plot(init_x, init_y, 'g-', linewidth=2, label='Initial Set')
                plt.plot(unsafe_x, unsafe_y, 'r-', linewidth=2, label='Unsafe Set')
            
            plt.xlabel('x₁')
            plt.ylabel('x₂')
            plt.title('State Trajectories')
            plt.legend()
            plt.grid(True)
            
            # Plot component trajectories
            plt.subplot(2, 2, 2)
            for i in range(n_simulations):
                plt.plot(np.arange(n_steps + 1), trajectories[i][:, 0], alpha=0.7)
                
            plt.xlabel('Time Step')
            plt.ylabel('x₁')
            plt.title('x₁ Trajectories')
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            for i in range(n_simulations):
                plt.plot(np.arange(n_steps + 1), trajectories[i][:, 1], alpha=0.7)
                
            plt.xlabel('Time Step')
            plt.ylabel('x₂')
            plt.title('x₂ Trajectories')
            plt.grid(True)
            
            # Plot barrier values
            plt.subplot(2, 2, 4)
            for i in range(n_simulations):
                plt.plot(np.arange(n_steps + 1), barrier_values[i], alpha=0.7)
                
            plt.axhline(y=1.0, color='r', linestyle='--', label=r'$B(x) = 1$')
            
            plt.xlabel('Time Step')
            plt.ylabel('B(x)')
            plt.title('Barrier Function Values')
            plt.legend()
            plt.grid(True)
            
        else:
            # For higher dimensions, plot components and barrier values
            plt.subplot(2, 1, 1)
            for i in range(n_simulations):
                for j in range(min(3, self.network.total_state_dim)):  # Show at most 3 components
                    plt.plot(np.arange(n_steps + 1), trajectories[i][:, j], alpha=0.7, 
                             label=f'x_{j+1} (sim {i+1})' if i == 0 else "")
                    
            plt.xlabel('Time Step')
            plt.ylabel('State Components')
            plt.title('State Trajectories')
            plt.legend()
            plt.grid(True)
            
            # Plot barrier values
            plt.subplot(2, 1, 2)
            for i in range(n_simulations):
                plt.plot(np.arange(n_steps + 1), barrier_values[i], alpha=0.7, label=f'Simulation {i+1}')
                
            plt.axhline(y=1.0, color='r', linestyle='--', label=r'$B(x) = 1$')
            
            plt.xlabel('Time Step')
            plt.ylabel('B(x)')
            plt.title('Barrier Function Values')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save or show figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _sample_initial_state(self):
        """
        Sample a state from the initial set.
        
        Returns:
            numpy.ndarray: Sampled state.
        """
        subsystem_states = []
        
        for subsystem in self.network.subsystems:
            # Sample from initial set
            if subsystem.initial_set is None:
                raise ValueError("Initial set not defined for all subsystems.")
                
            state = np.zeros(subsystem.state_dim)
            for i in range(subsystem.state_dim):
                state[i] = np.random.uniform(subsystem.initial_set[i, 0], subsystem.initial_set[i, 1])
                
            subsystem_states.append(state)
        
        # Combine states
        return np.concatenate(subsystem_states)
    
    def _sample_states(self, n_samples):
        """
        Sample states from the state space.
        
        Args:
            n_samples (int): Number of states to sample.
            
        Returns:
            numpy.ndarray: Sampled states, shape (n_samples, total_state_dim).
        """
        states = np.zeros((n_samples, self.network.total_state_dim))
        
        for i in range(n_samples):
            subsystem_states = []
            
            for subsystem in self.network.subsystems:
                state = np.zeros(subsystem.state_dim)
                for j in range(subsystem.state_dim):
                    state[j] = np.random.uniform(subsystem.state_bounds[j, 0], subsystem.state_bounds[j, 1])
                    
                subsystem_states.append(state)
            
            states[i] = np.concatenate(subsystem_states)
        
        return states
    
    def _get_state_bounds(self):
        """
        Get the bounds of the state space.
        
        Returns:
            numpy.ndarray: Bounds of the state space, shape (total_state_dim, 2).
        """
        bounds = []
        
        for subsystem in self.network.subsystems:
            bounds.append(subsystem.state_bounds)
        
        return np.vstack(bounds)
    
    def _simulate_with_true_dynamics(self, initial_state, n_steps, true_dynamics):
        """
        Simulate with the true system dynamics.
        
        Args:
            initial_state (numpy.ndarray): Initial state.
            n_steps (int): Number of steps.
            true_dynamics (callable): True system dynamics function.
            
        Returns:
            numpy.ndarray: Trajectory of states, shape (n_steps+1, total_state_dim).
        """
        # Initialize trajectory
        trajectory = np.zeros((n_steps + 1, self.network.total_state_dim))
        trajectory[0] = initial_state
        
        # Simulate
        for i in range(n_steps):
            trajectory[i + 1] = true_dynamics(trajectory[i])
            
        return trajectory 