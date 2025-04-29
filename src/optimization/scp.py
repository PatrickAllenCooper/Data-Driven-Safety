import numpy as np
import cvxpy as cp
from abc import ABC, abstractmethod

class ScenarioProgram(ABC):
    """
    Abstract base class for scenario convex programs.
    
    This class defines the interface for scenario convex programs
    used in data-driven safety verification.
    """
    
    def __init__(self, subsystem, barrier_representation, epsilon=1e-3):
        """
        Initialize the scenario program.
        
        Args:
            subsystem: Subsystem object.
            barrier_representation: Barrier function representation.
            epsilon (float, optional): Tolerance parameter. Defaults to 1e-3.
        """
        self.subsystem = subsystem
        self.barrier = barrier_representation
        self.epsilon = epsilon
        
        # Store results
        self.optimal_value = None
        self.optimal_params = None
        self.optimal_X = None
        self.optimal_gamma = None
    
    @abstractmethod
    def solve(self, data, **kwargs):
        """
        Solve the scenario program.
        
        Args:
            data: Data tuple (states, inputs, next_states).
            **kwargs: Additional parameters.
            
        Returns:
            dict: Results dictionary.
        """
        pass
    
    def compute_minimum_samples(self, confidence, epsilon=None):
        """
        Compute the minimum number of samples required for the given confidence level.
        
        Args:
            confidence (float): Desired confidence level (1 - beta).
            epsilon (float, optional): Tolerance parameter. If None, uses self.epsilon.
            
        Returns:
            int: Minimum number of samples.
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        # Get number of decision variables
        n_params = self.barrier.num_parameters()
        
        # For X matrix, we have (input_dim + state_dim)^2 variables
        n_X = (self.subsystem.input_dim + self.subsystem.state_dim) ** 2
        
        # Plus 1 for gamma
        n_vars = n_params + 1 + n_X
        
        # Compute minimum number of samples using the formula from the paper
        beta = 1 - confidence
        N = self._compute_N(beta, n_vars, epsilon)
        
        return N
    
    def _compute_N(self, beta, n_vars, epsilon):
        """
        Compute the minimum number of samples using the formula from Theorem 2.
        
        Args:
            beta (float): Confidence parameter (failure probability).
            n_vars (int): Number of decision variables.
            epsilon (float): Tolerance parameter.
            
        Returns:
            int: Minimum number of samples.
        """
        # Estimate Lipschitz constant
        # This is a simplification; in practice, this should be computed more accurately
        L = 10.0  # Placeholder
        
        # Compute normalized epsilon
        eps_normalized = (epsilon / L) ** n_vars
        
        # Find smallest N such that sum_{j=0}^{n_vars} (N choose j) * eps_normalized^j * (1-eps_normalized)^(N-j) <= beta
        N = n_vars
        while True:
            # Compute the sum
            prob_sum = 0
            for j in range(n_vars + 1):
                prob_sum += self._binom(N, j) * (eps_normalized ** j) * ((1 - eps_normalized) ** (N - j))
                
            if prob_sum <= beta:
                break
                
            N += 1
            
        return N
    
    def _binom(self, n, k):
        """
        Compute binomial coefficient (n choose k).
        
        Args:
            n (int): Number of elements.
            k (int): Number of elements to choose.
            
        Returns:
            float: Binomial coefficient.
        """
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
            
        # Compute using the formula
        result = 1
        for i in range(1, k + 1):
            result = result * (n - (i - 1)) / i
            
        return result


class LinearSCP(ScenarioProgram):
    """
    Scenario convex program for linear barrier function representations.
    
    This class implements the scenario convex program for barrier functions
    that are linear in their parameters (e.g., polynomial, RBF, Fourier, wavelet).
    """
    
    def solve(self, data, verbose=False):
        """
        Solve the scenario program.
        
        Args:
            data: Data tuple (states, inputs, next_states).
            verbose (bool, optional): Whether to print progress. Defaults to False.
            
        Returns:
            dict: Results dictionary.
        """
        # Extract data
        states, inputs, next_states = data
        n_samples = states.shape[0]
        
        # Extract dimensions
        n_x = self.subsystem.state_dim
        n_w = self.subsystem.input_dim
        n_params = self.barrier.num_parameters()
        
        # Define variables
        eta = cp.Variable()  # Slack variable
        gamma = cp.Variable()  # Safety threshold
        q = cp.Variable(n_params)  # Barrier function parameters
        X = cp.Variable((n_w + n_x, n_w + n_x), symmetric=True)  # Dissipation matrix
        
        # Add constraints
        constraints = [
            gamma <= 0.95,  # gamma < 1
            eta >= 0,       # Non-negative slack
        ]
        
        # Sample constraints
        for i in range(n_samples):
            x_i = states[i]
            w_i = inputs[i]
            x_next_i = next_states[i]
            z_i = np.concatenate((w_i, x_i))
            
            # Check if state is in initial or unsafe set
            in_initial = self.subsystem.is_in_initial_set(x_i)
            in_unsafe = self.subsystem.is_in_unsafe_set(x_i)
            
            # Compute barrier values at current and next state
            barrier_vals_i = self._compute_barrier_values(x_i)
            barrier_vals_next_i = self._compute_barrier_values(x_next_i)
            
            # Barrier value is q^T * barrier_vals
            B_i = q @ barrier_vals_i
            B_next_i = q @ barrier_vals_next_i
            
            # Add constraint for initial set: B(x) <= gamma if x in X_0
            if in_initial:
                constraints.append(B_i - gamma <= eta)
                
            # Add constraint for unsafe set: B(x) >= 1 if x in X_u
            if in_unsafe:
                constraints.append(-B_i + 1 <= eta)
                
            # Add constraint for decay: B(x^+) <= B(x) + z^T X z for all x, w
            constraints.append(B_next_i - B_i - cp.quad_form(z_i, X) <= eta)
        
        # Define objective
        objective = cp.Minimize(eta)
        
        # Solve problem
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=verbose)
        
        # Store results
        self.optimal_value = result
        self.optimal_params = q.value
        self.optimal_X = X.value
        self.optimal_gamma = gamma.value
        
        # Set parameters in barrier function
        self.barrier.set_parameters(self.optimal_params)
        
        # Set safety parameters in subsystem
        self.subsystem.set_safety_parameters(self.optimal_X, self.optimal_gamma)
        
        # Return results
        return {
            'optimal_value': result,
            'optimal_params': q.value,
            'optimal_X': X.value,
            'optimal_gamma': gamma.value,
            'status': prob.status
        }
    
    def _compute_barrier_values(self, state):
        """
        Compute the values of basis functions at a given state.
        
        Args:
            state (numpy.ndarray): State vector, shape (state_dim,).
            
        Returns:
            numpy.ndarray: Values of basis functions, shape (n_params,).
        """
        # For linear representations, we need to compute the values of the basis functions
        # without the parameters (since parameters are handled separately in the CP)
        
        # This depends on the barrier representation being used
        if hasattr(self.barrier, '_compute_multivariate_basis'):
            # For polynomial basis
            return self.barrier._compute_multivariate_basis(state.reshape(1, -1))[0]
        elif hasattr(self.barrier, '_tensor_product'):
            # For Fourier basis
            x_norm = self.barrier._normalize_input(state.reshape(1, -1))
            if self.barrier.dimension == 1:
                return self.barrier._compute_1d_basis(x_norm[:, 0])[0]
            else:
                bases = []
                for i in range(self.barrier.dimension):
                    bases.append(self.barrier._compute_1d_basis(x_norm[:, i]))
                return self.barrier._tensor_product(bases)[0]
        elif hasattr(self.barrier, '_interpolate_basis'):
            # For wavelet basis
            bases = []
            for i in range(self.barrier.dimension):
                x_clipped = np.clip(state[i], self.barrier.domain_bounds[i, 0], self.barrier.domain_bounds[i, 1])
                basis_i = self.barrier._interpolate_basis(np.array([x_clipped]), i)
                bases.append(basis_i)
            
            if self.barrier.dimension == 1:
                return bases[0][0]
            else:
                return self.barrier._tensor_product(bases)[0]
        elif hasattr(self.barrier, 'rbf'):
            # For RBF basis
            x = state.reshape(1, -1)
            result = np.zeros(self.barrier.n_centers + 1)
            
            for i, center in enumerate(self.barrier.centers):
                result[i] = self.barrier.rbf(x, center)
                
            # Last element is the offset (bias term)
            result[-1] = 1.0
            
            return result
        else:
            raise NotImplementedError("Barrier representation not supported in LinearSCP.")


class NeuralSCP(ScenarioProgram):
    """
    Scenario program for neural network barrier representations.
    
    This class implements a modified scenario program for neural network barrier functions,
    which are nonlinear in their parameters and require special handling.
    """
    
    def __init__(self, subsystem, barrier_representation, epsilon=1e-3, learning_rate=0.01, epochs=1000):
        """
        Initialize the neural SCP.
        
        Args:
            subsystem: Subsystem object.
            barrier_representation: Neural network barrier representation.
            epsilon (float, optional): Tolerance parameter. Defaults to 1e-3.
            learning_rate (float, optional): Learning rate for SGD. Defaults to 0.01.
            epochs (int, optional): Number of epochs. Defaults to 1000.
        """
        super().__init__(subsystem, barrier_representation, epsilon)
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def solve(self, data, verbose=False):
        """
        Solve the scenario program using gradient-based optimization.
        
        For neural networks, we use a two-stage approach:
        1. Train the neural network to satisfy the barrier conditions
        2. Solve a convex program to find the X matrix
        
        Args:
            data: Data tuple (states, inputs, next_states).
            verbose (bool, optional): Whether to print progress. Defaults to False.
            
        Returns:
            dict: Results dictionary.
        """
        import torch
        import torch.optim as optim
        
        # Extract data
        states, inputs, next_states = data
        n_samples = states.shape[0]
        
        # Extract dimensions
        n_x = self.subsystem.state_dim
        n_w = self.subsystem.input_dim
        
        # Convert data to torch tensors
        states_torch = torch.tensor(states, dtype=torch.float32)
        inputs_torch = torch.tensor(inputs, dtype=torch.float32)
        next_states_torch = torch.tensor(next_states, dtype=torch.float32)
        
        # Initialize parameters
        self.barrier.model.train()
        optimizer = optim.Adam(self.barrier.model.parameters(), lr=self.learning_rate)
        
        # Initialize gamma
        gamma = torch.tensor(0.95, dtype=torch.float32, requires_grad=True)
        gamma_optimizer = optim.Adam([gamma], lr=0.001)
        
        # Training loop
        best_loss = float('inf')
        best_params = None
        
        for epoch in range(self.epochs):
            # Zero gradients
            optimizer.zero_grad()
            gamma_optimizer.zero_grad()
            
            # Compute barrier values
            B = self.barrier.model(states_torch).squeeze()
            B_next = self.barrier.model(next_states_torch).squeeze()
            
            # Compute losses
            loss_initial = 0
            loss_unsafe = 0
            loss_decay = 0
            count_initial = 0
            count_unsafe = 0
            
            for i in range(n_samples):
                x_i = states[i]
                
                # Check if state is in initial or unsafe set
                in_initial = self.subsystem.is_in_initial_set(x_i)
                in_unsafe = self.subsystem.is_in_unsafe_set(x_i)
                
                # Add loss for initial set: B(x) <= gamma if x in X_0
                if in_initial:
                    loss_initial += torch.relu(B[i] - gamma)
                    count_initial += 1
                    
                # Add loss for unsafe set: B(x) >= 1 if x in X_u
                if in_unsafe:
                    loss_unsafe += torch.relu(1 - B[i])
                    count_unsafe += 1
                    
                # Add loss for decay: B(x^+) <= B(x) for all x, w
                # Note: We'll handle the z^T X z term in the second stage
                loss_decay += torch.relu(B_next[i] - B[i])
            
            # Normalize losses by counts
            if count_initial > 0:
                loss_initial /= count_initial
            if count_unsafe > 0:
                loss_unsafe /= count_unsafe
            loss_decay /= n_samples
            
            # Combined loss
            loss = loss_initial + loss_unsafe + loss_decay
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            gamma_optimizer.step()
            
            # Project gamma to [0, 0.99]
            with torch.no_grad():
                gamma.clamp_(0, 0.99)
            
            # Print progress
            if verbose and (epoch + 1) % (self.epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}, Gamma: {gamma.item():.6f}")
                print(f"  Initial: {loss_initial.item():.6f}, Unsafe: {loss_unsafe.item():.6f}, Decay: {loss_decay.item():.6f}")
            
            # Save best parameters
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = [p.detach().clone() for p in self.barrier.model.parameters()]
                best_gamma = gamma.item()
        
        # Set best parameters
        with torch.no_grad():
            for param, best_param in zip(self.barrier.model.parameters(), best_params):
                param.copy_(best_param)
                
        # Update barrier parameters
        self.barrier.parameters = self.barrier._get_model_params().detach().numpy()
        self.optimal_params = self.barrier.parameters
        self.optimal_gamma = best_gamma
        
        # Second stage: Solve a convex program to find the X matrix
        X = self._solve_X_matrix(data)
        self.optimal_X = X
        
        # Set safety parameters in subsystem
        self.subsystem.set_safety_parameters(self.optimal_X, self.optimal_gamma)
        
        # Return results
        return {
            'optimal_value': best_loss,
            'optimal_params': self.optimal_params,
            'optimal_X': self.optimal_X,
            'optimal_gamma': self.optimal_gamma,
            'status': 'optimal' if best_loss < self.epsilon else 'suboptimal'
        }
    
    def _solve_X_matrix(self, data):
        """
        Solve for the X matrix given fixed barrier parameters.
        
        Args:
            data: Data tuple (states, inputs, next_states).
            
        Returns:
            numpy.ndarray: Optimal X matrix.
        """
        # Extract data
        states, inputs, next_states = data
        n_samples = states.shape[0]
        
        # Extract dimensions
        n_x = self.subsystem.state_dim
        n_w = self.subsystem.input_dim
        
        # Define variables for X matrix
        X = cp.Variable((n_w + n_x, n_w + n_x), symmetric=True)
        eta = cp.Variable()  # Slack variable
        
        # Compute barrier values with fixed parameters
        B_values = np.zeros(n_samples)
        B_next_values = np.zeros(n_samples)
        
        for i in range(n_samples):
            B_values[i] = self.barrier.evaluate(states[i].reshape(1, -1))[0]
            B_next_values[i] = self.barrier.evaluate(next_states[i].reshape(1, -1))[0]
        
        # Add constraints
        constraints = [eta >= 0]  # Non-negative slack
        
        # Sample constraints for decay: B(x^+) <= B(x) + z^T X z for all x, w
        for i in range(n_samples):
            z_i = np.concatenate((inputs[i], states[i]))
            constraints.append(B_next_values[i] - B_values[i] - cp.quad_form(z_i, X) <= eta)
        
        # Define objective
        objective = cp.Minimize(eta)
        
        # Solve problem
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        return X.value 