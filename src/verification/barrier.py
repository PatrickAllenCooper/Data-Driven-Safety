import numpy as np
from src.optimization.scp import LinearSCP, NeuralSCP
from src.optimization.admm import ADMMSolver

class BarrierVerifier:
    """
    Barrier function verifier for safety verification.
    
    This class implements the core verification procedure using barrier functions.
    """
    
    def __init__(self, network, barrier_representations, confidence=0.99, epsilon=1e-3, verbose=False):
        """
        Initialize the barrier verifier.
        
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
        
        # Initialize optimizers for each subsystem
        self.optimizers = []
        for i, (subsystem, barrier) in enumerate(zip(network.subsystems, barrier_representations)):
            if hasattr(barrier, 'model'):
                # Neural network barrier
                self.optimizers.append(NeuralSCP(subsystem, barrier, epsilon))
            else:
                # Linear barrier
                self.optimizers.append(LinearSCP(subsystem, barrier, epsilon))
        
        # Initialize ADMM solver
        self.admm_solver = ADMMSolver(network, verbose=verbose)
        
        # Initialize results
        self.verification_result = None
        self.verification_confidence = None
    
    def verify(self, data_list=None, n_samples=None, from_initial=False, admm_max_iter=100, **kwargs):
        """
        Verify the safety of the network using barrier functions.
        
        Args:
            data_list (list, optional): List of data tuples (states, inputs, next_states) for each subsystem.
                If None, data is generated using n_samples.
            n_samples (int, optional): Number of samples to generate per subsystem.
                If None, computes the minimum number of samples required for the desired confidence.
            from_initial (bool, optional): Whether to sample from the initial set.
                If False, samples from the entire state space. Defaults to False.
            admm_max_iter (int, optional): Maximum number of ADMM iterations. Defaults to 100.
            **kwargs: Additional parameters for the optimizers.
            
        Returns:
            dict: Verification results.
        """
        # Compute minimum number of samples if not provided
        if n_samples is None and data_list is None:
            n_samples = self._compute_min_samples()
            if self.verbose:
                print(f"Using {n_samples} samples per subsystem for verification.")
        
        # Generate data if not provided
        if data_list is None:
            if self.verbose:
                print("Generating data...")
            data_list = self.network.generate_data(n_samples, from_initial)
        
        # Solve the SCP for each subsystem
        if self.verbose:
            print("Solving local scenario convex programs...")
        
        local_results = []
        for i, optimizer in enumerate(self.optimizers):
            if self.verbose:
                print(f"Subsystem {i+1}/{len(self.optimizers)}...")
            result = optimizer.solve(data_list[i], verbose=self.verbose, **kwargs)
            local_results.append(result)
        
        # Check if all local problems are feasible
        local_feasible = all(result['status'] in ["optimal", "optimal_inaccurate"] for result in local_results)
        
        if not local_feasible:
            if self.verbose:
                print("Some local problems are infeasible. Verification failed.")
            self.verification_result = False
            self.verification_confidence = 0.0
            
            return {
                'verified': False,
                'confidence': 0.0,
                'reason': "local_infeasible",
                'local_results': local_results,
                'admm_result': None
            }
        
        # Solve the global ADMM problem
        if self.verbose:
            print("Solving global ADMM problem...")
        
        self.admm_solver.max_iter = admm_max_iter
        admm_result = self.admm_solver.solve(self.barriers, data_list, verbose=self.verbose)
        
        # Check if the LMI condition is satisfied
        lmi_satisfied = admm_result['lmi_satisfied']
        
        if not lmi_satisfied:
            if self.verbose:
                print("LMI condition is not satisfied. Verification failed.")
            self.verification_result = False
            self.verification_confidence = 0.0
            
            return {
                'verified': False,
                'confidence': 0.0,
                'reason': "lmi_not_satisfied",
                'local_results': local_results,
                'admm_result': admm_result
            }
        
        # Compute confidence level
        confidence = self._compute_confidence(data_list)
        
        # Set results
        self.verification_result = True
        self.verification_confidence = confidence
        
        if self.verbose:
            print(f"Verification successful with confidence {confidence:.4f}.")
        
        return {
            'verified': True,
            'confidence': confidence,
            'reason': "success",
            'local_results': local_results,
            'admm_result': admm_result
        }
    
    def evaluate_barrier(self, state):
        """
        Evaluate the overall barrier function at a given state.
        
        Args:
            state (numpy.ndarray): State to evaluate, shape (total_state_dim,).
            
        Returns:
            float: Barrier function value.
        """
        # Split state into subsystem states
        subsystem_states = self.network._split_state(state)
        
        # Evaluate barrier function for each subsystem
        barrier_values = []
        for i, (subsystem, barrier) in enumerate(zip(self.network.subsystems, self.barriers)):
            value = barrier.evaluate(subsystem_states[i].reshape(1, -1))[0]
            barrier_values.append(value)
        
        # Sum the values to get the overall barrier function
        return sum(barrier_values)
    
    def is_state_safe(self, state):
        """
        Check if a state is certified safe by the barrier function.
        
        Args:
            state (numpy.ndarray): State to check, shape (total_state_dim,).
            
        Returns:
            bool: True if the state is certified safe, False otherwise.
        """
        if not self.verification_result:
            return False
            
        # Evaluate the barrier function
        barrier_value = self.evaluate_barrier(state)
        
        # Check if the barrier value is less than 1
        return barrier_value < 1.0
    
    def _compute_min_samples(self):
        """
        Compute the minimum number of samples required for the desired confidence level.
        
        Returns:
            int: Minimum number of samples.
        """
        # Compute for each subsystem
        n_samples_list = []
        
        for optimizer in self.optimizers:
            n = optimizer.compute_minimum_samples(self.confidence, self.epsilon)
            n_samples_list.append(n)
        
        # Return the maximum
        return max(n_samples_list)
    
    def _compute_confidence(self, data_list):
        """
        Compute the confidence level based on the number of samples used.
        
        Args:
            data_list (list): List of data tuples (states, inputs, next_states) for each subsystem.
            
        Returns:
            float: Confidence level.
        """
        # Compute confidence for each subsystem
        betas = []
        
        for i, optimizer in enumerate(self.optimizers):
            n_samples = data_list[i][0].shape[0]
            n_vars = optimizer.barrier.num_parameters() + 1 + (optimizer.subsystem.input_dim + optimizer.subsystem.state_dim) ** 2
            
            # Compute beta using the formula from Theorem 3
            eps_normalized = (self.epsilon / 10.0) ** n_vars  # Using a placeholder Lipschitz constant
            beta = 0
            
            for j in range(n_vars + 1):
                beta += optimizer._binom(n_samples, j) * (eps_normalized ** j) * ((1 - eps_normalized) ** (n_samples - j))
                
            betas.append(beta)
        
        # Overall confidence is 1 - sum(betas)
        confidence = 1.0 - sum(betas)
        
        # Clip to [0, 1]
        return max(0.0, min(1.0, confidence)) 