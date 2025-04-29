import numpy as np
from src.basis_functions import BarrierRepresentation

class PolynomialBarrier(BarrierRepresentation):
    """
    Polynomial basis functions for barrier representation.
    
    This class implements polynomial basis functions for barrier representation, 
    which is the baseline approach used in the original paper.
    
    The barrier function is expressed as a linear combination of monomials:
    B(x) = sum_{j=1}^{degree+1} q_j * x^{j-1}
    """
    
    def __init__(self, dimension, degree=2, name="Polynomial"):
        """
        Initialize the polynomial barrier representation.
        
        Args:
            dimension (int): Dimension of the state space.
            degree (int, optional): Degree of the polynomial. Defaults to 2.
            name (str, optional): Name of the representation. Defaults to "Polynomial".
        """
        super().__init__(dimension, name)
        self.degree = degree
        
        # For multivariate systems, we use a different structure
        if dimension > 1:
            # We'll use all monomials up to the given degree
            # For example, with dim=2, degree=2, we'll have:
            # 1, x1, x2, x1^2, x1*x2, x2^2
            self._n_params = self._compute_n_params()
        else:
            # For univariate systems, we use monomials up to the given degree
            # For example, with degree=2, we'll have: 1, x, x^2
            self._n_params = degree + 1
            
    def _compute_n_params(self):
        """
        Compute the number of parameters for multivariate polynomial.
        
        Returns:
            int: Number of parameters.
        """
        # Number of monomials up to degree d in n variables is (n+d)! / (n! * d!)
        n, d = self.dimension, self.degree
        
        # Compute binomial coefficient (n+d)! / (n! * d!)
        result = 1
        for i in range(1, d+1):
            result = result * (n+i) / i
        
        return int(result)

    def num_parameters(self):
        """
        Return the number of parameters in this representation.
        
        Returns:
            int: Number of parameters.
        """
        return self._n_params
    
    def evaluate(self, x, params=None):
        """
        Evaluate the polynomial barrier function at given state(s).
        
        Args:
            x (numpy.ndarray): State vector(s), shape (n_dim,) or (n_samples, n_dim).
            params (numpy.ndarray, optional): Parameters of the barrier function.
                If None, use self.parameters.
                
        Returns:
            numpy.ndarray: Barrier function value(s).
        """
        params = self.parameters if params is None else params
        
        if params is None:
            raise ValueError("Parameters are not set and not provided.")
        
        # Make x 2D if it's not already
        x = np.atleast_2d(x)
        
        if self.dimension == 1:
            # For univariate case, compute powers directly
            # [1, x, x^2, ..., x^degree]
            x_powers = np.vstack([x.flatten() ** i for i in range(self.degree + 1)]).T
            return x_powers @ params
        else:
            # For multivariate case, compute all monomials up to the given degree
            # This is more complex and we'll use a helper function
            basis_values = self._compute_multivariate_basis(x)
            return basis_values @ params
    
    def _compute_multivariate_basis(self, x):
        """
        Compute all multivariate monomials up to the given degree.
        
        Args:
            x (numpy.ndarray): State vectors, shape (n_samples, n_dim).
            
        Returns:
            numpy.ndarray: Basis function values, shape (n_samples, n_params).
        """
        n_samples = x.shape[0]
        basis_values = np.ones((n_samples, self._n_params))
        
        # First basis function is just 1
        idx = 1
        
        # Add monomials of increasing degree
        for d in range(1, self.degree + 1):
            # Generate all combinations of indices that sum to d
            for combo in self._generate_index_combinations(d):
                # Compute the monomial x_1^{combo[0]} * x_2^{combo[1]} * ...
                monomial = np.ones(n_samples)
                for i, power in enumerate(combo):
                    if power > 0:
                        monomial *= x[:, i] ** power
                
                basis_values[:, idx] = monomial
                idx += 1
                
        return basis_values
    
    def _generate_index_combinations(self, degree):
        """
        Generate all combinations of indices that sum to the given degree.
        
        Args:
            degree (int): The degree of the monomials.
            
        Returns:
            list: List of tuples, where each tuple is a combination of indices.
        """
        def generate_recursive(n, d, current=[]):
            if n == 1:
                yield current + [d]
            else:
                for i in range(d + 1):
                    yield from generate_recursive(n - 1, d - i, current + [i])
        
        return list(generate_recursive(self.dimension, degree, []))
    
    def gradient(self, x, params=None):
        """
        Compute the gradient of the polynomial barrier function with respect to the state.
        
        Args:
            x (numpy.ndarray): State vector(s), shape (n_dim,) or (n_samples, n_dim).
            params (numpy.ndarray, optional): Parameters of the barrier function.
                If None, use self.parameters.
                
        Returns:
            numpy.ndarray: Gradient of barrier function, shape (n_samples, n_dim).
        """
        params = self.parameters if params is None else params
        
        if params is None:
            raise ValueError("Parameters are not set and not provided.")
        
        # Make x 2D if it's not already
        x = np.atleast_2d(x)
        n_samples = x.shape[0]
        
        if self.dimension == 1:
            # For univariate case, compute derivatives directly
            # [0, 1, 2*x, 3*x^2, ..., degree*x^(degree-1)]
            grad = np.zeros((n_samples, 1))
            for i in range(1, self.degree + 1):
                grad[:, 0] += params[i] * i * (x.flatten() ** (i - 1))
            return grad
        else:
            # For multivariate case, compute gradients for all monomials
            grad = np.zeros((n_samples, self.dimension))
            
            # Constant term has zero gradient
            idx = 1
            
            # Add gradients of monomials of increasing degree
            for d in range(1, self.degree + 1):
                for combo in self._generate_index_combinations(d):
                    # Compute gradient for each dimension
                    for i in range(self.dimension):
                        if combo[i] > 0:
                            # Compute d/dx_i of x_1^{combo[0]} * x_2^{combo[1]} * ...
                            g = params[idx] * combo[i]
                            for j in range(self.dimension):
                                if i == j:
                                    if combo[j] > 1:
                                        g *= x[:, j] ** (combo[j] - 1)
                                else:
                                    if combo[j] > 0:
                                        g *= x[:, j] ** combo[j]
                            
                            grad[:, i] += g
                    
                    idx += 1
            
            return grad
    
    def lipschitz_constant(self, domain_bounds, params=None):
        """
        Estimate the Lipschitz constant of the polynomial barrier function.
        
        This is a conservative estimate based on the maximum absolute value of
        the gradient over the domain.
        
        Args:
            domain_bounds (numpy.ndarray): Bounds of the domain, shape (n_dim, 2).
            params (numpy.ndarray, optional): Parameters of the barrier function.
                If None, use self.parameters.
                
        Returns:
            float: Estimated Lipschitz constant.
        """
        params = self.parameters if params is None else params
        
        if params is None:
            raise ValueError("Parameters are not set and not provided.")
        
        # For a simple estimate, we use a grid search over the domain
        # This is not the most efficient method but works for low dimensions
        
        # Create a grid over the domain
        n_points = 10  # Number of points per dimension
        grid_points = []
        
        for d in range(self.dimension):
            grid_points.append(np.linspace(
                domain_bounds[d, 0], domain_bounds[d, 1], n_points
            ))
        
        # Create meshgrid for all dimensions
        mesh = np.meshgrid(*grid_points)
        
        # Reshape to get all combinations
        points = np.vstack([m.flatten() for m in mesh]).T
        
        # Compute gradients at all grid points
        grads = self.gradient(points, params)
        
        # Compute the norm of the gradient at each point
        grad_norms = np.linalg.norm(grads, axis=1)
        
        # The Lipschitz constant is the maximum norm of the gradient
        return np.max(grad_norms)
    
    def __str__(self):
        return f"{self.name} (dim={self.dimension}, degree={self.degree})" 