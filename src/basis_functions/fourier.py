import numpy as np
from src.basis_functions import BarrierRepresentation

class FourierBarrier(BarrierRepresentation):
    """
    Fourier basis functions for barrier representation.
    
    This class implements Fourier basis functions for barrier representation,
    which express the barrier function as a truncated Fourier series:
    
    For 1D:
    B(x) = a_0/2 + sum_{j=1}^{n_order} [a_j * cos(j*w*x) + b_j * sin(j*w*x)]
    
    For higher dimensions, we use tensor products of 1D bases.
    """
    
    def __init__(self, dimension, n_order=3, domain_bounds=None, name="Fourier"):
        """
        Initialize the Fourier barrier representation.
        
        Args:
            dimension (int): Dimension of the state space.
            n_order (int, optional): Order of the Fourier series. Defaults to 3.
            domain_bounds (numpy.ndarray, optional): Bounds of the domain, shape (n_dim, 2).
                Required to determine the fundamental frequency.
                If None, uses [0, 2*pi]^n_dim.
            name (str, optional): Name of the representation. Defaults to "Fourier".
        """
        super().__init__(dimension, name)
        self.n_order = n_order
        
        # Set domain bounds
        if domain_bounds is None:
            self.domain_bounds = np.zeros((dimension, 2))
            self.domain_bounds[:, 0] = 0
            self.domain_bounds[:, 1] = 2 * np.pi
        else:
            self.domain_bounds = domain_bounds
            
        # Compute fundamental frequencies for each dimension
        self.frequencies = np.zeros(dimension)
        for i in range(dimension):
            # Fundamental frequency w = 2*pi / (domain_bounds[1] - domain_bounds[0])
            self.frequencies[i] = 2 * np.pi / (self.domain_bounds[i, 1] - self.domain_bounds[i, 0])
        
        # Compute number of parameters
        if dimension == 1:
            # For 1D: 1 DC term (a_0/2) + n_order cosines + n_order sines
            self._n_params = 1 + 2 * n_order
        else:
            # For higher dimensions, we use tensor products of 1D bases
            # This results in (1 + 2 * n_order)^dimension terms
            self._n_params = (1 + 2 * n_order) ** dimension
    
    def num_parameters(self):
        """
        Return the number of parameters in this representation.
        
        Returns:
            int: Number of parameters.
        """
        return self._n_params
    
    def _normalize_input(self, x):
        """
        Normalize the input to the range [0, 2*pi] for each dimension.
        
        Args:
            x (numpy.ndarray): State vector(s), shape (n_samples, n_dim).
            
        Returns:
            numpy.ndarray: Normalized state vector(s).
        """
        x_norm = np.zeros_like(x)
        for i in range(self.dimension):
            # Linear mapping from [a, b] to [0, 2*pi]
            a, b = self.domain_bounds[i, 0], self.domain_bounds[i, 1]
            x_norm[:, i] = 2 * np.pi * (x[:, i] - a) / (b - a)
        return x_norm
        
    def _compute_1d_basis(self, x):
        """
        Compute 1D Fourier basis functions for a given input.
        
        Args:
            x (numpy.ndarray): Input, shape (n_samples,).
            
        Returns:
            numpy.ndarray: Basis function values, shape (n_samples, 1+2*n_order).
        """
        n_samples = x.shape[0]
        basis = np.zeros((n_samples, 1 + 2 * self.n_order))
        
        # DC term (constant)
        basis[:, 0] = 1
        
        # Cosine and sine terms
        for j in range(1, self.n_order + 1):
            basis[:, 2*j-1] = np.cos(j * x)  # Cosine term
            basis[:, 2*j] = np.sin(j * x)    # Sine term
            
        return basis
    
    def _tensor_product(self, bases):
        """
        Compute tensor product of basis functions for each dimension.
        
        Args:
            bases (list): List of basis function values for each dimension.
                Each element has shape (n_samples, 1+2*n_order).
                
        Returns:
            numpy.ndarray: Tensor product of bases, shape (n_samples, (1+2*n_order)^dimension).
        """
        n_samples = bases[0].shape[0]
        n_bases = [b.shape[1] for b in bases]
        
        # Initialize result
        result = np.ones((n_samples, np.prod(n_bases)))
        
        # Compute tensor product
        # This is a generalization of the outer product to multiple dimensions
        index = 0
        indices = [0] * self.dimension
        
        while True:
            # Compute product of basis functions for current indices
            term = np.ones(n_samples)
            for i in range(self.dimension):
                term *= bases[i][:, indices[i]]
                
            result[:, index] = term
            index += 1
            
            # Increment indices (multi-dimensional counter)
            d = 0
            while d < self.dimension:
                indices[d] += 1
                if indices[d] < n_bases[d]:
                    break
                indices[d] = 0
                d += 1
                
            if d == self.dimension:
                break
                
        return result
        
    def evaluate(self, x, params=None):
        """
        Evaluate the Fourier barrier function at given state(s).
        
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
        
        # Normalize input to [0, 2*pi]
        x_norm = self._normalize_input(x)
        
        if self.dimension == 1:
            # For 1D, compute basis directly
            basis = self._compute_1d_basis(x_norm[:, 0])
        else:
            # For higher dimensions, compute basis for each dimension and take tensor product
            bases = []
            for i in range(self.dimension):
                bases.append(self._compute_1d_basis(x_norm[:, i]))
                
            basis = self._tensor_product(bases)
            
        # Compute dot product with parameters
        return basis @ params
    
    def gradient(self, x, params=None):
        """
        Compute the gradient of the Fourier barrier function with respect to the state.
        
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
        
        # For efficiency, compute gradients numerically
        # This is simpler than computing analytical gradients for tensor products
        h = 1e-6  # Step size for numerical differentiation
        grad = np.zeros((n_samples, self.dimension))
        
        for i in range(self.dimension):
            # Compute numerical gradient for dimension i
            x_plus = x.copy()
            x_plus[:, i] += h
            
            x_minus = x.copy()
            x_minus[:, i] -= h
            
            # Central difference approximation
            grad[:, i] = (self.evaluate(x_plus, params) - self.evaluate(x_minus, params)) / (2 * h)
            
        return grad
    
    def lipschitz_constant(self, domain_bounds, params=None):
        """
        Estimate the Lipschitz constant of the Fourier barrier function.
        
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
        
        # For Fourier series, an analytic upper bound on the Lipschitz constant
        # is based on the maximum frequency and coefficients
        if self.dimension == 1:
            # For 1D, we can compute a tighter bound
            L_bound = 0
            for j in range(1, self.n_order + 1):
                # Compute the maximum derivative of sin/cos terms
                a_j = params[2*j - 1]  # Cosine coefficient
                b_j = params[2*j]      # Sine coefficient
                L_bound += j * np.sqrt(a_j**2 + b_j**2) * self.frequencies[0]
                
            # Scale by 2*pi / (b-a) to account for input normalization
            L_bound *= 2 * np.pi / (domain_bounds[0, 1] - domain_bounds[0, 0])
        else:
            # For higher dimensions, use an empirical estimate
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
            L_bound = np.max(grad_norms)
        
        return L_bound
    
    def __str__(self):
        return f"{self.name} (dim={self.dimension}, order={self.n_order})" 