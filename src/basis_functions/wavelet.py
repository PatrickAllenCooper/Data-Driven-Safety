import numpy as np
import pywt
from src.basis_functions import BarrierRepresentation

class WaveletBarrier(BarrierRepresentation):
    """
    Wavelet basis functions for barrier representation.
    
    This class implements wavelet basis functions for barrier representation,
    which express the barrier function as a linear combination of wavelet functions:
    
    B(x) = sum_{j} q_j * psi_j(x)
    
    where psi_j are wavelet basis functions at different scales and translations.
    """
    
    def __init__(self, dimension, wavelet='db4', level=3, domain_bounds=None, name="Wavelet"):
        """
        Initialize the wavelet barrier representation.
        
        Args:
            dimension (int): Dimension of the state space.
            wavelet (str, optional): Wavelet family. Defaults to 'db4'.
            level (int, optional): Decomposition level. Defaults to 3.
            domain_bounds (numpy.ndarray, optional): Bounds of the domain, shape (n_dim, 2).
                If None, uses [0, 1]^n_dim.
            name (str, optional): Name of the representation. Defaults to "Wavelet".
        """
        super().__init__(dimension, name)
        self.wavelet = wavelet
        self.level = level
        
        # Set domain bounds
        if domain_bounds is None:
            self.domain_bounds = np.zeros((dimension, 2))
            self.domain_bounds[:, 0] = 0
            self.domain_bounds[:, 1] = 1
        else:
            self.domain_bounds = domain_bounds
        
        # Create wavelet object
        self.wavelet_obj = pywt.Wavelet(wavelet)
        
        # Calculate number of parameters
        # For 1D: Approximate coefficients + Detail coefficients at each level
        # Number of coefficients at level j is 2^j
        if dimension == 1:
            self._n_params = 2 ** level + sum(2 ** j for j in range(1, level + 1))
        else:
            # For multidimensional case, we use a tensor product of wavelets
            # Each dimension contributes a multiplicative factor
            n_1d_params = 2 ** level + sum(2 ** j for j in range(1, level + 1))
            self._n_params = n_1d_params ** dimension
            
        # Pre-compute wavelet basis functions on a grid
        self._precompute_bases()
    
    def _precompute_bases(self, n_points=128):
        """
        Precompute wavelet basis functions on a grid.
        
        Args:
            n_points (int, optional): Number of points in the grid. Defaults to 128.
        """
        # For each dimension, we precompute wavelet basis functions on a grid
        self.grid_points = []
        self.wavelet_bases = []
        
        for d in range(self.dimension):
            # Create grid for this dimension
            a, b = self.domain_bounds[d, 0], self.domain_bounds[d, 1]
            grid = np.linspace(a, b, n_points)
            self.grid_points.append(grid)
            
            # Compute wavelet basis functions for this dimension
            basis = self._compute_wavelet_basis_1d(grid)
            self.wavelet_bases.append(basis)
    
    def _compute_wavelet_basis_1d(self, x):
        """
        Compute 1D wavelet basis functions for a given grid.
        
        Args:
            x (numpy.ndarray): Grid points, shape (n_points,).
            
        Returns:
            numpy.ndarray: Basis function values, shape (n_points, n_params_1d).
        """
        n_points = len(x)
        a, b = self.domain_bounds[0, 0], self.domain_bounds[0, 1]
        
        # Normalize grid to [0, 1]
        x_norm = (x - a) / (b - a)
        
        # Initialize basis matrix
        n_1d_params = 2 ** self.level + sum(2 ** j for j in range(1, self.level + 1))
        basis = np.zeros((n_points, n_1d_params))
        
        # Compute father wavelet (scaling function) coefficients at the coarsest level
        # These correspond to the approximate coefficients
        start_idx = 0
        for i in range(2 ** self.level):
            # Evaluate scaling function centered at i/(2^level)
            t = 2 ** self.level * x_norm - i
            basis[:, start_idx + i] = self._evaluate_scaling_function(t)
        
        start_idx += 2 ** self.level
        
        # Compute mother wavelet (detail) coefficients at each level
        for j in range(1, self.level + 1):
            for i in range(2 ** j):
                # Evaluate wavelet function at scale j, translation i
                t = 2 ** j * x_norm - i
                basis[:, start_idx + i] = self._evaluate_wavelet_function(t)
            
            start_idx += 2 ** j
        
        return basis
    
    def _evaluate_scaling_function(self, t):
        """
        Evaluate the scaling function (father wavelet) at given points.
        
        Args:
            t (numpy.ndarray): Input points.
            
        Returns:
            numpy.ndarray: Scaling function values.
        """
        # For simplicity, we use a B-spline approximation
        phi = np.zeros_like(t, dtype=float)
        
        # Support of the scaling function
        support = len(self.wavelet_obj.dec_lo)
        
        # Evaluate B-spline
        mask = (t >= 0) & (t < support)
        if np.any(mask):
            # Simple B-spline approximation
            phi[mask] = np.maximum(0, 1 - np.abs(2 * t[mask] / support - 1))
            
        return phi
    
    def _evaluate_wavelet_function(self, t):
        """
        Evaluate the wavelet function (mother wavelet) at given points.
        
        Args:
            t (numpy.ndarray): Input points.
            
        Returns:
            numpy.ndarray: Wavelet function values.
        """
        # For simplicity, we use a wavelet approximation
        psi = np.zeros_like(t, dtype=float)
        
        # Support of the wavelet function
        support = len(self.wavelet_obj.dec_hi)
        
        # Evaluate wavelet (simplified)
        mask = (t >= 0) & (t < support)
        if np.any(mask):
            # Simple wavelet approximation (Mexican hat-like for Daubechies)
            s = t[mask] / support
            psi[mask] = np.sin(2 * np.pi * s) * np.exp(-(s - 0.5) ** 2 / 0.1)
            
        return psi
    
    def num_parameters(self):
        """
        Return the number of parameters in this representation.
        
        Returns:
            int: Number of parameters.
        """
        return self._n_params
    
    def _interpolate_basis(self, x, dim):
        """
        Interpolate precomputed basis functions at given points.
        
        Args:
            x (numpy.ndarray): Points to interpolate at, shape (n_samples,).
            dim (int): Dimension index.
            
        Returns:
            numpy.ndarray: Interpolated basis function values, shape (n_samples, n_params_1d).
        """
        # Get grid and precomputed basis for this dimension
        grid = self.grid_points[dim]
        basis = self.wavelet_bases[dim]
        
        # Interpolate each basis function
        n_samples = len(x)
        n_basis = basis.shape[1]
        result = np.zeros((n_samples, n_basis))
        
        for i in range(n_basis):
            # Use linear interpolation
            result[:, i] = np.interp(x, grid, basis[:, i])
            
        return result
    
    def _tensor_product(self, bases):
        """
        Compute tensor product of basis functions for each dimension.
        
        Args:
            bases (list): List of basis function values for each dimension.
                
        Returns:
            numpy.ndarray: Tensor product of bases.
        """
        n_samples = bases[0].shape[0]
        n_bases = [b.shape[1] for b in bases]
        
        # Initialize result
        result = np.ones((n_samples, np.prod(n_bases)))
        
        # Compute tensor product
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
        Evaluate the wavelet barrier function at given state(s).
        
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
        
        # Interpolate basis functions for each dimension
        bases = []
        for i in range(self.dimension):
            # Clip x to domain bounds for stability
            x_clipped = np.clip(x[:, i], self.domain_bounds[i, 0], self.domain_bounds[i, 1])
            basis_i = self._interpolate_basis(x_clipped, i)
            bases.append(basis_i)
        
        # For 1D, return directly
        if self.dimension == 1:
            return bases[0] @ params
        
        # For higher dimensions, compute tensor product
        basis = self._tensor_product(bases)
        return basis @ params
    
    def gradient(self, x, params=None):
        """
        Compute the gradient of the wavelet barrier function with respect to the state.
        
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
        
        # For wavelets, we compute gradients numerically
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
        Estimate the Lipschitz constant of the wavelet barrier function.
        
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
        
        # For wavelets, rely on numerical estimation
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
        return f"{self.name} (dim={self.dimension}, wavelet={self.wavelet}, level={self.level})"

    def _normalize_input(self, x):
        """
        Normalize input to the domain [0, 1]^n_dim.
        
        Args:
            x (numpy.ndarray): Input points, shape (n_samples, n_dim).
            
        Returns:
            numpy.ndarray: Normalized input, shape (n_samples, n_dim).
        """
        # Make x 2D if it's not already
        x = np.atleast_2d(x)
        
        # Normalize each dimension to [0, 1]
        x_norm = np.zeros_like(x, dtype=float)
        
        for d in range(self.dimension):
            a, b = self.domain_bounds[d, 0], self.domain_bounds[d, 1]
            # Clip values to domain bounds for stability
            x_clipped = np.clip(x[:, d], a, b)
            # Normalize to [0, 1]
            x_norm[:, d] = (x_clipped - a) / (b - a)
            
        return x_norm
        
    def _compute_barrier_values(self, state):
        """
        Compute the values of basis functions at a given state.
        
        Args:
            state (numpy.ndarray): State vector, shape (state_dim,).
            
        Returns:
            numpy.ndarray: Values of basis functions, shape (n_params,).
        """
        state = np.atleast_2d(state)
        bases = []
        for i in range(self.dimension):
            x_clipped = np.clip(state[:, i], self.domain_bounds[i, 0], self.domain_bounds[i, 1])
            basis_i = self._interpolate_basis(x_clipped, i)
            bases.append(basis_i)
        
        if self.dimension == 1:
            return bases[0][0]
        else:
            return self._tensor_product(bases)[0] 