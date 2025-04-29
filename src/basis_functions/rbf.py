import numpy as np
from src.basis_functions import BarrierRepresentation
from sklearn.cluster import KMeans

class RBFBarrier(BarrierRepresentation):
    """
    Radial Basis Function (RBF) representation for barrier functions.
    
    This class implements RBF barrier functions, which are represented as:
    B(x) = sum_{j=1}^{n_centers} q_j * phi(||x - c_j||)
    
    where phi is a radial basis function, c_j are centers, and q_j are parameters.
    """
    
    def __init__(self, dimension, n_centers=10, width=1.0, centers=None, name="RBF"):
        """
        Initialize the RBF barrier representation.
        
        Args:
            dimension (int): Dimension of the state space.
            n_centers (int, optional): Number of RBF centers. Defaults to 10.
            width (float, optional): Width parameter of RBFs. Defaults to 1.0.
            centers (numpy.ndarray, optional): Centers of RBFs, shape (n_centers, n_dim).
                If None, centers will be initialized when fit_centers is called.
            name (str, optional): Name of the representation. Defaults to "RBF".
        """
        super().__init__(dimension, name)
        self.n_centers = n_centers
        self.width = width
        self.centers = centers
        
        # Number of parameters: one weight per center + offset
        self._n_params = n_centers + 1
    
    def num_parameters(self):
        """
        Return the number of parameters in this representation.
        
        Returns:
            int: Number of parameters.
        """
        return self._n_params
    
    def fit_centers(self, data):
        """
        Initialize centers using K-means clustering on data.
        
        Args:
            data (numpy.ndarray): Data points, shape (n_samples, n_dim).
            
        Returns:
            self: Returns self.
        """
        # Use K-means to find centers
        kmeans = KMeans(n_clusters=self.n_centers)
        kmeans.fit(data)
        self.centers = kmeans.cluster_centers_
        return self
    
    def rbf(self, x, c):
        """
        Evaluate the RBF at given points.
        
        Args:
            x (numpy.ndarray): Points, shape (n_samples, n_dim).
            c (numpy.ndarray): Center, shape (n_dim,).
            
        Returns:
            numpy.ndarray: RBF values, shape (n_samples,).
        """
        # Gaussian RBF: phi(r) = exp(-r^2 / (2*width^2))
        distances = np.sum((x - c) ** 2, axis=1)
        return np.exp(-distances / (2 * self.width ** 2))
    
    def evaluate(self, x, params=None):
        """
        Evaluate the RBF barrier function at given state(s).
        
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
        
        if self.centers is None:
            raise ValueError("Centers are not initialized. Call fit_centers first.")
        
        # Make x 2D if it's not already
        x = np.atleast_2d(x)
        
        # Compute RBF values for each center
        # Last parameter is the offset (bias term)
        offset = params[-1]
        rbf_values = np.zeros(x.shape[0])
        
        for i, center in enumerate(self.centers):
            rbf_values += params[i] * self.rbf(x, center)
        
        return rbf_values + offset
    
    def gradient(self, x, params=None):
        """
        Compute the gradient of the RBF barrier function with respect to the state.
        
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
        
        if self.centers is None:
            raise ValueError("Centers are not initialized. Call fit_centers first.")
        
        # Make x 2D if it's not already
        x = np.atleast_2d(x)
        n_samples = x.shape[0]
        
        # Gradient of Gaussian RBF:
        # d/dx phi(||x - c||) = -phi(||x - c||) * (x - c) / width^2
        
        grad = np.zeros((n_samples, self.dimension))
        
        for i, center in enumerate(self.centers):
            rbf_values = self.rbf(x, center)
            
            for j in range(self.dimension):
                grad[:, j] += -params[i] * rbf_values * (x[:, j] - center[j]) / (self.width ** 2)
        
        return grad
    
    def lipschitz_constant(self, domain_bounds, params=None):
        """
        Estimate the Lipschitz constant of the RBF barrier function.
        
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
        
        if self.centers is None:
            raise ValueError("Centers are not initialized. Call fit_centers first.")
        
        # For RBFs, we can derive an analytical bound for the Lipschitz constant
        # based on the maximum gradient norm
        
        # For Gaussian RBF, the gradient norm is maximized at specific points
        # relative to the centers. A conservative bound is:
        # L = sum_i |q_i| * 1/width * exp(-0.5)
        
        center_weights = params[:-1]  # Exclude the offset
        L_bound = np.sum(np.abs(center_weights)) * (1 / self.width) * np.exp(-0.5)
        
        # For a more accurate estimate, use grid search as in polynomial case
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
        L_empirical = np.max(grad_norms)
        
        # Return the maximum of the analytical and empirical estimates
        return max(L_bound, L_empirical)
    
    def __str__(self):
        return f"{self.name} (dim={self.dimension}, centers={self.n_centers}, width={self.width})" 