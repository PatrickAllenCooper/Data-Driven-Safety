from abc import ABC, abstractmethod
import numpy as np

class BarrierRepresentation(ABC):
    """
    Abstract base class for barrier function representations.
    
    This class defines the interface for different barrier function 
    representations used in the data-driven safety verification framework.
    """
    
    def __init__(self, dimension, name=None):
        """
        Initialize the barrier representation.
        
        Args:
            dimension (int): Dimension of the state space.
            name (str, optional): Name of the representation.
        """
        self.dimension = dimension
        self.name = name if name is not None else self.__class__.__name__
        self.parameters = None
    
    @abstractmethod
    def num_parameters(self):
        """
        Return the number of parameters in this representation.
        
        Returns:
            int: Number of parameters.
        """
        pass
    
    @abstractmethod
    def evaluate(self, x, params=None):
        """
        Evaluate the barrier function at given state(s).
        
        Args:
            x (numpy.ndarray): State vector(s), shape (n_dim,) or (n_samples, n_dim).
            params (numpy.ndarray, optional): Parameters of the barrier function.
                If None, use self.parameters.
                
        Returns:
            numpy.ndarray: Barrier function value(s).
        """
        pass
    
    @abstractmethod
    def gradient(self, x, params=None):
        """
        Compute the gradient of the barrier function with respect to the state.
        
        Args:
            x (numpy.ndarray): State vector(s), shape (n_dim,) or (n_samples, n_dim).
            params (numpy.ndarray, optional): Parameters of the barrier function.
                If None, use self.parameters.
                
        Returns:
            numpy.ndarray: Gradient of barrier function, shape (n_dim,) or (n_samples, n_dim).
        """
        pass
    
    @abstractmethod
    def lipschitz_constant(self, domain_bounds, params=None):
        """
        Compute or estimate the Lipschitz constant of the barrier function.
        
        Args:
            domain_bounds (numpy.ndarray): Bounds of the domain, shape (n_dim, 2).
            params (numpy.ndarray, optional): Parameters of the barrier function.
                If None, use self.parameters.
                
        Returns:
            float: Estimated Lipschitz constant.
        """
        pass
    
    def set_parameters(self, params):
        """
        Set the parameters of the barrier function.
        
        Args:
            params (numpy.ndarray): Parameters of the barrier function.
        """
        self.parameters = params
    
    def __str__(self):
        return f"{self.name} (dim={self.dimension}, params={self.num_parameters()})" 