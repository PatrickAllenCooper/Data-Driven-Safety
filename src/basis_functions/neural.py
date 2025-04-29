import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.basis_functions import BarrierRepresentation

class NeuralBarrier(BarrierRepresentation):
    """
    Neural network representation for barrier functions.
    
    This class implements barrier functions represented as feedforward neural networks.
    While not directly a linear combination of basis functions, this can be viewed as
    an adaptive basis function approach where the bases are learned from data.
    
    Note: Since NNs are nonlinear in their parameters, they require a modified SCP formulation,
    which is handled in the optimization module.
    """
    
    def __init__(self, dimension, hidden_layers=(32, 32), activation='relu', name="Neural"):
        """
        Initialize the neural network barrier representation.
        
        Args:
            dimension (int): Dimension of the state space.
            hidden_layers (tuple, optional): Sizes of hidden layers. Defaults to (32, 32).
            activation (str, optional): Activation function. Options: 'relu', 'tanh', 'sigmoid'.
                Defaults to 'relu'.
            name (str, optional): Name of the representation. Defaults to "Neural".
        """
        super().__init__(dimension, name)
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        
        # Create neural network model
        self.model = self._create_model()
        
        # Count number of parameters
        self._n_params = sum(p.numel() for p in self.model.parameters())
        
        # Initialize Lipschitz constant estimate
        self._lipschitz_estimate = None
    
    def _create_model(self):
        """
        Create the neural network model.
        
        Returns:
            torch.nn.Module: Neural network model.
        """
        # Define activation function
        if self.activation_name == 'relu':
            activation = nn.ReLU()
        elif self.activation_name == 'tanh':
            activation = nn.Tanh()
        elif self.activation_name == 'sigmoid':
            activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")
        
        # Create layers
        layers = []
        
        # Input layer
        prev_size = self.dimension
        
        # Hidden layers
        for size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation)
            prev_size = size
        
        # Output layer (scalar output)
        layers.append(nn.Linear(prev_size, 1))
        
        # Create sequential model
        return nn.Sequential(*layers)
    
    def num_parameters(self):
        """
        Return the number of parameters in this representation.
        
        Returns:
            int: Number of parameters.
        """
        return self._n_params
    
    def set_parameters(self, params):
        """
        Set the parameters of the neural network.
        
        Args:
            params (numpy.ndarray or torch.Tensor): Flattened parameters of the network.
        """
        # Convert to torch tensor if numpy array
        if isinstance(params, np.ndarray):
            params = torch.from_numpy(params).float()
        
        # Store parameters
        self.parameters = params.detach().numpy() if isinstance(params, torch.Tensor) else params
        
        # Set parameters in model
        self._set_model_params(params)
        
        # Reset Lipschitz estimate
        self._lipschitz_estimate = None
    
    def _set_model_params(self, params):
        """
        Set the parameters of the model from a flattened parameter vector.
        
        Args:
            params (torch.Tensor): Flattened parameters of the network.
        """
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=torch.float32)
            
        # Extract individual parameter tensors
        idx = 0
        for name, param in self.model.named_parameters():
            # Get the size of the current parameter
            num_params = param.numel()
            
            # Extract the relevant portion of the flattened parameters
            param_data = params[idx:idx+num_params].reshape(param.shape)
            
            # Set the parameter value
            with torch.no_grad():
                param.copy_(param_data)
                
            # Update index
            idx += num_params
    
    def _get_model_params(self):
        """
        Get the parameters of the model as a flattened tensor.
        
        Returns:
            torch.Tensor: Flattened parameters of the network.
        """
        return torch.cat([p.flatten() for p in self.model.parameters()])
    
    def evaluate(self, x, params=None):
        """
        Evaluate the neural network barrier function at given state(s).
        
        Args:
            x (numpy.ndarray): State vector(s), shape (n_dim,) or (n_samples, n_dim).
            params (numpy.ndarray, optional): Parameters of the barrier function.
                If None, use self.parameters.
                
        Returns:
            numpy.ndarray: Barrier function value(s).
        """
        if params is not None:
            # Temporarily set parameters for evaluation
            original_params = self._get_model_params().detach().numpy()
            self._set_model_params(params)
        
        # Convert input to torch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Make x 2D if it's not already
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Evaluate model
        with torch.no_grad():
            result = self.model(x).squeeze().numpy()
        
        # Restore original parameters if needed
        if params is not None:
            self._set_model_params(original_params)
        
        return result
    
    def gradient(self, x, params=None):
        """
        Compute the gradient of the neural network barrier function with respect to the state.
        
        Args:
            x (numpy.ndarray): State vector(s), shape (n_dim,) or (n_samples, n_dim).
            params (numpy.ndarray, optional): Parameters of the barrier function.
                If None, use self.parameters.
                
        Returns:
            numpy.ndarray: Gradient of barrier function, shape (n_samples, n_dim).
        """
        if params is not None:
            # Temporarily set parameters for evaluation
            original_params = self._get_model_params().detach().numpy()
            self._set_model_params(params)
        
        # Convert input to torch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Make x 2D if it's not already
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Ensure requires_grad
        x = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(x)
        
        # Compute gradients
        grads = []
        for i in range(len(output)):
            # Zero gradients
            if x.grad is not None:
                x.grad.zero_()
            
            # Backward pass
            output[i].backward(retain_graph=(i < len(output) - 1))
            
            # Store gradient
            grads.append(x.grad[i].clone().detach().numpy())
        
        # Restore original parameters if needed
        if params is not None:
            self._set_model_params(original_params)
        
        return np.array(grads)
    
    def lipschitz_constant(self, domain_bounds, params=None):
        """
        Estimate the Lipschitz constant of the neural network barrier function.
        
        For neural networks, this is more complex. We use a combination of:
        1. Product of operator norms of weight matrices for fully connected layers
        2. Empirical estimation on a grid of points
        
        Args:
            domain_bounds (numpy.ndarray): Bounds of the domain, shape (n_dim, 2).
            params (numpy.ndarray, optional): Parameters of the barrier function.
                If None, use self.parameters.
                
        Returns:
            float: Estimated Lipschitz constant.
        """
        if params is not None:
            # Temporarily set parameters for estimation
            original_params = self._get_model_params().detach().numpy()
            self._set_model_params(params)
        
        # If we already computed the Lipschitz estimate, return it
        if self._lipschitz_estimate is not None and params is None:
            return self._lipschitz_estimate
        
        # For ReLU networks, the Lipschitz constant can be upper-bounded by
        # the product of the spectral norms of the weight matrices
        L_bound = 1.0
        
        for module in self.model:
            if isinstance(module, nn.Linear):
                # Compute spectral norm (maximum singular value)
                # This is an upper bound on the Lipschitz constant of the linear layer
                weight = module.weight.detach()
                spectral_norm = torch.linalg.norm(weight, ord=2)
                L_bound *= spectral_norm.item()
        
        # Also perform empirical estimation on a grid
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
        grads = self.gradient(points)
        
        # Compute the norm of the gradient at each point
        grad_norms = np.linalg.norm(grads, axis=1)
        
        # Empirical Lipschitz constant is the maximum norm of the gradient
        L_empirical = np.max(grad_norms)
        
        # Take the minimum of the two estimates
        # The analytical bound is often too conservative
        L = min(L_bound, L_empirical * 1.5)  # Add 50% margin to empirical estimate
        
        # Store the result for future calls
        self._lipschitz_estimate = L
        
        # Restore original parameters if needed
        if params is not None:
            self._set_model_params(original_params)
        
        return L
    
    def train(self, x, y, lr=0.01, epochs=1000, batch_size=32, verbose=False):
        """
        Train the neural network on data.
        
        This is an additional utility method specific to neural networks,
        which can be used to pre-train the network before optimization.
        
        Args:
            x (numpy.ndarray): Input data, shape (n_samples, n_dim).
            y (numpy.ndarray): Target values, shape (n_samples,).
            lr (float, optional): Learning rate. Defaults to 0.01.
            epochs (int, optional): Number of epochs. Defaults to 1000.
            batch_size (int, optional): Batch size. Defaults to 32.
            verbose (bool, optional): Whether to print progress. Defaults to False.
            
        Returns:
            self: Returns self.
        """
        # Convert data to torch tensors
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_x, batch_y in dataloader:
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x).squeeze()
                
                # Compute loss
                loss = loss_fn(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Accumulate loss
                total_loss += loss.item()
            
            # Print progress
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
        
        # Update parameters
        self.parameters = self._get_model_params().detach().numpy()
        
        # Reset Lipschitz estimate
        self._lipschitz_estimate = None
        
        return self
    
    def __str__(self):
        return f"{self.name} (dim={self.dimension}, layers={self.hidden_layers}, activation={self.activation_name})" 