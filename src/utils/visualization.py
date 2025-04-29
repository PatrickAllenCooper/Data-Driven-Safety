import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_barrier_function_1d(barrier, domain, subsystem=None, resolution=100, ax=None, show=True):
    """
    Plot a 1D barrier function.
    
    Args:
        barrier: Barrier function representation.
        domain (numpy.ndarray): Domain bounds [min, max].
        subsystem: Subsystem object (optional).
        resolution (int, optional): Resolution of the plot. Defaults to 100.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure.
        show (bool, optional): Whether to show the plot. Defaults to True.
        
    Returns:
        matplotlib.axes.Axes: Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grid
    x = np.linspace(domain[0], domain[1], resolution)
    
    # Evaluate barrier function
    B = np.zeros(resolution)
    for i in range(resolution):
        B[i] = barrier.evaluate(np.array([[x[i]]]))[0]
    
    # Plot barrier function
    ax.plot(x, B, 'b-', linewidth=2, label='Barrier Function')
    
    # Add threshold line
    ax.axhline(y=1.0, color='r', linestyle='--', label='Threshold (B=1)')
    
    # If subsystem is provided, add initial and unsafe sets
    if subsystem is not None:
        if subsystem.initial_set is not None:
            x_0_min, x_0_max = subsystem.initial_set[0]
            ax.axvspan(x_0_min, x_0_max, alpha=0.2, color='g', label='Initial Set')
            
        if subsystem.unsafe_set is not None:
            x_u_min, x_u_max = subsystem.unsafe_set[0]
            ax.axvspan(x_u_min, x_u_max, alpha=0.2, color='r', label='Unsafe Set')
    
    # Add labels and legend
    ax.set_xlabel('x')
    ax.set_ylabel('B(x)')
    ax.set_title('Barrier Function')
    ax.legend()
    ax.grid(True)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax

def plot_barrier_function_2d(barrier, domain_x, domain_y, subsystem=None, resolution=50, ax=None, colorbar=True, show=True):
    """
    Plot a 2D barrier function.
    
    Args:
        barrier: Barrier function representation.
        domain_x (numpy.ndarray): Domain bounds for x [min, max].
        domain_y (numpy.ndarray): Domain bounds for y [min, max].
        subsystem: Subsystem object (optional).
        resolution (int, optional): Resolution of the plot. Defaults to 50.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure.
        colorbar (bool, optional): Whether to add a colorbar. Defaults to True.
        show (bool, optional): Whether to show the plot. Defaults to True.
        
    Returns:
        matplotlib.axes.Axes: Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid
    x = np.linspace(domain_x[0], domain_x[1], resolution)
    y = np.linspace(domain_y[0], domain_y[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate barrier function
    Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = barrier.evaluate(np.array([[X[i, j], Y[i, j]]]))[0]
    
    # Plot barrier function
    contour = ax.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.7)
    
    # Add contour line for B(x) = 1
    ax.contour(X, Y, Z, levels=[1.0], colors='r', linewidths=2, linestyles='dashed')
    
    # Add colorbar if requested
    if colorbar:
        plt.colorbar(contour, ax=ax, label='B(x)')
    
    # If subsystem is provided, add initial and unsafe sets
    if subsystem is not None:
        if subsystem.initial_set is not None:
            x_0_min, x_0_max = subsystem.initial_set[0]
            y_0_min, y_0_max = subsystem.initial_set[1]
            
            # Create rectangle patch for initial set
            rect_x = [x_0_min, x_0_max, x_0_max, x_0_min, x_0_min]
            rect_y = [y_0_min, y_0_min, y_0_max, y_0_max, y_0_min]
            ax.plot(rect_x, rect_y, 'g-', linewidth=2, label='Initial Set')
            
        if subsystem.unsafe_set is not None:
            x_u_min, x_u_max = subsystem.unsafe_set[0]
            y_u_min, y_u_max = subsystem.unsafe_set[1]
            
            # Create rectangle patch for unsafe set
            rect_x = [x_u_min, x_u_max, x_u_max, x_u_min, x_u_min]
            rect_y = [y_u_min, y_u_min, y_u_max, y_u_max, y_u_min]
            ax.plot(rect_x, rect_y, 'r-', linewidth=2, label='Unsafe Set')
    
    # Add labels and legend
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Barrier Function')
    ax.legend()
    ax.grid(True)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax

def plot_barrier_function_3d(barrier, domain_x, domain_y, resolution=30, ax=None, show=True):
    """
    Plot a 2D barrier function in 3D.
    
    Args:
        barrier: Barrier function representation.
        domain_x (numpy.ndarray): Domain bounds for x [min, max].
        domain_y (numpy.ndarray): Domain bounds for y [min, max].
        resolution (int, optional): Resolution of the plot. Defaults to 30.
        ax (matplotlib.axes.Axes3D, optional): Axes to plot on. If None, creates a new figure.
        show (bool, optional): Whether to show the plot. Defaults to True.
        
    Returns:
        matplotlib.axes.Axes3D: Axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
    
    # Create grid
    x = np.linspace(domain_x[0], domain_x[1], resolution)
    y = np.linspace(domain_y[0], domain_y[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate barrier function
    Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = barrier.evaluate(np.array([[X[i, j], Y[i, j]]]))[0]
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, linewidth=0)
    
    # Add horizontal plane at z=1
    xx, yy = np.meshgrid([domain_x[0], domain_x[1]], [domain_y[0], domain_y[1]])
    zz = np.ones_like(xx)
    ax.plot_surface(xx, yy, zz, color='r', alpha=0.3)
    
    # Add colorbar
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='B(x)')
    
    # Add labels
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('B(x)')
    ax.set_title('Barrier Function (3D)')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax

def plot_state_trajectory(trajectory, subsystem=None, ax=None, show=True):
    """
    Plot a state trajectory.
    
    Args:
        trajectory (numpy.ndarray): State trajectory, shape (n_steps, state_dim).
        subsystem: Subsystem object (optional).
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure.
        show (bool, optional): Whether to show the plot. Defaults to True.
        
    Returns:
        matplotlib.axes.Axes: Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    state_dim = trajectory.shape[1]
    n_steps = trajectory.shape[0]
    
    # For 1D states
    if state_dim == 1:
        ax.plot(np.arange(n_steps), trajectory, 'b-', linewidth=2)
        
        # If subsystem is provided, add initial and unsafe sets
        if subsystem is not None:
            if subsystem.initial_set is not None:
                x_0_min, x_0_max = subsystem.initial_set[0]
                ax.axhspan(x_0_min, x_0_max, alpha=0.2, color='g', label='Initial Set')
                
            if subsystem.unsafe_set is not None:
                x_u_min, x_u_max = subsystem.unsafe_set[0]
                ax.axhspan(x_u_min, x_u_max, alpha=0.2, color='r', label='Unsafe Set')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('x')
    
    # For 2D states
    elif state_dim == 2:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Initial Point')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='Final Point')
        
        # If subsystem is provided, add initial and unsafe sets
        if subsystem is not None:
            if subsystem.initial_set is not None:
                x_0_min, x_0_max = subsystem.initial_set[0]
                y_0_min, y_0_max = subsystem.initial_set[1]
                
                # Create rectangle patch for initial set
                rect_x = [x_0_min, x_0_max, x_0_max, x_0_min, x_0_min]
                rect_y = [y_0_min, y_0_min, y_0_max, y_0_max, y_0_min]
                ax.plot(rect_x, rect_y, 'g-', linewidth=2, label='Initial Set')
                
            if subsystem.unsafe_set is not None:
                x_u_min, x_u_max = subsystem.unsafe_set[0]
                y_u_min, y_u_max = subsystem.unsafe_set[1]
                
                # Create rectangle patch for unsafe set
                rect_x = [x_u_min, x_u_max, x_u_max, x_u_min, x_u_min]
                rect_y = [y_u_min, y_u_min, y_u_max, y_u_max, y_u_min]
                ax.plot(rect_x, rect_y, 'r-', linewidth=2, label='Unsafe Set')
        
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
    
    # For higher dimensional states
    else:
        # Plot each component as a separate line
        for i in range(min(state_dim, 5)):  # Plot at most 5 components
            ax.plot(np.arange(n_steps), trajectory[:, i], linewidth=2, label=f'x_{i+1}')
            
        ax.set_xlabel('Time Step')
        ax.set_ylabel('State Components')
    
    ax.set_title('State Trajectory')
    ax.legend()
    ax.grid(True)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax

def plot_barrier_values(barrier_values, ax=None, show=True):
    """
    Plot barrier function values over time.
    
    Args:
        barrier_values (numpy.ndarray): Barrier function values, shape (n_steps,).
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure.
        show (bool, optional): Whether to show the plot. Defaults to True.
        
    Returns:
        matplotlib.axes.Axes: Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    n_steps = len(barrier_values)
    
    # Plot barrier values
    ax.plot(np.arange(n_steps), barrier_values, 'b-', linewidth=2)
    
    # Add threshold line
    ax.axhline(y=1.0, color='r', linestyle='--', label='Threshold (B=1)')
    
    # Add labels and legend
    ax.set_xlabel('Time Step')
    ax.set_ylabel('B(x)')
    ax.set_title('Barrier Function Values')
    ax.legend()
    ax.grid(True)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax

def plot_comparison(barrier_list, domain_x, domain_y=None, labels=None, resolution=50, show=True):
    """
    Plot a comparison of multiple barrier functions.
    
    Args:
        barrier_list (list): List of barrier function representations.
        domain_x (numpy.ndarray): Domain bounds for x [min, max].
        domain_y (numpy.ndarray, optional): Domain bounds for y [min, max]. If None, 1D plot.
        labels (list, optional): List of labels for each barrier function.
        resolution (int, optional): Resolution of the plot. Defaults to 50.
        show (bool, optional): Whether to show the plot. Defaults to True.
        
    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    n_barriers = len(barrier_list)
    
    if labels is None:
        labels = [f"Barrier {i+1}" for i in range(n_barriers)]
    
    # 1D comparison
    if domain_y is None:
        fig, axes = plt.subplots(1, n_barriers, figsize=(5*n_barriers, 6), sharex=True, sharey=True)
        
        if n_barriers == 1:
            axes = [axes]
        
        for i, (barrier, label) in enumerate(zip(barrier_list, labels)):
            plot_barrier_function_1d(barrier, domain_x, resolution=resolution, ax=axes[i], show=False)
            axes[i].set_title(label)
            
        fig.suptitle('Barrier Function Comparison', fontsize=16)
        
    # 2D comparison
    else:
        fig, axes = plt.subplots(1, n_barriers, figsize=(5*n_barriers, 6), sharex=True, sharey=True)
        
        if n_barriers == 1:
            axes = [axes]
        
        for i, (barrier, label) in enumerate(zip(barrier_list, labels)):
            plot_barrier_function_2d(barrier, domain_x, domain_y, resolution=resolution, ax=axes[i], colorbar=(i==n_barriers-1), show=False)
            axes[i].set_title(label)
            
        fig.suptitle('Barrier Function Comparison', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if show:
        plt.show()
    
    return fig

def plot_verification_results(verifier, show=True):
    """
    Plot verification results.
    
    Args:
        verifier: SafetyVerifier object.
        show (bool, optional): Whether to show the plot. Defaults to True.
        
    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    if verifier.verification_result is None:
        raise ValueError("No verification results available. Run verify() first.")
    
    # Create figure
    if verifier.network.total_state_dim <= 2:
        fig = plt.figure(figsize=(12, 10))
        
        # For 1D or 2D, plot barrier function
        if verifier.network.total_state_dim == 1:
            # Plot barrier function
            ax1 = fig.add_subplot(2, 1, 1)
            domain = [verifier.network.subsystems[0].state_bounds[0, 0], verifier.network.subsystems[0].state_bounds[0, 1]]
            plot_barrier_function_1d(verifier.barrier_verifier.barriers[0], domain, verifier.network.subsystems[0], ax=ax1, show=False)
            
            # Plot simulations
            ax2 = fig.add_subplot(2, 1, 2)
            
            # Run a few simulations
            n_simulations = 5
            n_steps = 50
            
            # Sample initial states
            initial_states = np.zeros((n_simulations, verifier.network.total_state_dim))
            for i in range(n_simulations):
                initial_states[i] = np.random.uniform(
                    verifier.network.subsystems[0].initial_set[0, 0],
                    verifier.network.subsystems[0].initial_set[0, 1]
                )
            
            # Simulate
            for i in range(n_simulations):
                trajectory = verifier.network.simulate(initial_states[i], n_steps)
                ax2.plot(np.arange(n_steps + 1), trajectory, alpha=0.7)
            
            # Add initial and unsafe sets
            initial_set = verifier.network.subsystems[0].initial_set[0]
            unsafe_set = verifier.network.subsystems[0].unsafe_set[0]
            
            ax2.axhspan(initial_set[0], initial_set[1], alpha=0.2, color='g', label='Initial Set')
            ax2.axhspan(unsafe_set[0], unsafe_set[1], alpha=0.2, color='r', label='Unsafe Set')
            
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('x')
            ax2.set_title('Simulations')
            ax2.legend()
            ax2.grid(True)
            
        else:  # 2D case
            # Plot barrier function
            ax1 = fig.add_subplot(2, 2, 1)
            domain_x = [verifier.network._split_state(np.zeros(verifier.network.total_state_dim))[0][0], verifier.network._split_state(np.zeros(verifier.network.total_state_dim))[0][1]]
            domain_y = [verifier.network._split_state(np.zeros(verifier.network.total_state_dim))[1][0], verifier.network._split_state(np.zeros(verifier.network.total_state_dim))[1][1]]
            
            # For single 2D subsystem
            if len(verifier.network.subsystems) == 1:
                domain_x = [verifier.network.subsystems[0].state_bounds[0, 0], verifier.network.subsystems[0].state_bounds[0, 1]]
                domain_y = [verifier.network.subsystems[0].state_bounds[1, 0], verifier.network.subsystems[0].state_bounds[1, 1]]
                
                plot_barrier_function_2d(verifier.barrier_verifier.barriers[0], domain_x, domain_y, verifier.network.subsystems[0], ax=ax1, show=False)
            else:
                # For two 1D subsystems, need to evaluate combined barrier
                domain_x = [verifier.network.subsystems[0].state_bounds[0, 0], verifier.network.subsystems[0].state_bounds[0, 1]]
                domain_y = [verifier.network.subsystems[1].state_bounds[0, 0], verifier.network.subsystems[1].state_bounds[0, 1]]
                
                # Create grid
                resolution = 50
                x = np.linspace(domain_x[0], domain_x[1], resolution)
                y = np.linspace(domain_y[0], domain_y[1], resolution)
                X, Y = np.meshgrid(x, y)
                
                # Evaluate barrier function
                Z = np.zeros((resolution, resolution))
                for i in range(resolution):
                    for j in range(resolution):
                        state = np.array([X[i, j], Y[i, j]])
                        Z[i, j] = verifier.barrier_verifier.evaluate_barrier(state)
                
                # Plot barrier function
                contour = ax1.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.7)
                plt.colorbar(contour, ax=ax1, label='B(x)')
                
                # Plot B(x) = 1 level set
                ax1.contour(X, Y, Z, levels=[1.0], colors='r', linewidths=2, linestyles='dashed')
                
                # Add initial and unsafe sets
                initial_x = [verifier.network.subsystems[0].initial_set[0, 0], verifier.network.subsystems[0].initial_set[0, 1], 
                             verifier.network.subsystems[0].initial_set[0, 1], verifier.network.subsystems[0].initial_set[0, 0], 
                             verifier.network.subsystems[0].initial_set[0, 0]]
                initial_y = [verifier.network.subsystems[1].initial_set[0, 0], verifier.network.subsystems[1].initial_set[0, 0], 
                             verifier.network.subsystems[1].initial_set[0, 1], verifier.network.subsystems[1].initial_set[0, 1], 
                             verifier.network.subsystems[1].initial_set[0, 0]]
                
                unsafe_x = [verifier.network.subsystems[0].unsafe_set[0, 0], verifier.network.subsystems[0].unsafe_set[0, 1], 
                            verifier.network.subsystems[0].unsafe_set[0, 1], verifier.network.subsystems[0].unsafe_set[0, 0], 
                            verifier.network.subsystems[0].unsafe_set[0, 0]]
                unsafe_y = [verifier.network.subsystems[1].unsafe_set[0, 0], verifier.network.subsystems[1].unsafe_set[0, 0], 
                            verifier.network.subsystems[1].unsafe_set[0, 1], verifier.network.subsystems[1].unsafe_set[0, 1], 
                            verifier.network.subsystems[1].unsafe_set[0, 0]]
                
                ax1.plot(initial_x, initial_y, 'g-', linewidth=2, label='Initial Set')
                ax1.plot(unsafe_x, unsafe_y, 'r-', linewidth=2, label='Unsafe Set')
                
                ax1.set_xlabel('x₁')
                ax1.set_ylabel('x₂')
                ax1.set_title('Barrier Function')
                ax1.legend()
                ax1.grid(True)
            
            # Plot simulations (state trajectories)
            ax2 = fig.add_subplot(2, 2, 2)
            
            # Run a few simulations
            n_simulations = 5
            n_steps = 50
            
            # Sample initial states
            initial_states = []
            for i in range(n_simulations):
                if len(verifier.network.subsystems) == 1:
                    # Single 2D subsystem
                    state = np.zeros(2)
                    state[0] = np.random.uniform(
                        verifier.network.subsystems[0].initial_set[0, 0],
                        verifier.network.subsystems[0].initial_set[0, 1]
                    )
                    state[1] = np.random.uniform(
                        verifier.network.subsystems[0].initial_set[1, 0],
                        verifier.network.subsystems[0].initial_set[1, 1]
                    )
                else:
                    # Two 1D subsystems
                    state = np.zeros(2)
                    state[0] = np.random.uniform(
                        verifier.network.subsystems[0].initial_set[0, 0],
                        verifier.network.subsystems[0].initial_set[0, 1]
                    )
                    state[1] = np.random.uniform(
                        verifier.network.subsystems[1].initial_set[0, 0],
                        verifier.network.subsystems[1].initial_set[0, 1]
                    )
                
                initial_states.append(state)
            
            # Simulate
            for i in range(n_simulations):
                trajectory = verifier.network.simulate(initial_states[i], n_steps)
                ax2.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.7)
                ax2.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=5)  # Initial point
            
            # Add initial and unsafe sets (same as in ax1)
            if len(verifier.network.subsystems) == 1:
                # Single 2D subsystem
                initial_x = [verifier.network.subsystems[0].initial_set[0, 0], verifier.network.subsystems[0].initial_set[0, 1], 
                             verifier.network.subsystems[0].initial_set[0, 1], verifier.network.subsystems[0].initial_set[0, 0], 
                             verifier.network.subsystems[0].initial_set[0, 0]]
                initial_y = [verifier.network.subsystems[0].initial_set[1, 0], verifier.network.subsystems[0].initial_set[1, 0], 
                             verifier.network.subsystems[0].initial_set[1, 1], verifier.network.subsystems[0].initial_set[1, 1], 
                             verifier.network.subsystems[0].initial_set[1, 0]]
                
                unsafe_x = [verifier.network.subsystems[0].unsafe_set[0, 0], verifier.network.subsystems[0].unsafe_set[0, 1], 
                            verifier.network.subsystems[0].unsafe_set[0, 1], verifier.network.subsystems[0].unsafe_set[0, 0], 
                            verifier.network.subsystems[0].unsafe_set[0, 0]]
                unsafe_y = [verifier.network.subsystems[0].unsafe_set[1, 0], verifier.network.subsystems[0].unsafe_set[1, 0], 
                            verifier.network.subsystems[0].unsafe_set[1, 1], verifier.network.subsystems[0].unsafe_set[1, 1], 
                            verifier.network.subsystems[0].unsafe_set[1, 0]]
            else:
                # Two 1D subsystems
                initial_x = [verifier.network.subsystems[0].initial_set[0, 0], verifier.network.subsystems[0].initial_set[0, 1], 
                             verifier.network.subsystems[0].initial_set[0, 1], verifier.network.subsystems[0].initial_set[0, 0], 
                             verifier.network.subsystems[0].initial_set[0, 0]]
                initial_y = [verifier.network.subsystems[1].initial_set[0, 0], verifier.network.subsystems[1].initial_set[0, 0], 
                             verifier.network.subsystems[1].initial_set[0, 1], verifier.network.subsystems[1].initial_set[0, 1], 
                             verifier.network.subsystems[1].initial_set[0, 0]]
                
                unsafe_x = [verifier.network.subsystems[0].unsafe_set[0, 0], verifier.network.subsystems[0].unsafe_set[0, 1], 
                            verifier.network.subsystems[0].unsafe_set[0, 1], verifier.network.subsystems[0].unsafe_set[0, 0], 
                            verifier.network.subsystems[0].unsafe_set[0, 0]]
                unsafe_y = [verifier.network.subsystems[1].unsafe_set[0, 0], verifier.network.subsystems[1].unsafe_set[0, 0], 
                            verifier.network.subsystems[1].unsafe_set[0, 1], verifier.network.subsystems[1].unsafe_set[0, 1], 
                            verifier.network.subsystems[1].unsafe_set[0, 0]]
            
            ax2.plot(initial_x, initial_y, 'g-', linewidth=2, label='Initial Set')
            ax2.plot(unsafe_x, unsafe_y, 'r-', linewidth=2, label='Unsafe Set')
            
            ax2.set_xlabel('x₁')
            ax2.set_ylabel('x₂')
            ax2.set_title('State Trajectories')
            ax2.legend()
            ax2.grid(True)
            
            # Plot individual state components
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            
            # Plot x₁ and x₂ components of the first simulation
            for i in range(n_simulations):
                trajectory = verifier.network.simulate(initial_states[i], n_steps)
                ax3.plot(np.arange(n_steps + 1), trajectory[:, 0], alpha=0.7)
                ax4.plot(np.arange(n_steps + 1), trajectory[:, 1], alpha=0.7)
            
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('x₁')
            ax3.set_title('x₁ Trajectories')
            ax3.grid(True)
            
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('x₂')
            ax4.set_title('x₂ Trajectories')
            ax4.grid(True)
            
        # Set figure title
        verification_status = "Verified Safe" if verifier.verification_result else "Not Verified"
        confidence = verifier.verification_confidence
        fig.suptitle(f"Safety Verification Results: {verification_status} (Confidence: {confidence:.4f})", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if show:
            plt.show()
    
    else:
        # For higher dimensions, create a simple summary plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        verification_status = "Verified Safe" if verifier.verification_result else "Not Verified"
        confidence = verifier.verification_confidence
        
        # Create a simple text summary
        ax.text(0.5, 0.6, f"Safety Verification Results", fontsize=16, ha='center')
        ax.text(0.5, 0.5, f"Status: {verification_status}", fontsize=14, ha='center')
        ax.text(0.5, 0.4, f"Confidence: {confidence:.4f}", fontsize=14, ha='center')
        
        ax.axis('off')
        
        if show:
            plt.show()
    
    return fig 