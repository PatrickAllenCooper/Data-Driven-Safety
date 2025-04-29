import numpy as np
from src.systems.subsystem import LinearSubsystem, NonlinearSubsystem

def generate_room_temperature_system(n_rooms=100, T_E=15, T_h=55, seed=None):
    """
    Generate the room temperature control system from the original paper.
    
    This function creates a network of interconnected rooms, where each room
    is a subsystem with temperature dynamics.
    
    Args:
        n_rooms (int, optional): Number of rooms. Defaults to 100.
        T_E (float, optional): External temperature. Defaults to 15.
        T_h (float, optional): Heater temperature. Defaults to 55.
        seed (int, optional): Random seed. Defaults to None.
        
    Returns:
        tuple: (subsystems, interconnection_matrix, initial_sets, unsafe_sets).
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Set parameters from the paper
    alpha = 5e-2  # Heat transfer coefficient between rooms
    alpha_e = 8e-3  # Heat transfer coefficient to exterior
    alpha_h = 3.6e-3  # Heater coefficient
    
    # State bounds for each room
    state_bounds = np.array([[19, 28]])  # Temperature range
    
    # Input bounds (neighboring room temperatures)
    input_bounds = np.array([[19, 28]] * 2)  # Two neighbors for each room
    
    # Initial and unsafe sets for each room
    initial_set = np.array([[20.5, 22.5]])  # Initial temperature range
    unsafe_set = np.array([[24, 28]])  # Unsafe temperature range (too hot)
    
    # Create subsystems (rooms)
    subsystems = []
    
    for i in range(n_rooms):
        # Neighboring room indices
        left = (i - 1) % n_rooms
        right = (i + 1) % n_rooms
        
        # Controller for room i: u_i(x_i) = -0.002398 * x_i + 0.5357
        # From the original paper
        
        # Linearized dynamics for room i
        # x_i(k+1) = a * x_i(k) + b_left * x_left(k) + b_right * x_right(k) + c
        
        # a = 1 - 2*alpha - alpha_e + alpha_h*(-0.002398*x_i + 0.5357) - 0.002398*alpha_h*T_h
        a = 1 - 2*alpha - alpha_e - 0.002398*alpha_h*T_h
        
        # Linear coefficient for x_i
        A = np.array([[a + alpha_h*(-0.002398)]])
        
        # Linear coefficients for inputs (neighboring rooms)
        B = np.array([[alpha, alpha]])
        
        # Constant term
        c = np.array([alpha_e*T_E + 0.5357*alpha_h*T_h])
        
        # Create subsystem
        subsystems.append(LinearSubsystem(
            A=A, B=B, c=c,
            state_bounds=state_bounds,
            input_bounds=input_bounds,
            name=f"Room_{i}"
        ))
        
        # Set initial and unsafe sets
        subsystems[i].set_initial_set(initial_set)
        subsystems[i].set_unsafe_set(unsafe_set)
    
    # Create interconnection matrix
    # For each room, the inputs are the temperatures of the neighboring rooms
    M = np.zeros((2*n_rooms, n_rooms))
    
    for i in range(n_rooms):
        left = (i - 1) % n_rooms
        right = (i + 1) % n_rooms
        
        # First input is from the left room
        M[2*i, left] = 1
        
        # Second input is from the right room
        M[2*i + 1, right] = 1
    
    # Prepare initial and unsafe sets for all subsystems
    initial_sets = [initial_set] * n_rooms
    unsafe_sets = [unsafe_set] * n_rooms
    
    return subsystems, M, initial_sets, unsafe_sets

def generate_nonlinear_oscillator_system(n_oscillators=2, coupling_strength=0.1, seed=None):
    """
    Generate a network of coupled nonlinear oscillators.
    
    Each oscillator is a 2D nonlinear system with position and velocity.
    
    Args:
        n_oscillators (int, optional): Number of oscillators. Defaults to 2.
        coupling_strength (float, optional): Coupling strength between oscillators. Defaults to 0.1.
        seed (int, optional): Random seed. Defaults to None.
        
    Returns:
        tuple: (subsystems, interconnection_matrix, initial_sets, unsafe_sets).
    """
    if seed is not None:
        np.random.seed(seed)
        
    # State bounds for each oscillator
    state_bounds = np.array([[-2, 2], [-2, 2]])  # Position and velocity bounds
    
    # Input bounds (position and velocity of neighboring oscillators)
    input_bounds = np.array([[-2, 2], [-2, 2]] * n_oscillators)
    
    # Initial and unsafe sets for each oscillator
    initial_set = np.array([[-0.1, 0.1], [-0.1, 0.1]])  # Near equilibrium
    unsafe_set = np.array([[1.5, 2], [-2, 2]])  # Position too far from equilibrium
    
    # Create subsystems (oscillators)
    subsystems = []
    
    for i in range(n_oscillators):
        # Nonlinear dynamics for oscillator i
        def dynamics_fn(state, inputs, i=i):
            # state: [position, velocity]
            position, velocity = state
            
            # Calculate coupling forces from neighbors
            coupling_force = 0
            for j in range(n_oscillators):
                if j != i:
                    # Extract position of oscillator j from inputs
                    position_j = inputs[2*j]
                    coupling_force += coupling_strength * (position_j - position)
            
            # Damped oscillator dynamics with coupling
            # dx/dt = v
            # dv/dt = -x - 0.1*v + coupling_force
            
            # Euler integration with dt = 0.1
            dt = 0.1
            position_next = position + dt * velocity
            velocity_next = velocity + dt * (-position - 0.1*velocity + coupling_force)
            
            return np.array([position_next, velocity_next])
        
        # Create wrapper for dynamics function to match the expected signature
        dynamics_wrapper = lambda state, inputs, i=i: dynamics_fn(state, inputs, i)
        
        # Create subsystem
        subsystems.append(NonlinearSubsystem(
            dynamics_fn=dynamics_wrapper,
            state_dim=2,
            input_dim=2*n_oscillators,
            state_bounds=state_bounds,
            input_bounds=input_bounds,
            name=f"Oscillator_{i}"
        ))
        
        # Set initial and unsafe sets
        subsystems[i].set_initial_set(initial_set)
        subsystems[i].set_unsafe_set(unsafe_set)
    
    # Create interconnection matrix
    # For each oscillator, the inputs are the states of all oscillators (including itself)
    M = np.zeros((2*n_oscillators*n_oscillators, 2*n_oscillators))
    
    for i in range(n_oscillators):
        for j in range(n_oscillators):
            # Input indices for oscillator i from oscillator j
            row_start = 2*n_oscillators*i + 2*j
            col_start = 2*j
            
            # Position
            M[row_start, col_start] = 1
            # Velocity
            M[row_start + 1, col_start + 1] = 1
    
    # Prepare initial and unsafe sets for all subsystems
    initial_sets = [initial_set] * n_oscillators
    unsafe_sets = [unsafe_set] * n_oscillators
    
    return subsystems, M, initial_sets, unsafe_sets

def generate_synthetic_data(subsystem, n_samples, noise_level=0.01, from_initial=False):
    """
    Generate synthetic data for a subsystem with optional noise.
    
    Args:
        subsystem: Subsystem object.
        n_samples (int): Number of samples to generate.
        noise_level (float, optional): Standard deviation of noise. Defaults to 0.01.
        from_initial (bool, optional): Whether to sample from the initial set. Defaults to False.
        
    Returns:
        tuple: (states, inputs, next_states) with added noise.
    """
    # Generate clean data
    states, inputs, next_states = subsystem.generate_data(n_samples, from_initial)
    
    # Add noise to next states
    noise = np.random.normal(0, noise_level, next_states.shape)
    next_states_noisy = next_states + noise
    
    return states, inputs, next_states_noisy

def generate_training_validation_data(subsystem, n_train, n_val, noise_level=0.01, from_initial=False):
    """
    Generate training and validation data for a subsystem.
    
    Args:
        subsystem: Subsystem object.
        n_train (int): Number of training samples.
        n_val (int): Number of validation samples.
        noise_level (float, optional): Standard deviation of noise. Defaults to 0.01.
        from_initial (bool, optional): Whether to sample from the initial set. Defaults to False.
        
    Returns:
        tuple: (train_data, val_data), where each is a tuple (states, inputs, next_states).
    """
    # Generate training data
    train_data = generate_synthetic_data(subsystem, n_train, noise_level, from_initial)
    
    # Generate validation data
    val_data = generate_synthetic_data(subsystem, n_val, noise_level, from_initial)
    
    return train_data, val_data 