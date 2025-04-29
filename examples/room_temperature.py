import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from src.systems.network import Network
from src.utils.data_generation import generate_room_temperature_system, generate_synthetic_data
from src.basis_functions.polynomial import PolynomialBarrier
from src.basis_functions.rbf import RBFBarrier
from src.verification.safety import SafetyVerifier

def main():
    """
    Run the room temperature control example from the original paper.
    """
    print("Running room temperature control example...")
    
    # Create room temperature control system
    n_rooms = 5  # Using 5 rooms for a simpler example (100 in the original paper)
    print(f"Creating room temperature control system with {n_rooms} rooms...")
    subsystems, M, initial_sets, unsafe_sets = generate_room_temperature_system(n_rooms, seed=42)
    
    # Create network
    network = Network(subsystems, M, name="Room Temperature Network")
    
    # Print network information
    print(f"Network: {network}")
    print(f"Number of subsystems: {network.n_subsystems}")
    print(f"Total state dimension: {network.total_state_dim}")
    print(f"Total input dimension: {network.total_input_dim}")
    
    # Create polynomial barrier functions (degree 2, as in the original paper)
    print("Creating polynomial barrier functions...")
    poly_barriers = [PolynomialBarrier(1, degree=2) for _ in range(n_rooms)]
    
    # Generate data for training
    print("Generating training data...")
    data_list = []
    n_samples = 1000  # Smaller than the original paper for demonstration
    
    for i, subsystem in enumerate(subsystems):
        data = generate_synthetic_data(subsystem, n_samples, noise_level=0.001)
        data_list.append(data)
    
    # Create safety verifier
    print("Creating safety verifier...")
    verifier = SafetyVerifier(network, poly_barriers, confidence=0.95, verbose=True)
    
    # Verify safety
    print("Verifying safety...")
    result = verifier.verify(data_list=data_list, n_samples=None, admm_max_iter=10)
    
    # Print verification result
    print("\nVerification Result:")
    print(f"  Verified: {result['verified']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Reason: {result['reason']}")
    
    if result['admm_result'] is not None:
        print(f"  ADMM Iterations: {result['admm_result']['iter_count']}")
        print(f"  LMI Satisfied: {result['admm_result']['lmi_satisfied']}")
        print(f"  Max Eigenvalue: {result['admm_result']['max_eigenvalue']:.6f}")
    
    # Validate with simulations
    print("\nValidating with simulations...")
    validation_result = verifier.validate_simulation(n_simulations=20, n_steps=50)
    
    print(f"Validation Result:")
    print(f"  Success: {validation_result['success']}")
    print(f"  Violations: {validation_result['violations']}")
    print(f"  Violation Rate: {validation_result['violation_rate']:.4f}")
    
    # Try a different barrier representation (RBF)
    print("\nTrying RBF barrier functions...")
    rbf_barriers = []
    
    for i in range(n_rooms):
        # Create RBF barrier with 10 centers
        rbf = RBFBarrier(1, n_centers=10, width=0.5)
        
        # Fit centers using the data
        states = data_list[i][0]
        rbf.fit_centers(states)
        
        rbf_barriers.append(rbf)
    
    # Create new verifier with RBF barriers
    rbf_verifier = SafetyVerifier(network, rbf_barriers, confidence=0.95, verbose=True)
    
    # Verify safety with RBF barriers
    print("Verifying safety with RBF barriers...")
    rbf_result = rbf_verifier.verify(data_list=data_list, n_samples=None, admm_max_iter=10)
    
    # Print verification result
    print("\nRBF Verification Result:")
    print(f"  Verified: {rbf_result['verified']}")
    print(f"  Confidence: {rbf_result['confidence']:.4f}")
    print(f"  Reason: {rbf_result['reason']}")
    
    if rbf_result['admm_result'] is not None:
        print(f"  ADMM Iterations: {rbf_result['admm_result']['iter_count']}")
        print(f"  LMI Satisfied: {rbf_result['admm_result']['lmi_satisfied']}")
        print(f"  Max Eigenvalue: {rbf_result['admm_result']['max_eigenvalue']:.6f}")
    
    # Compare polynomial and RBF barriers for the first room
    if result['verified'] and rbf_result['verified']:
        print("\nComparing polynomial and RBF barriers for the first room...")
        domain = [19, 28]  # Temperature range
        
        plt.figure(figsize=(12, 6))
        
        # Plot polynomial barrier
        plt.subplot(1, 2, 1)
        plot_barrier(poly_barriers[0], domain, subsystems[0])
        plt.title("Polynomial Barrier")
        
        # Plot RBF barrier
        plt.subplot(1, 2, 2)
        plot_barrier(rbf_barriers[0], domain, subsystems[0])
        plt.title("RBF Barrier")
        
        plt.tight_layout()
        plt.savefig("barrier_comparison.png")
        plt.show()
    
    # Visualize verification results
    if result['verified']:
        print("\nVisualizing verification results...")
        fig = verifier.visualize_barrier(save_path="poly_barrier.png")
        plt.close(fig)
        
        fig = verifier.visualize_simulation(n_simulations=5, n_steps=50, save_path="poly_simulation.png")
        plt.close(fig)
    
    if rbf_result['verified']:
        print("\nVisualizing RBF verification results...")
        fig = rbf_verifier.visualize_barrier(save_path="rbf_barrier.png")
        plt.close(fig)
        
        fig = rbf_verifier.visualize_simulation(n_simulations=5, n_steps=50, save_path="rbf_simulation.png")
        plt.close(fig)

def plot_barrier(barrier, domain, subsystem):
    """
    Plot a barrier function for the room temperature system.
    
    Args:
        barrier: Barrier function representation.
        domain (list): Domain bounds [min, max].
        subsystem: Subsystem object.
    """
    # Create grid
    x = np.linspace(domain[0], domain[1], 100)
    
    # Evaluate barrier function
    y = np.zeros(100)
    for i in range(100):
        y[i] = barrier.evaluate(np.array([[x[i]]]))[0]
    
    # Plot barrier function
    plt.plot(x, y, 'b-', linewidth=2, label='Barrier Function')
    
    # Add threshold line
    plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold (B=1)')
    
    # Add initial and unsafe sets
    x_0_min, x_0_max = subsystem.initial_set[0]
    x_u_min, x_u_max = subsystem.unsafe_set[0]
    
    plt.axvspan(x_0_min, x_0_max, alpha=0.2, color='g', label='Initial Set')
    plt.axvspan(x_u_min, x_u_max, alpha=0.2, color='r', label='Unsafe Set')
    
    # Add labels and legend
    plt.xlabel('Temperature')
    plt.ylabel('B(x)')
    plt.grid(True)
    plt.legend()

if __name__ == "__main__":
    main() 