import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# Add the parent directory to the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from src.systems.network import Network
from src.utils.data_generation import (
    generate_nonlinear_oscillator_system,
    generate_synthetic_data,
    generate_training_validation_data
)
from src.utils.visualization import (
    plot_barrier_function_2d,
    plot_comparison,
    plot_verification_results
)
from src.basis_functions.polynomial import PolynomialBarrier
from src.basis_functions.rbf import RBFBarrier
from src.basis_functions.fourier import FourierBarrier
from src.basis_functions.wavelet import WaveletBarrier
from src.basis_functions.neural import NeuralBarrier
from src.verification.safety import SafetyVerifier

def compare_representations():
    """
    Compare different barrier function representations on a nonlinear oscillator system.
    """
    print("Comparing different barrier function representations...")
    
    # Create nonlinear oscillator system
    n_oscillators = 1  # Single 2D oscillator
    print(f"Creating nonlinear oscillator system...")
    subsystems, M, initial_sets, unsafe_sets = generate_nonlinear_oscillator_system(n_oscillators, seed=42)
    
    # Create network
    network = Network(subsystems, M, name="Nonlinear Oscillator")
    
    # Print network information
    print(f"Network: {network}")
    print(f"Number of subsystems: {network.n_subsystems}")
    print(f"Total state dimension: {network.total_state_dim}")
    print(f"Total input dimension: {network.total_input_dim}")
    
    # Generate data for training and validation
    print("Generating training and validation data...")
    n_train = 2000
    n_val = 500
    
    train_data, val_data = generate_training_validation_data(
        subsystems[0], n_train, n_val, noise_level=0.001
    )
    data_list = [train_data]
    
    # Create different barrier representations
    print("Creating barrier representations...")
    
    # Domain bounds
    domain_x = [-2, 2]
    domain_y = [-2, 2]
    domain_bounds = np.array([domain_x, domain_y])
    
    # Polynomial barrier (degree 4)
    poly_barrier = PolynomialBarrier(2, degree=4)
    
    # RBF barrier (25 centers)
    rbf_barrier = RBFBarrier(2, n_centers=25, width=0.5)
    rbf_barrier.fit_centers(train_data[0])  # Fit centers using training data
    
    # Fourier barrier (order 4)
    fourier_barrier = FourierBarrier(2, n_order=4, domain_bounds=domain_bounds)
    
    # Wavelet barrier (level 2)
    wavelet_barrier = WaveletBarrier(2, level=2, domain_bounds=domain_bounds)
    
    # Neural network barrier (2 hidden layers with 16 neurons each)
    neural_barrier = NeuralBarrier(2, hidden_layers=(16, 16), activation='tanh')
    
    # Store all representations in a list
    representations = [
        ("Polynomial", poly_barrier),
        ("RBF", rbf_barrier),
        ("Fourier", fourier_barrier),
        ("Wavelet", wavelet_barrier),
        ("Neural", neural_barrier)
    ]
    
    # Compare verification performance
    results = []
    verifiers = []
    
    for name, barrier in representations:
        print(f"\nVerifying with {name} barrier...")
        
        # Create verifier
        verifier = SafetyVerifier(network, [barrier], confidence=0.95, verbose=False)
        verifiers.append(verifier)
        
        # Verify safety (measure time)
        start_time = time.time()
        result = verifier.verify(data_list=data_list, n_samples=None, admm_max_iter=10)
        end_time = time.time()
        
        # Store result
        result['time'] = end_time - start_time
        results.append((name, result))
        
        # Print verification result
        print(f"{name} Verification Result:")
        print(f"  Verified: {result['verified']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Time: {result['time']:.2f} seconds")
        
        if result['admm_result'] is not None:
            print(f"  ADMM Iterations: {result['admm_result']['iter_count']}")
            print(f"  LMI Satisfied: {result['admm_result']['lmi_satisfied']}")
            print(f"  Max Eigenvalue: {result['admm_result']['max_eigenvalue']:.6f}")
    
    # Compare barrier functions visually
    print("\nComparing barrier functions visually...")
    barrier_list = [barrier for _, barrier in representations]
    labels = [name for name, _ in representations]
    
    fig = plot_comparison(barrier_list, domain_x, domain_y, labels=labels, show=False)
    fig.suptitle("Comparison of Barrier Function Representations", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("barrier_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot each barrier function in detail
    print("Plotting each barrier function in detail...")
    for i, (name, barrier) in enumerate(representations):
        if i < len(verifiers) and verifiers[i].verification_result:
            fig = verifiers[i].visualize_barrier(save_path=f"{name.lower()}_barrier.png")
            plt.close(fig)
            
            fig = verifiers[i].visualize_simulation(n_simulations=5, n_steps=50, save_path=f"{name.lower()}_simulation.png")
            plt.close(fig)
    
    # Validate with simulations
    print("\nValidating with simulations...")
    validation_results = []
    
    for i, (name, _) in enumerate(representations):
        if i < len(verifiers) and verifiers[i].verification_result:
            validation_result = verifiers[i].validate_simulation(n_simulations=20, n_steps=50)
            validation_results.append((name, validation_result))
            
            print(f"{name} Validation Result:")
            print(f"  Success: {validation_result['success']}")
            print(f"  Violations: {validation_result['violations']}")
            print(f"  Violation Rate: {validation_result['violation_rate']:.4f}")
    
    # Create comparison table
    print("\nComparison Summary:")
    print("-" * 85)
    print(f"{'Representation':<15} | {'Verified':<8} | {'Confidence':<10} | {'Time (s)':<9} | {'Violations':<10}")
    print("-" * 85)
    
    for i, (name, result) in enumerate(results):
        verified = result['verified']
        confidence = result['confidence']
        comp_time = result['time']
        
        violations = "N/A"
        if verified and i < len(validation_results):
            validation_result = [r[1] for r in validation_results if r[0] == name]
            if validation_result:
                violations = validation_result[0]['violations']
        
        print(f"{name:<15} | {str(verified):<8} | {confidence:10.4f} | {comp_time:9.2f} | {violations:<10}")
    
    print("-" * 85)
    
    # Return verifiers for further analysis if needed
    return verifiers, results, validation_results

def analyze_representation_properties():
    """
    Analyze properties of different barrier function representations.
    """
    print("\nAnalyzing Representation Properties:")
    print("-" * 100)
    print(f"{'Representation':<15} | {'Parameters':<10} | {'Expressiveness':<13} | {'Lipschitz':<10} | {'Computational Cost':<20}")
    print("-" * 100)
    
    # Create representations with comparable complexity
    n_params = 25  # Target parameter count
    
    # Domain bounds for a 2D example
    domain_x = [-2, 2]
    domain_y = [-2, 2]
    domain_bounds = np.array([domain_x, domain_y])
    
    # Compute degree needed for polynomial to have approximately n_params parameters
    # For 2D with degree d: n_params = (d+1)(d+2)/2
    poly_degree = int(np.sqrt(2 * n_params) - 1)
    poly_barrier = PolynomialBarrier(2, degree=poly_degree)
    actual_poly_params = poly_barrier.num_parameters()
    
    # RBF with n_centers = n_params - 1 (for the bias term)
    rbf_barrier = RBFBarrier(2, n_centers=n_params-1, width=0.5)
    
    # Fourier barrier (choose order to get close to n_params)
    # For 2D with order n: n_params = (2n+1)^2
    fourier_order = int(np.sqrt(n_params) // 2)
    fourier_barrier = FourierBarrier(2, n_order=fourier_order, domain_bounds=domain_bounds)
    actual_fourier_params = fourier_barrier.num_parameters()
    
    # Wavelet barrier
    # For 2D with level l: n_params depends on wavelet family
    wavelet_level = 2  # Choose level that gives reasonable parameter count
    wavelet_barrier = WaveletBarrier(2, level=wavelet_level, domain_bounds=domain_bounds)
    actual_wavelet_params = wavelet_barrier.num_parameters()
    
    # Neural network barrier (choose architecture to get close to n_params)
    # For 2 layers with h neurons each: n_params ≈ 2*h + h^2 + 1
    # Solve for h: h^2 + 2h + 1 - n_params = 0
    h = int(np.sqrt(n_params - 1) - 1)
    neural_barrier = NeuralBarrier(2, hidden_layers=(h,), activation='tanh')
    actual_neural_params = neural_barrier.num_parameters()
    
    # Dictionary of representation properties
    properties = {
        "Polynomial": {
            "parameters": actual_poly_params,
            "expressiveness": "Medium",
            "lipschitz": "Unbounded",
            "computational_cost": "Low"
        },
        "RBF": {
            "parameters": rbf_barrier.num_parameters(),
            "expressiveness": "High",
            "lipschitz": "Controlled",
            "computational_cost": "Medium"
        },
        "Fourier": {
            "parameters": actual_fourier_params,
            "expressiveness": "Medium",
            "lipschitz": "Bounded",
            "computational_cost": "Medium-High"
        },
        "Wavelet": {
            "parameters": actual_wavelet_params,
            "expressiveness": "High",
            "lipschitz": "Localized",
            "computational_cost": "High"
        },
        "Neural": {
            "parameters": actual_neural_params,
            "expressiveness": "Very High",
            "lipschitz": "Estimable",
            "computational_cost": "Very High"
        }
    }
    
    # Print properties
    for name, props in properties.items():
        print(f"{name:<15} | {props['parameters']:<10} | {props['expressiveness']:<13} | {props['lipschitz']:<10} | {props['computational_cost']:<20}")
    
    print("-" * 100)
    print("Expressiveness: Ability to represent complex functions")
    print("Lipschitz: Behavior of Lipschitz constant")
    print("Computational Cost: Relative cost of evaluation and optimization")
    
    return properties

if __name__ == "__main__":
    # Run comparison
    verifiers, results, validation_results = compare_representations()
    
    # Analyze properties
    properties = analyze_representation_properties() 