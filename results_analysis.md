# Results Analysis: Barrier Function Representations for Data-Driven Safety Verification

## Overview

This document presents the results of comparing different barrier function representations for data-driven safety verification of discrete-time networks. The experiments test five different barrier function representations across two example systems:

1. **Room Temperature Control System**: A network of interconnected rooms with temperature dynamics
2. **Nonlinear Oscillator System**: A nonlinear 2D system with oscillatory behavior

The barrier function representations compared are:
- **Polynomial**: Traditional polynomial basis functions (degree 4)
- **RBF**: Radial basis functions with Gaussian kernels (25 centers)
- **Fourier**: Fourier basis functions (order 4)
- **Wavelet**: Wavelet basis functions (level 2)
- **Neural**: Neural network representation (hidden layers: 16, 16)

## Experimental Setup

### Room Temperature System
- 5 interconnected rooms with temperature dynamics
- State dimension per subsystem: 1 (temperature)
- Input dimension per subsystem: 2 (neighboring room temperatures)
- Initial set: Temperature between 20.5°C and 22.5°C
- Unsafe set: Temperature between 24°C and 28°C
- Training data: 1000 samples per subsystem

### Nonlinear Oscillator System
- Single nonlinear oscillator with 2D state space
- Initial set: Small region near the origin
- Unsafe set: Region away from the origin
- Training data: 2000 samples
- Validation data: 500 samples

## Results Summary

### Verification Results

| Representation | Verified | Confidence | Room Temp Time (s) | Oscillator Time (s) | Max Eigenvalue |
|----------------|----------|------------|---------------------|---------------------|----------------|
| Polynomial     | False    | 0.0000     | ~30                 | 13.24               | 0.013098       |
| RBF            | False    | 0.0000     | ~35                 | 15.83               | 0.012724       |
| Wavelet        | False    | 0.0000     | ~40                 | 23.03               | 10.383304      |
| Fourier        | False    | 0.0000     | ~38                 | 21.09               | 0.015191       |
| Neural         | False    | 0.0000     | ~60                 | 60.74               | 0.009569       |

### Representation Properties

| Representation | Parameters | Expressiveness | Lipschitz   | Computational Cost |
|----------------|------------|----------------|-------------|-------------------|
| Polynomial     | 28         | Medium         | Unbounded   | Low               |
| RBF            | 25         | High           | Controlled  | Medium            |
| Fourier        | 25         | Medium         | Bounded     | Medium-High       |
| Wavelet        | 100        | High           | Localized   | High              |
| Neural         | 13         | Very High      | Estimable   | Very High         |

## Analysis

### Verification Performance

None of the barrier function representations were able to verify the safety properties of the systems. This is indicated by:

1. **Verification Status**: All methods returned `Verified: False`
2. **Confidence Levels**: All methods returned a confidence of 0.0000
3. **LMI Condition**: The linear matrix inequality (LMI) condition was not satisfied for any method, as shown by positive eigenvalues

However, we observe interesting differences in performance:

- The **Neural Network** representation achieved the smallest maximum eigenvalue (0.009569) for the oscillator system, suggesting it came closest to satisfying the LMI condition.
- **RBF** and **Polynomial** representations had similar performance with eigenvalues of 0.012724 and 0.013098, respectively.
- The **Wavelet** representation had a significantly higher maximum eigenvalue (10.383304), indicating it struggled the most with satisfying the LMI condition.

### Computational Efficiency

The computational requirements varied significantly among the different representations:

- **Polynomial** representation was the fastest to compute (13.24 seconds for the oscillator system).
- **RBF** was slightly slower (15.83 seconds).
- **Fourier** and **Wavelet** representations required more computation time (21.09 and 23.03 seconds, respectively).
- **Neural Network** representation was by far the most computationally intensive (60.74 seconds), nearly 5 times slower than polynomial.

This pattern aligns with the general understanding that neural networks require more computational resources due to their non-convex optimization landscape and need for gradient-based training.

### Expressiveness vs. Efficiency Trade-off

The results highlight the fundamental trade-off between expressiveness and computational efficiency:

- **Polynomial** basis functions offer good computational efficiency but may lack expressiveness for complex behaviors.
- **RBF** functions provide a good balance between expressiveness and efficiency, especially with strategically placed centers.
- **Fourier** basis functions are suitable for systems with periodic behaviors but require more computation.
- **Wavelet** basis functions offer multi-resolution analysis but come with higher computational costs and in this case struggled with verification.
- **Neural Networks** provide the highest expressiveness but at a significant computational cost and with challenges in optimization.

### ADMM Convergence

The ADMM algorithm used for compositional verification requires multiple iterations to converge. From the console output:

- Each verification task ran 10 ADMM iterations.
- The primal and dual residuals decreased over iterations but did not reach the desired tolerance in the allotted iterations.
- The room temperature example showed a final primal residual of 0.079332 and dual residual of 0.068936.

This suggests that:
1. The ADMM algorithm was making progress toward a solution
2. Additional iterations might have further reduced the residuals
3. The global dissipativity condition might be inherently difficult to satisfy for these systems

## Simulation Validation

While formal verification was unsuccessful, simulation-based validation was performed. For the room temperature system, the validation resulted in:

```
Validation Result:
  Success: True
  Violations: 0
  Violation Rate: 0.0000
```

This indicates that no safety violations were observed in the simulated trajectories, despite the failure of formal verification. This discrepancy suggests that:

1. The barrier function may be overly conservative
2. The unsafe region might not be reachable in practice
3. The formal verification may be failing due to numerical issues or insufficient data

## Conclusion

These experiments provide valuable insights into the performance characteristics of different barrier function representations for safety verification:

1. **No Single Best Representation**: Each representation offers different trade-offs between expressiveness, computational efficiency, and verifiability.

2. **Verification Challenges**: Safety verification of complex systems remains challenging, with all methods failing to formally verify the safety properties despite simulation-based evidence suggesting safety.

3. **Promising Approaches**: Neural networks and RBF representations showed the most promising results in terms of coming closest to satisfying the LMI condition, suggesting that focusing on these representations with improved optimization techniques might lead to successful verification.

4. **Future Directions**: 
   - Exploring hybrid approaches that combine different representations
   - Improving the ADMM algorithm's convergence for the global optimization
   - Developing specialized optimization techniques for neural network barrier functions
   - Investigating the gap between simulation-based validation and formal verification

These findings can guide the selection of appropriate barrier function representations for different verification tasks, considering system complexity, available computational resources, and required verification confidence. 