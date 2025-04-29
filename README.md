# Alternative Barrier Function Representations for Data-Driven Safety Verification

## Project Overview

This repository contains implementation and analysis of alternative barrier function representations for data-driven safety verification of discrete-time networks. This project extends the compositional approach presented in Noroozi et al. (2022) by exploring how different basis function selections impact verification performance, computational efficiency, and safety guarantees.

## Technical Details

### Mathematical Foundations

#### Barrier Functions

A function $B: \mathcal{X} \rightarrow \mathbb{R}$ is a barrier function for a discrete-time system $x^+ = g(x)$ if there exist $\gamma, \sigma \in \mathbb{R}$, $\gamma < \sigma$, such that:

$$B(x) \leq \gamma, \quad \forall x \in \mathcal{X}_0,$$
$$B(x) \geq \sigma, \quad \forall x \in \mathcal{X}_u,$$
$$B(g(x)) \leq B(x), \quad \forall x \in \mathcal{X}.$$

where $\mathcal{X}_0$ is the initial set and $\mathcal{X}_u$ is the unsafe set.

#### Compositional Approach

For a network of interconnected subsystems $\Sigma_i: x_i^+ = g_i(x_i, w_i)$, we define sub-barrier functions $B_i: \mathcal{X}_i \rightarrow \mathbb{R}$ for each subsystem satisfying:

$$B_i(x_i) \leq \gamma_i, \quad \forall x_i \in \mathcal{X}_{0i},$$
$$B_i(x_i) \geq 1, \quad \forall x_i \in \mathcal{X}_{ui},$$
$$B_i(g_i(x_i, w_i)) \leq B_i(x_i) + z_i^\top X_i z_i, \quad \forall x_i \in \mathcal{X}_i, w_i \in \mathcal{W}_i,$$

where $z_i = [w_i^\top, x_i^\top]^\top$ and $X_i$ is a dissipation matrix.

If a global dissipativity condition (formulated as a linear matrix inequality) holds, then $B(x) = \sum_{i=1}^\ell B_i(x_i)$ is a barrier function for the entire network.

### Data-Driven Framework

#### Scenario Convex Programming (SCP)

Since the system dynamics are unknown, we approximate the barrier function verification conditions using sampled data. For each subsystem, we formulate a robust convex program (RCP) and approximate it with a scenario convex program (SCP):

$$\min_{\eta_i, v_i, X_i} \eta_i$$
$$\text{s.t.} \max_{j \in \{1,2,3\}} c_j(\hat{x}_{il}, \hat{w}_{il}, v_i, X_i) \leq \eta_i, \quad \forall (\hat{x}_{il}, \hat{w}_{il}, \hat{x}_{il}^+) \in \mathcal{D}_i,$$
$$v_i := (\gamma_i, q_i), \quad \gamma_i < 1,$$

where $c_j$ represent the barrier function conditions and $\mathcal{D}_i$ is the collected data.

#### Alternating Direction Method of Multipliers (ADMM)

To ensure the global dissipativity condition holds, we use ADMM to solve a distributed optimization problem:

$$\min_{\eta_i, v_i, X_i, Z_i} \sum_{i=1}^\ell \left( \eta_i + \mathbb{1}_{\mathcal{L}_i}(\eta_i, v_i, X_i) \right) + \mathbb{1}_{\mathcal{G}}(Z_1, \ldots, Z_\ell)$$
$$\text{s.t.} X_i - Z_i = 0.$$

This iterative algorithm alternates between local optimizations (for each subsystem) and a global optimization step enforcing the dissipativity condition.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/data-driven-safety.git
   cd data-driven-safety
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

The project is organized as follows:

```
data-driven-safety/
├── examples/                  # Example applications
│   ├── room_temperature.py    # Room temperature control example (from original paper)
│   └── comparison.py          # Comparison of barrier function representations
├── src/                       # Source code
│   ├── basis_functions/       # Different barrier function representations
│   │   ├── polynomial.py      # Polynomial basis functions (baseline)
│   │   ├── rbf.py             # Radial basis functions
│   │   ├── fourier.py         # Fourier basis functions
│   │   ├── wavelet.py         # Wavelet basis functions
│   │   └── neural.py          # Neural network representation
│   ├── optimization/          # Optimization algorithms
│   │   ├── scp.py             # Scenario convex programs
│   │   └── admm.py            # ADMM algorithm for global optimization
│   ├── systems/               # System models
│   │   ├── subsystem.py       # Subsystem classes
│   │   └── network.py         # Network class for interconnected systems
│   ├── verification/          # Verification framework
│   │   ├── barrier.py         # Barrier function verification
│   │   └── safety.py          # Safety verification interface
│   └── utils/                 # Utility functions
│       ├── data_generation.py # Data generation utilities
│       └── visualization.py   # Visualization utilities
├── requirements.txt           # Required Python packages
├── run_examples.py            # Script to run examples
└── README.md                  # This file
```

## Usage

### Running Examples

To run all examples:

```bash
python run_examples.py
```

To run a specific example:

```bash
python run_examples.py room_temperature  # Run room temperature example
python run_examples.py comparison        # Run comparison example
```

### Example: Room Temperature Control

This example implements the room temperature control problem from the original paper. It compares polynomial and RBF barrier functions for safety verification of a building with multiple rooms.

```python
from src.systems.network import Network
from src.utils.data_generation import generate_room_temperature_system
from src.basis_functions.polynomial import PolynomialBarrier
from src.verification.safety import SafetyVerifier

# Create system
subsystems, M, initial_sets, unsafe_sets = generate_room_temperature_system(n_rooms=5)
network = Network(subsystems, M)

# Create barrier functions
barriers = [PolynomialBarrier(1, degree=2) for _ in range(len(subsystems))]

# Create verifier
verifier = SafetyVerifier(network, barriers, confidence=0.95)

# Verify safety
result = verifier.verify()
print(f"Verified: {result['verified']}, Confidence: {result['confidence']:.4f}")
```

### Implementing Custom Barrier Functions

You can implement custom barrier function representations by extending the `BarrierRepresentation` base class:

```python
from src.basis_functions import BarrierRepresentation
import numpy as np

class CustomBarrier(BarrierRepresentation):
    def __init__(self, dimension, param1, param2):
        super().__init__(dimension)
        self.param1 = param1
        self.param2 = param2
        self._n_params = ... # Number of parameters
    
    def num_parameters(self):
        return self._n_params
    
    def evaluate(self, x, params=None):
        # Implement evaluation logic
        ...
        
    def gradient(self, x, params=None):
        # Implement gradient computation
        ...
        
    def lipschitz_constant(self, domain_bounds, params=None):
        # Implement Lipschitz constant estimation
        ...
```

## Features

- **Multiple Barrier Function Representations**: 

  - **Polynomial**: Classic choice using monomials as basis functions. For a 1D system, $B(x) = \sum_{j=0}^{d} q_j x^j$ where $d$ is the polynomial degree.
  
  - **Radial Basis Functions (RBF)**: Uses Gaussian RBFs, $B(x) = \sum_{j=1}^{n} q_j \exp(-\|x-c_j\|^2/2\sigma^2) + b$ where $c_j$ are centers and $\sigma$ is the width parameter.
  
  - **Fourier**: Uses sinusoidal basis functions, particularly effective for periodic behaviors. For 1D: $B(x) = \frac{a_0}{2} + \sum_{j=1}^{n} [a_j \cos(j\omega x) + b_j \sin(j\omega x)]$.
  
  - **Wavelet**: Localized basis functions at different scales, offering multi-resolution analysis. Uses a wavelet family (e.g., Daubechies) with coefficients at multiple scales.
  
  - **Neural Network**: Adaptive representation using feedforward neural networks, offering high expressiveness but requiring different optimization techniques.

- **Compositional Verification**: Verify large-scale systems by decomposing into subsystems
- **Data-Driven Approach**: Learn barrier functions from data with formal guarantees
- **Visualization Tools**: Visualize barrier functions, safe regions, and system trajectories
- **Comparative Analysis**: Compare different representations on standardized test cases

## Experimental Results

The project provides comprehensive comparisons of different barrier function representations:

1. **Verification Performance**: Success rate and confidence level
2. **Computational Efficiency**: Time and memory requirements
3. **Expressiveness**: Ability to represent complex barrier functions
4. **Sample Complexity**: Minimum data requirements for reliable verification
5. **Trade-offs**: Balance between expressiveness and optimization complexity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Noroozi, N., Salamati, A., & Zamani, M. (2022). Data-Driven Safety Verification of Discrete-Time Networks: A Compositional Approach. *IEEE Control Systems Letters*, 6, 2210-2215.
2. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning*, 3(1), 1-122.
3. Esfahani, P. M., Sutter, T., & Lygeros, J. (2015). Performance bounds for the scenario approach and an extension to a class of non-convex programs. *IEEE Transactions on Automatic Control*, 60(1), 46-58.
4. Prajna, S., Jadbabaie, A., & Pappas, G. J. (2007). A framework for worst-case and stochastic safety verification using barrier certificates. *IEEE Transactions on Automatic Control*, 52(8), 1415-1428.