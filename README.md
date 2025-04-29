# Alternative Barrier Function Representations for Data-Driven Safety Verification

## Project Overview

This repository contains implementation and analysis of alternative barrier function representations for data-driven safety verification of discrete-time networks. This project extends the compositional approach presented in Noroozi et al. (2022) by exploring how different basis function selections impact verification performance, computational efficiency, and safety guarantees.

## Original Paper Summary

The original paper "Data-Driven Safety Verification of Discrete-Time Networks: A Compositional Approach" by Noroozi, Salamati, and Zamani (IEEE Control Systems Letters, 2022) addresses two critical challenges in data-driven methods for system analysis: (1) the lack of formal out-of-sample performance guarantees and (2) computational complexity when dealing with large-scale interconnected systems.

The authors propose a compositional data-driven approach for safety verification of networks of discrete-time subsystems with formal guarantees. Key innovations include:

1. **Barrier Function Decomposition**: The approach decomposes the overall barrier function into "sub-barrier" functions for each subsystem, expressed as linear combinations of user-defined basis functions.

2. **Data-Driven Formulation**: For each subsystem, conditions on sub-barrier candidates are formulated as robust convex programs (RCPs), which are approximated using scenario convex programs (SCPs) based on collected data samples.

3. **Probabilistic Guarantees**: The approach provides explicit formulas to compute the minimum number of data samples required to guarantee a desired error bound between the optimal values of RCPs and corresponding SCPs.

4. **Global Compositionality**: A global dissipativity condition (expressed as a linear matrix inequality) ensures that the sum of sub-barrier functions forms a valid barrier function for the entire network.

5. **Efficient Optimization**: The resulting large-scale optimization problem is solved efficiently using the Alternating Direction Method of Multipliers (ADMM) algorithm.

The paper demonstrates the approach's effectiveness on a room temperature control problem in a 100-room building, showing significant computational advantages over non-compositional methods. The compositional approach reduces the computational complexity from exponential in the total system dimension to linear in the number of subsystems.

## Project Description

This project extends the original paper by systematically investigating alternative barrier function representations beyond the polynomial basis functions used in the original work. We explore:

1. **Diverse Basis Function Families**:
   - Polynomial basis functions (baseline)
   - Radial basis functions (RBFs)
   - Fourier basis functions
   - Wavelet basis functions
   - Neural network representations

2. **Representation Analysis**:
   - Effect of basis function selection on verification performance
   - Impact on convexity and solvability of optimization problems
   - Trade-offs between expressiveness and computational efficiency
   - Influence on the minimum required data samples

3. **Adaptive Basis Selection**:
   - Development of methods to automatically select optimal basis functions
   - Investigation of data-driven approaches for basis function adaptation
   - Sensitivity analysis of barrier function performance to basis selection

4. **Benchmark Comparison**:
   - Comprehensive comparison across different system types
   - Standardized evaluation metrics for verification quality
   - Computational complexity analysis

## Methodology

### 1. Theoretical Framework

We maintain the core theoretical framework from the original paper:
- Subsystem decomposition with sub-barrier functions
- Scenario-based convex optimization formulation
- Global compositionality via dissipativity conditions

For each alternative representation, we derive:
- Appropriate parameterizations of sub-barrier functions
- Modified optimization problems (SCPs)
- Updated Lipschitz constants for probabilistic guarantees

### 2. Implementation Components

The project implementation consists of several components:

- **Basis Function Library**: Implementations of various basis function families with configurable parameters
- **Optimization Framework**: Modified SCP formulations for each basis type
- **Data Generation**: Simulation environments for generating training and validation data
- **ADMM Solver**: Customized ADMM implementation supporting different barrier representations
- **Validation Framework**: Tools for assessing verification performance and safety guarantees
- **Visualization Tools**: Utilities for visualizing barrier functions and safe regions

### 3. Experimental Design

We design experiments to systematically evaluate each representation:

- **Case Studies**:
  - Room temperature control (from original paper)
  - Additional networked dynamical systems
  - Synthetic benchmarks with known ground truth

- **Evaluation Metrics**:
  - Verification accuracy (false positive/negative rates)
  - Sample complexity (minimum data requirements)
  - Computational efficiency (time, memory)
  - Representation compactness

- **Sensitivity Analysis**:
  - Parameter sensitivity studies
  - Robustness to noise and disturbances
  - Generalization to unseen operating conditions

## Expected Outcomes

This project aims to provide:

1. **Theoretical Insights**: Understanding the relationship between barrier function representation and verification performance
2. **Practical Guidelines**: Recommendations for selecting appropriate basis functions for different system types
3. **Improved Algorithms**: Enhanced verification techniques with better computational efficiency
4. **Benchmark Results**: Comprehensive comparison of different representations on standardized test cases
5. **Software Framework**: Reusable implementation of the compositional verification approach with support for multiple representations

## Implementation Details

### Data Structures

- `BarrierRepresentation`: Abstract base class for different barrier function representations
- `SubsystemModel`: Interface for subsystem dynamics and data collection
- `ScenarioProgram`: Implementation of scenario convex programs for different representations
- `ADMMSolver`: Modified ADMM algorithm supporting various barrier representations
- `SafetyVerifier`: High-level interface for safety verification

### Key Algorithms

1. **Basis Function Selection**:
   - Adaptive basis selection based on data characteristics
   - Cross-validation techniques for representation quality

2. **Modified SCP Formulation**:
   - Tailored constraints for different basis types
   - Specialized solvers for specific representations

3. **Enhanced ADMM**:
   - Customized update rules for different barrier representations
   - Acceleration techniques for faster convergence

4. **Verification Procedures**:
   - Procedures for computing probabilistic safety guarantees
   - Methods for validating sub-barrier conditions

### Dependencies

- MATLAB with optimization toolboxes
- CVX for solving convex programs
- YALMIP for handling LMIs
- Python (optional) for visualization and analysis

## Research Questions

This project specifically addresses the following research questions:

1. How does the choice of basis functions impact the expressiveness of barrier certificates?
2. What are the trade-offs between representation power and computational complexity?
3. Can adaptive basis selection improve verification performance?
4. How do different representations affect the minimum required number of data samples?
5. What is the relationship between basis function selection and the global dissipativity condition?

## Timeline and Milestones

1. **Project Setup and Baseline Implementation** (Weeks 1-2)
   - Reproduce results from original paper
   - Implement infrastructure for alternative representations

2. **Implementation of Alternative Representations** (Weeks 3-5)
   - Develop and test each basis function family
   - Implement modified optimization problems

3. **Experimental Evaluation** (Weeks 6-8)
   - Conduct comparative studies on benchmark systems
   - Analyze verification performance and computational efficiency

4. **Adaptive Methods Development** (Weeks 9-10)
   - Implement and evaluate adaptive basis selection
   - Conduct sensitivity studies

5. **Documentation and Analysis** (Weeks 11-12)
   - Compile comprehensive benchmark results
   - Prepare final report and documentation

## Contribution to Convex Optimization

This project contributes to the field of convex optimization by:

1. Exploring how different function representations affect the convexity properties of safety verification problems
2. Investigating the relationship between function parameterization and optimization landscape
3. Developing techniques to handle potentially non-convex representations within a convex optimization framework
4. Analyzing the impact of representation choice on the trade-off between expressiveness and tractability

## References

1. Noroozi, N., Salamati, A., & Zamani, M. (2022). Data-Driven Safety Verification of Discrete-Time Networks: A Compositional Approach. *IEEE Control Systems Letters*, 6, 2210-2215.
2. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning*, 3(1), 1-122.
3. Esfahani, P. M., Sutter, T., & Lygeros, J. (2015). Performance bounds for the scenario approach and an extension to a class of non-convex programs. *IEEE Transactions on Automatic Control*, 60(1), 46-58.
4. Prajna, S., Jadbabaie, A., & Pappas, G. J. (2007). A framework for worst-case and stochastic safety verification using barrier certificates. *IEEE Transactions on Automatic Control*, 52(8), 1415-1428.

## Installation and Usage

[To be completed with implementation-specific details]

## Contributors

[To be completed with team information]