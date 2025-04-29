# Data-Driven Safety Verification of Discrete-Time Networks: A Compositional Approach

**Authors**: Navid Noroozi, Ali Salamati, Majid Zamani  
**Published in**: IEEE Control Systems Letters, Vol. 6, 2022  
**Manuscript received**: September 14, 2021; revised November 14, 2021; accepted December 1, 2021  
**Publication date**: December 14, 2021  
**Funding**: Supported by Deutsche Forschungsgemeinschaft (DFG) under Grant WI 1458/16-1 for Noroozi and Salamati, and H2020 ERC Starting Grant AutoCPS for Zamani.

---

## Abstract

This letter proposes a compositional data-driven approach for safety verification of networks of discrete-time subsystems with formal guarantees. For each subsystem, we search for a sub-barrier candidate represented as a linear combination of user-defined basis functions. Conditions on sub-barrier candidates are formulated as robust convex programs (RCPs), approximated via scenario convex programs (SCPs) using sampled data. We provide a formula to compute the minimum number of samples ensuring a desired mismatch between RCP and SCP optimal values probabilistically. A global dissipativity condition ensures the sum of sub-barrier candidates forms a barrier function for the network. The resulting optimization problem is solved efficiently using the alternating direction method of multipliers (ADMM). The approach is applied to a 100-room building temperature control problem.

**Index Terms**: Data-driven methods, formal safety verification, interconnected systems, barrier functions.

---

## I. Introduction

The increasing complexity of dynamical networks, coupled with advances in big data and distributed sensing, has driven the adoption of data-driven methods for system analysis and control. This letter focuses on safety verification of discrete-time networks with unknown dynamics, using a compositional approach that leverages barrier functions and data-driven techniques.

**Manuscript Details**:  
- **Received**: September 14, 2021  
- **Revised**: November 14, 2021  
- **Accepted**: December 1, 2021  
- **Published**: December 14, 2021  
- **Corresponding Author**: Navid Noroozi (navid.noroozi@signon-group.com)  
- **Affiliations**:  
  - Navid Noroozi: SIGNON Deutschland GmbH, Berlin, Germany  
  - Ali Salamati: Ludwig Maximilian University of Munich, Germany  
  - Majid Zamani: Ludwig Maximilian University of Munich, Germany, and University of Colorado Boulder, USA  

---

## II. Literature Review

Recent approaches combine barrier certificates with data-driven techniques for safety verification:  
- **[11]**: Verifies parametric continuous-time nonlinear systems using barrier functions and collected data.  
- **[12]**: Synthesizes controllers from limited data and performs safety analysis with barrier certificates.  
- **[13]**: Learns control barrier functions from safe trajectories with Lipschitz continuity assumptions.  
- **[14]**: Synthesizes controllers for unknown nonlinear systems using Gaussian processes and barrier certificates.  
- **[15]**: Provides a scenario-driven approach for sample requirements in chance-constrained optimization.  
- **[16]**: Computes barrier functions for stochastic systems but faces scalability issues due to exponential data growth with system dimension.

**Contributions**:  
- Reduces computational complexity to subsystem level, linear in the number of subsystems.  
- Provides probabilistic safety confidence for the network based on subsystem confidences.  
- First result integrating data-driven techniques, barrier functions, and dissipativity for unknown interconnected systems.

---

## III. Notation

- For vectors \( (x, y) \in \mathbb{R}^n \times \mathbb{R}^w \), \( (x, y) = [x^\top, y^\top]^\top \).  
- \( I_n \): Identity matrix of dimension \( n \).  
- \( |v|_\infty \): Infinity norm of vector \( v \in \mathbb{R}^n \).  
- \( \|A\|_F = \sqrt{\text{trace}(A^\top A)} \): Frobenius norm of matrix \( A \in \mathbb{R}^{n \times w} \).  
- \( \mathbb{1}_\mathcal{A}(x) \): Indicator function, 1 if \( x \in \mathcal{A} \), 0 otherwise.  
- \( (\Omega, \mathcal{F}, \mathbb{P}) \): Probability space; \( \Omega^N \): \( N \)-Cartesian product with measure \( \mathbb{P}^N \).

---

## IV. Problem Statement

Consider a discrete-time system:
\[
\Sigma: x^+ = g(x),
\]
where \( x \in \mathcal{X} \subseteq \mathbb{R}^n \), and \( g: \mathcal{X} \rightarrow \mathcal{X} \) is unknown. The goal is to verify if trajectories starting from an initial set \( \mathcal{X}_0 \subseteq \mathcal{X} \setminus \mathcal{X}_u \) avoid an unsafe set \( \mathcal{X}_u \subseteq \mathcal{X} \).

**Definition 1 (Barrier Function)**: A function \( B: \mathcal{X} \rightarrow \mathbb{R} \) is a barrier function for \( \Sigma \) if there exist \( \gamma, \sigma \in \mathbb{R} \), \( \gamma < \sigma \), such that:
\[
B(x) \leq \gamma, \quad \forall x \in \mathcal{X}_0,
\]
\[
B(x) \geq \sigma, \quad \forall x \in \mathcal{X}_u,
\]
\[
B(g(x)) \leq B(x), \quad \forall x \in \mathcal{X}.
\]

**System Decomposition**: The system \( \Sigma \) is an interconnection of \( \ell \) subsystems:
\[
\Sigma_i: x_i^+ = g_i(x_i, w_i),
\]
where \( x_i \in \mathcal{X}_i \subseteq \mathbb{R}^{n_i} \), \( w_i \in \mathcal{W}_i \subseteq \mathbb{R}^{p_i} \), \( \mathcal{X} = \prod_{i=1}^\ell \mathcal{X}_i \), \( \mathcal{W} = \prod_{i=1}^\ell \mathcal{W}_i \). The interconnection is defined by a static matrix \( M \in \mathbb{R}^{n \times p} \), where \( n = \sum_{i=1}^\ell n_i \), \( p = \sum_{i=1}^\ell p_i \), and \( (w_1, \ldots, w_\ell) = M(x_1, \ldots, x_\ell) \).

**Sets**: Initial and unsafe sets are partitioned as \( \mathcal{X}_0 = \prod_{i=1}^\ell \mathcal{X}_{0i} \), \( \mathcal{X}_u = \prod_{i=1}^\ell \mathcal{X}_{ui} \).

**Definition 2 (Sub-Barrier Function)**: A function \( B_i: \mathcal{X}_i \rightarrow \mathbb{R} \) is a sub-barrier function for \( \Sigma_i \) if there exist a matrix \( X_i \in \mathbb{R}^{(p_i+n_i) \times (p_i+n_i)} \) with partitions \( Purchasing X_i^{j,k} \), \( j,k \in \{1,2\} \), and \( \gamma_i < 1 \) such that:
\[
B_i(x_i) \leq \gamma_i, \quad \forall x_i \in \mathcal{X}_{0i},
\]
\[
B_i(x_i) \geq 1, \quad \forall x_i \in \mathcal{X}_{ui},
\]
\[
B_i(g_i(x_i, w_i)) \leq B_i(x_i) + z_i^\top X_i z_i, \quad \forall x_i \in \mathcal{X}_i, w_i \in \mathcal{W}_i,
\]
where \( z_i = [w_i^\top, x_i^\top]^\top \).

**Theorem 1**: If each subsystem \( \Sigma_i \) has a sub-barrier function \( B_i \), and the following linear matrix inequality (LMI) holds:
\[
\Delta := \begin{bmatrix} M \\ I_p \end{bmatrix} \begin{bmatrix} \text{diag}(X_1^{11}, \ldots, X_\ell^{11}) & \text{diag}(X_1^{12}, \ldots, X_\ell^{12}) \\ \text{diag}(X_1^{21}, \ldots, X_\ell^{21}) & \text{diag}(X_1^{22}, \ldots, X_\ell^{22}) \end{bmatrix} \begin{bmatrix} M \\ I_p \end{bmatrix} \leq 0,
\]
then \( B(x) = \sum_{i=1}^\ell B_i(x_i) \) is a barrier function for \( \Sigma \).

---

## V. Compositional Data-Driven Safety Verification

### A. Computation of Sub-Barrier Functions

Sub-barrier functions are represented as:
\[
B_i(x_i) = \sum_{j=1}^{r_i} q_{ij} p_j(x_i),
\]
where \( p_j: \mathbb{R}^{n_i} \rightarrow \mathbb{R} \) are user-defined basis functions, and \( q_i = (q_{i1}, \ldots, q_{ir_i}) \in \mathbb{R}^{r_i} \).

**Problem 1 (RCP)**: For subsystem \( \Sigma_i \), solve:
\[
\min_{\eta_i, v_i, X_i} \eta_i
\]
\[
\text{s.t.} \max_{j \in \{1,2,3\}} c_j(x_i, w_i, v_i, X_i) \leq \eta_i, \quad \forall x_i \in \mathcal{X}_i, \forall w_i \in \mathcal{W}_i,
\]
\[
v_i := (\gamma_i, q_i), \quad \gamma_i < 1,
\]
where:
\[
c_1(x_i, w_i, v_i, X_i) = (B_i(x_i) - \gamma_i) \mathbb{1}_{\mathcal{X}_{0i}}(x_i),
\]
\[
c_2(x_i, w_i, v_i, X_i) = (-B_i(x_i) + 1) \mathbb{1}_{\mathcal{X}_{ui}}(x_i),
\]
\[
c_3(x_i, w_i, v_i, X_i) = B_i(g_i(x_i, w_i)) - B_i(x_i) - z_i^\top X_i z_i.
\]

Since \( g_i \) is unknown, RCP is approximated by a scenario convex program (SCP) using data \( \mathcal{D}_i = \{(\hat{x}_{il}, \hat{w}_{il}, \hat{x}_{il}^+)\}_{l=1}^{N_i} \).

**Problem 2 (SCP)**: Solve:
\[
\min_{\eta_i, v_i, X_i} \eta_i
\]
\[
\text{s.t.} \max_{j \in \{1,2,3\}} c_j(\hat{x}_{il}, \hat{w}_{il}, v_i, X_i) \leq \eta_i, \quad \forall (\hat{x}_{il}, \hat{w}_{il}, \hat{x}_{il}^+) \in \mathcal{D}_i,
\]
\[
v_i := (\gamma_i, q_i), \quad \gamma_i < 1.
\]

**Assumption 1**: Functions \( c_j \) are locally Lipschitz with constants \( L_j > 0 \), \( j = 1, 2, 3 \).

**Theorem 2**: For subsystem \( \Sigma_i \), if \( N_i \geq N(\bar{\epsilon}_i, \beta_i) \), where:
\[
N(\bar{\epsilon}_i, \beta_i) := \min \left\{ \bar{n} \in \mathbb{N} \mid \sum_{j=0}^{s_i} \binom{\bar{n}}{j} \bar{\epsilon}_i^j (1 - \bar{\epsilon}_i)^{\bar{n}-j} \leq \beta_i \right\},
\]
\[
\bar{\epsilon}_i = \left( \frac{\epsilon_i}{L} \right)^{s_i}, \quad s_i = r_i + 1 + (p_i + n_i)^2, \quad L = \max \{ L_1, L_2, L_3 \},
\]
and \( \eta_{\text{SCP}-\text{i}}^* + \epsilon_i \leq 0 \), then \( B_i \) satisfies sub-barrier conditions with confidence at least \( 1 - \beta_i \).

**Theorem 3**: If all subsystems satisfy Theorem 2 and matrices \( X_i \) satisfy the LMI in Theorem 1, then \( B(x) = \sum_{i=1}^\ell B_i(x_i) \) is a barrier function for \( \Sigma \) with confidence at least \( 1 - \sum_{i=1}^\ell \beta_i \).

### B. Computation of the Overall Barrier Function

To enforce the LMI, the ADMM algorithm is used with local constraints:
\[
\mathcal{L}_i := \left\{ (\eta_i, v_i, X_i) : \max_{j \in \{1,2,3\}} c_j(\hat{x}_{il}, \hat{w}_{il}, v_i, X_i) \leq \eta_i, \forall (\hat{x}_{il}, \hat{w}_{il}, \hat{x}_{il}^+) \in \mathcal{D}_i \right\},
\]
and global constraint:
\[
\mathcal{G} := \left\{ (X_1, \ldots, X_\ell) : \Delta \leq 0 \right\}.
\]

**Problem 3**: Solve:
\[
\min_{\eta_i, v_i, X_i, Z_i} \sum_{i=1}^\ell \left( \eta_i + \mathbb{1}_{\mathcal{L}_i}(\eta_i, v_i, X_i) \right) + \mathbb{1}_{\mathcal{G}}(Z_1, \ldots, Z_\ell)
\]
\[
\text{s.t.} X_i - Z_i = 0.
\]

**Algorithm 1: Compositional Data-Driven Safety Verification**
- **Input**: \( \beta_i \), initial \( Z_i^0 \), \( \Lambda_i^0 \), Lipschitz constant \( L \), interconnection matrix \( M \).
- **Output**: \( \eta_i \), \( v_i \), \( X_i \) for \( i = 1, \ldots, \ell \).
- **Steps**:
  1. Choose \( \epsilon_i \leq L \).
  2. Compute \( N_i \geq N(\bar{\epsilon}_i, \beta_i) \).
  3. For each subsystem, solve local problem:
     \[
     (\eta_i^{k+1}, v_i^{k+1}, X_i^{k+1}) \in \underset{\eta_i^*, v_i^*, X_i^* \in \mathcal{L}_i}{\text{argmin}} \eta_i^* + \|X_i^* - Z_i^k + \Lambda_i^k\|_F^2.
     \]
  4. If \( (X_1^{k+1}, \ldots, X_\ell^{k+1}) \notin \mathcal{G} \), solve global problem:
     \[
     (Z_1^{k+1}, \ldots, Z_\ell^{k+1}) \in \underset{Z_1^*, \ldots, Z_\ell^* \in \mathcal{G}}{\text{argmin}} \sum_{i=1}^\ell \|X_i^{k+1} - Z_i^* + \Lambda_i^k\|_F^2.
     \]
  5. Update dual variables: \( \Lambda_i^{k+1} = X_i^{k+1} - Z_i^{k+1} + \Lambda_i^k \).
  6. Repeat until convergence.

---

## VI. Example: Room Temperature Control

The approach is applied to a 100-room circular building with dynamics:
\[
x(k+1) = A x(k) + \alpha_e T_E + \alpha_h T_h u(k),
\]
where \( A \) is a circulant matrix, \( x \in \mathbb{R}^{100} \) is the state (room temperatures), \( u \in \mathbb{R}^{100} \) is the control input, \( T_E = 15^\circ \text{C} \), \( T_h = 55^\circ \text{C} \), \( \alpha = 5 \times 10^{-2} \), \( \alpha_e = 8 \times 10^{-3} \), \( \alpha_h = 3.6 \times 10^{-3} \). The controller is \( u_i(x_i) = -0.002398 x_i + 0.5357 \).

**Sets**:
- State set: \( \mathcal{X} = [19, 28]^{100} \).
- Initial set: \( \mathcal{X}_0 = [20.5, 22.5]^{100} \).
- Unsafe set: \( \mathcal{X}_u = [24, 28]^{100} \).

**Subsystem Dynamics**:
\[
x_i(k+1) = a x_i(k) + \alpha_e T_{E_i} + 0.5357 \alpha_h T_h + \alpha w_i(k),
\]
where \( a = 1 - 2\alpha - \alpha_e + \alpha_h (0.002398 x_i - 0.5357) - 0.002398 \alpha_h^2 T_h \), and \( w_i \) represents neighboring room temperatures.

**Parameters**:
- Basis functions: \( p_j(x_i) = x_i^{j-1} \), \( j = 1, 2, 3 \).
- Sub-barrier: \( B_i(x_i) = \sum_{j=1}^3 q_{ij} x_i^{j-1} \).
- Confidence: \( \beta_i = 10^{-3} \).
- Lipschitz constant: \( L = 468 \), computed assuming subsystem Lipschitz constant of 2 and \( |q_i|_\infty \leq 15 \).
- \( \epsilon_i = 1 \), \( N_i = 17474 \).

**Results**:
- Sub-barrier functions: \( B_i(x_i) = 0.6466 x_i^2 + 14.81 x_i + 14.5 \).
- \( \eta_{\text{SCP}-\text{i}}^* = -10 \), \( \gamma_i = 0.95 \).
- ADMM converges in 6 steps, 380 seconds on an iMac (3.5 GHz Quad-Core Intel Core i7, 32 GB RAM).
- LMI \( \Delta \leq 0 \) holds with largest eigenvalue \(-0.4681\).
- Overall barrier function \( B(x) = \sum_{i=1}^{100} B_i(x_i) \) satisfies safety conditions with confidence \( 0.9 \).

**Comparison**: Table I shows the exponential data growth in [16] versus linear complexity in this approach, making it scalable for large networks.

---

## VII. Figures

- **Fig. 1**: Sub-barrier function \( B_i \) with initial (\( [20.5, 22.5] \)) and unsafe (\( [24, 28] \)) sets.
- **Fig. 2**: Decay condition for \( B_{15} \), with \( X_{15} = \begin{bmatrix} 0.0627 & 0.0445 \\ -0.0105 & -0.1264 \end{bmatrix} \).

---

## VIII. References

1. H. Ravanbakhsh and S. Sankaranarayanan, “Learning control Lyapunov functions from counterexamples and demonstrations,” Auton. Robots, vol. 43, no. 2, pp. 275–307, 2019.
2. S. Sadraddini and C. Belta, “Formal guarantees in data-driven model identification and control synthesis,” in Proc. 21st Int. Conf. Hybrid Syst., 2018.
3. [Additional references as cited in the original paper].