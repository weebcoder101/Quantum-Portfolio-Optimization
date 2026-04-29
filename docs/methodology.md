```markdown
# Methodology

## Overview

This document describes the four-phase methodology used to solve the **constrained portfolio optimization problem** using both classical and quantum approaches. The pipeline progresses from raw market data to a fully benchmarked Quantum Approximate Optimization Algorithm (QAOA) implementation, with each phase building directly on the outputs of the previous one.

The core research question: **Can a hybrid quantum-classical optimizer (QAOA) match or outperform classical methods for selecting an optimal subset of assets under real-world constraints?**

---

## Phase 1: Data Preparation & QUBO Formulation

### Objective
Transform raw financial market data into a mathematically rigorous optimization problem suitable for both classical solvers and quantum hardware.

### Approach

**1.1 Asset Universe Definition**

We selected an 8-stock universe spanning multiple market sectors to ensure diversification is a meaningful constraint:

| Ticker | Sector |
|--------|--------|
| NVDA | Technology |
| MSFT | Technology |
| LLY | Healthcare |
| XOM | Energy |
| JNJ | Healthcare |
| PG | Consumer Staples |
| JPM | Financials |
| BRK-B | Financials |

Historical price data was retrieved using the `yfinance` API, and daily log-returns were computed over a trailing window.

**1.2 Covariance Estimation**

An 8×8 sample covariance matrix **Σ** was estimated from the historical returns. This matrix captures how each pair of assets co-moves — the foundation of portfolio risk measurement under Modern Portfolio Theory (Markowitz, 1952).

**1.3 Multi-Objective Cost Function**

The optimization balances three competing objectives:

- **Maximize expected return**: Σᵢ xᵢμᵢ (where μᵢ is the mean return of asset i)
- **Minimize portfolio risk**: x^T Σ x (quadratic form over the covariance matrix)
- **Maximize diversification**: Penalize sector concentration

These are combined via scalarization into a single cost function:

```
C(x) = -α·Σᵢ xᵢμᵢ + β·Σᵢⱼ xᵢΣᵢⱼxⱼ - γ·D(x)
```

Where α, β, γ are tunable weights controlling the trade-off between return, risk, and diversification.

**1.4 Constraint Encoding**

Three hard constraints were encoded as quadratic penalty terms added to the cost function:

| Constraint | Description | Penalty Form |
|------------|-------------|--------------|
| **Cardinality** | Select exactly K=4 assets | P₁·(Σᵢ xᵢ - K)² |
| **Budget** | Total normalized cost ≤ B' = 1.638 | P₂·(Σᵢ cᵢxᵢ - B)² (with slack variables) |
| **Sector Cap** | ≤2 assets from any single sector | P₃·Σₛ max(0, Σᵢ∈ₛ xᵢ - 2)² |

**1.5 QUBO Construction**

With binary decision variables xᵢ ∈ {0, 1} (include asset i or not), the combined cost + penalties form a **Quadratic Unconstrained Binary Optimization (QUBO)** problem:

```
min x^T Q x + c^T x
```

The budget constraint required 5 slack variables (binary encoding), expanding the problem to **13 binary variables** (8 decision + 5 slack), yielding a 13×13 QUBO matrix.

**1.6 RIT Penalty Calibration**

Penalty coefficients were derived using information-theoretic principles rather than arbitrary tuning:

- **α₁ (cardinality)** = 0.036058 — computed from a sufficiency floor condition × 1.05 safety margin
- **α₂ (budget)** = 0.007356 — derived from a peer condition ensuring the penalty dominates any constraint-violating gain

This ensures the penalty landscape reliably steers optimizers away from infeasible solutions without overwhelming the objective signal.

### Key Output
- 8×8 covariance matrix
- 13×13 QUBO matrix Q
- Calibrated penalty coefficients

### Tools
`Python`, `NumPy`, `pandas`, `yfinance`

---

## Phase 2: Classical Baseline Implementation

### Objective
Establish exact and heuristic classical solutions as benchmarks against which the quantum approach is evaluated.

### Approach

**2.1 Brute-Force Enumeration**

With 13 binary variables, the full solution space contains 2¹³ = 8,192 possible portfolios. This is small enough to enumerate exhaustively:

- Every bitstring x ∈ {0,1}¹³ was evaluated against the QUBO cost function
- Infeasible solutions (violating constraints) were identified by their high penalty costs
- The global optimum was identified with certainty

This provides the **ground truth** — the provably best solution.

**2.2 Greedy Heuristic**

A sequential greedy algorithm was implemented as a fast approximation:

1. Start with an empty portfolio
2. At each step, add the asset that most improves the cost function
3. Continue until K=4 assets are selected
4. Check constraint feasibility

This tests whether a simple, fast heuristic can match the exact solution — relevant for evaluating whether the problem actually *needs* sophisticated methods.

**2.3 Continuous Markowitz Relaxation**

For comparison, a continuous relaxation (wᵢ ∈ [0,1] instead of {0,1}) was solved using `scipy.optimize.minimize`. This represents the textbook Markowitz efficient frontier approach and provides a lower bound on the achievable risk for a given return target.

### Key Output
- **Optimal portfolio**: {NVDA, JNJ, PG, JPM} with objective value -0.018961
- Classical runtime benchmarks (brute force: ~0.1s, greedy: <0.1s)
- Efficient frontier reference curve

### Tools
`Python`, `NumPy`, `SciPy`

---

## Phase 3: QUBO Validation & Ising Mapping

### Objective
Verify the correctness of the QUBO formulation and prepare it for quantum execution by mapping to an Ising Hamiltonian.

### Approach

**3.1 QUBO Matrix Validation**

Before feeding the QUBO to a quantum algorithm, we validated:

- **Symmetry check**: Q = Q^T (required for valid QUBO)
- **Ground state verification**: The brute-force optimal bitstring from Phase 2 was confirmed to correspond to the minimum eigenvalue of the QUBO matrix
- **Penalty effectiveness**: Verified that all constraint-violating solutions have strictly higher cost than the best feasible solution

**3.2 Ising Hamiltonian Mapping**

Quantum hardware operates on spin variables sᵢ ∈ {-1, +1}, not binary bits. The standard conversion:

```
xᵢ = (1 + sᵢ) / 2
```

was applied to transform the QUBO into an equivalent Ising Hamiltonian:

```
H = Σᵢⱼ Jᵢⱼ ZᵢZⱼ + Σᵢ hᵢ Zᵢ + offset
```

Where Zᵢ are Pauli-Z operators, Jᵢⱼ are coupling strengths, and hᵢ are local fields — all derived algebraically from the QUBO matrix Q.

**3.3 Spectral Analysis**

The Ising Hamiltonian was diagonalized to confirm:

- **Ground state energy**: E* = -0.6156
- **Energy gap**: 7.24×10⁻⁵ (gap between ground state and first excited state)
- **Degeneracy structure**: Number of near-degenerate low-energy states

The small energy gap is a critical finding — it means the quantum optimizer must achieve very high precision to distinguish the optimal solution from near-optimal competitors.

### Key Output
- Validated 13×13 QUBO matrix
- Ising Hamiltonian with 91 interaction terms (fully connected)
- Ground state energy and spectral gap characterization

### Tools
`Python`, `NumPy`, `Qiskit`

---

## Phase 4: QAOA Hybrid Quantum-Classical Optimization

### Objective
Implement the Quantum Approximate Optimization Algorithm (QAOA) to solve the portfolio selection problem and benchmark it against the classical baselines from Phase 2.

### Approach

**4.1 QAOA Circuit Architecture**

The QAOA ansatz at circuit depth p consists of:

1. **Initialization**: All 13 qubits prepared in the |+⟩ state (equal superposition)
2. **Cost layer** U_C(γ): Applies the Ising Hamiltonian as a unitary evolution — encodes the portfolio cost function into quantum phases
3. **Mixer layer** U_M(β): Applies Pauli-X rotations to enable exploration of the solution space
4. **Repetition**: Steps 2-3 are repeated p times, with independent parameters (γₗ, βₗ) for each layer

```
|ψ(γ,β)⟩ = [U_M(βₚ)·U_C(γₚ)] × ... × [U_M(β₁)·U_C(γ₁)] |+⟩^⊗13
```

Experiments were run at depths p = 1, 2, 3, 5, and 8 to study convergence behavior.

**4.2 CVaR Objective Function**

Instead of optimizing the standard expectation value ⟨C⟩, we used the **Conditional Value-at-Risk (CVaR)** objective at the α = 0.10 quantile. CVaR focuses the optimizer on the *best 10% of measurement outcomes*, providing:

- More robust convergence in the presence of noise
- Stronger concentration on low-energy (high-quality) solutions
- Reduced sensitivity to the "tail" of poor measurement results

This follows the approach of Barkoutsos et al., Quantum 4, 256 (2020).

**4.3 Classical Optimization Loop**

The 2p variational parameters (γ₁...γₚ, β₁...βₚ) were optimized using:

- **Optimizer**: Powell's conjugate direction method
- **Multi-start strategy**: 50 random restarts to mitigate local minima
- **INTERP warm-start**: Parameters from depth p were interpolated to initialize depth p+1, accelerating convergence at higher depths

Each optimization iteration:
1. Constructs the QAOA circuit with current parameters
2. Simulates measurement (8,192 shots on Qiskit Aer statevector simulator)
3. Computes CVaR₀.₁₀ from the measurement distribution
4. Feeds the cost back to Powell optimizer
5. Repeats until convergence

**4.4 Solution Extraction**

From the final optimized circuit, solutions were extracted using two complementary methods:

- **Max-probability**: The bitstring with highest measurement frequency
- **Min-energy in support**: The bitstring with lowest cost among all observed measurements

This dual extraction increases the chance of recovering the true optimum, especially when the probability distribution is spread across near-degenerate states.

**4.5 Comparative Analysis**

Results were benchmarked against Phase 2 baselines across multiple dimensions:

| Metric | What It Measures |
|--------|-----------------|
| **Solution quality** | Does QAOA find the same optimal portfolio? |
| **Objective value** | How close is the QAOA cost to the classical optimum? |
| **Runtime** | Wall-clock time for classical vs. quantum simulation |
| **Probability concentration** | P(ground state) at each circuit depth |
| **Convergence rate** | Cost vs. iteration for the classical optimization loop |

**4.6 Quantum Advantage Assessment**

A critical and honest evaluation was performed:

- At n=13 qubits, **no quantum advantage is observed** — classical brute force solves the problem in 0.1 seconds
- QAOA at low depth (p=1,2) fails to concentrate probability on the optimal solution
- At higher depths (p=5,8), QAOA recovers the correct portfolio but at significantly higher computational cost
- The small energy gap (7.24×10⁻⁵) makes the problem particularly challenging for variational quantum methods

These findings are consistent with the current understanding of NISQ-era limitations and do not diminish the value of the methodology — they demonstrate intellectual honesty about where quantum methods currently stand.

### Key Output
- QAOA convergence data across depths p = 1, 2, 3, 5, 8
- Classical vs. quantum comparison tables
- Probability concentration metrics
- Honest quantum advantage assessment

### Tools
`Python`, `Qiskit 1.0+`, `qiskit-algorithms`, `qiskit-optimization`, `Qiskit Aer` (statevector simulator), `matplotlib`

---

## Summary of Phase Dependencies

```
Phase 1 (Data & QUBO)
    │
    ├──► Phase 2 (Classical Baselines) ──────┐
    │                                         │
    └──► Phase 3 (Validation & Ising Map)     ├──► Comparative Analysis
              │                               │
              └──► Phase 4 (QAOA) ────────────┘
```

Each phase produces artifacts consumed by downstream phases, ensuring a traceable and reproducible pipeline from raw data to final benchmarked results.

---

## References

1. Markowitz, H. (1952). "Portfolio Selection." *The Journal of Finance*, 7(1), 77-91.
2. Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm." arXiv:1411.4028.
3. Barkoutsos, P. K., et al. (2020). "Improving Variational Quantum Optimization using CVaR." *Quantum*, 4, 256.
4. Lucas, A. (2014). "Ising formulations of many NP problems." *Frontiers in Physics*, 2, 5.
```

