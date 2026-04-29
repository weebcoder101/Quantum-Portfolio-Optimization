"""
Phase 3 — QUBO Matrix Validation & Ground-State Verification
Constrained Combinatorial Portfolio Selection
RIT Information-Theoretic Penalty Framework

CORRECTED VERSION (Convention Fix + QC Safety Margin):
  1. Symmetric QUBO convention: off-diagonal Q[i,j] = half the upper-triangular
     value, because z^T Q z inherently doubles off-diagonal entries.
  2. Penalty coefficients: α_c = max(peer_condition, sufficiency_floor)
  3. Structure norms: recomputed for the symmetric convention.
  4. QC safety margin: α₁ *= 1.05 to widen the 1.76×10⁻⁶ energy gap
     for noise-resilient execution on real quantum hardware.
  5. phase1_obj reference corrected to -0.018961.
"""

import numpy as np
from itertools import combinations
from typing import NamedTuple, List, Tuple
import sys
import os

# ============================================================
# OUTPUT FILE SETUP
# ============================================================

OUTPUT_FILENAME = "phase3.12_output.txt"
_output_file = None


def tee_print(*args, **kwargs):
    """Print to both stdout and the output file."""
    global _output_file
    print(*args, **kwargs)
    if _output_file is not None:
        kwargs.pop('file', None)
        print(*args, file=_output_file, **kwargs)
        _output_file.flush()


# ============================================================
# Problem Data (exact values from Phase 1)
# ============================================================

TICKERS = ["NVDA", "MSFT", "LLY", "XOM", "JNJ", "PG", "JPM", "BRK-B"]
N_ASSETS = 8
K = 4
LAMBDA_R = 0.341503

C_PRIME = np.array([
    0.24947, 0.48927, 1.00000, 0.17067,
    0.25953, 0.17092, 0.35891, 0.54454
])

B_PRIME = 1.638036

MU = np.array([
    0.69074, 0.11672, 0.27136, 0.10990,
    0.13740, 0.00855, 0.29607, 0.12231
])

SIGMA = np.array([
    [ 0.23358,  0.05762,  0.03029, -0.00295, -0.02060, -0.01361,  0.02952,  0.00432],
    [ 0.05762,  0.05502,  0.01078, -0.00270, -0.00482, -0.00153,  0.01446,  0.00621],
    [ 0.03029,  0.01078,  0.12318, -0.00071,  0.01085,  0.01121,  0.01273,  0.01151],
    [-0.00295, -0.00270, -0.00071,  0.05232,  0.00538,  0.00386,  0.01381,  0.01065],
    [-0.02060, -0.00482,  0.01085,  0.00538,  0.02923,  0.01077,  0.00434,  0.00808],
    [-0.01361, -0.00153,  0.01121,  0.00386,  0.01077,  0.02896,  0.00188,  0.00810],
    [ 0.02952,  0.01446,  0.01273,  0.01381,  0.00434,  0.00188,  0.05151,  0.01927],
    [ 0.00432,  0.00621,  0.01151,  0.01065,  0.00808,  0.00810,  0.01927,  0.02487],
])

# Slack variable configuration
DELTA = 0.1
M_SLACK = 5
N_TOTAL = N_ASSETS + M_SLACK  # 13 qubits

W_SLACK = np.array([DELTA * (2 ** k) for k in range(M_SLACK)])

SECTORS = {
    "Technology":       {0, 1},
    "Financials":       {6, 7},
    "Healthcare":       {2, 4},
    "Consumer Staples": {5},
    "Energy":           {3},
}

M_S = 2

# ============================================================
# QC DEPLOYMENT CONFIGURATION
# ============================================================
# Safety margin multiplier for α₁ (cardinality penalty).
# The theoretical sufficiency floor yields an energy gap of
# only 1.76×10⁻⁶ between ground and first interloper (k=3).
# On noisy quantum hardware this is dangerously tight.
# A 5% uplift widens the gap to ~0.002, providing
# resilience against gate noise, readout error, and
# finite-shot sampling variance.
QC_SAFETY_MARGIN = 1.05  # Set to 1.00 for exact theoretical match

# ============================================================
# RIT Information Costs (from Phase 1)
# ============================================================
I_1 = 1.870717
I_2 = 0.977632
I_TOTAL = I_1 + I_2


# ============================================================
# Penalty Coefficient Derivation
# ============================================================

def compute_objective_matrix():
    """Build the 8×8 objective-only QUBO block (no penalties)."""
    H_obj = np.zeros((N_ASSETS, N_ASSETS))
    for i in range(N_ASSETS):
        H_obj[i, i] = SIGMA[i, i] - LAMBDA_R * MU[i]
        for j in range(i + 1, N_ASSETS):
            H_obj[i, j] = SIGMA[i, j]
            H_obj[j, i] = SIGMA[i, j]
    return H_obj


def compute_structure_norms():
    """
    Compute ||V_c^QUBO||_F for each constraint's penalty structure
    matrix in the SYMMETRIC convention.

    For the symmetric QUBO where E = z^T Q z:
      Off-diagonal entries are HALF the upper-triangular values.
      The Frobenius norm sums ALL entries (i,j), so off-diagonal
      entries appear twice (once at (i,j) and once at (j,i)).

    Cardinality structure V_1:
      Diagonal: (1-2K) per asset
      Off-diagonal (symmetric): 1 per pair (not 2)
      ||V_1||_F = sqrt(n*(1-2K)^2 + n*(n-1)*1^2)

    Budget structure V_2:
      Diagonal entry j: a_j(a_j - 2B')
      Off-diagonal (i,j): a_i * a_j (not 2*a_i*a_j)
    """
    # Cardinality
    V1_diag_sq = N_ASSETS * (1 - 2 * K) ** 2          # 8 * 49 = 392
    V1_offdiag_sq = N_ASSETS * (N_ASSETS - 1) * 1**2  # 56 * 1 = 56
    norm_V1 = np.sqrt(V1_diag_sq + V1_offdiag_sq)     # sqrt(448) ≈ 21.166

    # Budget
    a = np.concatenate([C_PRIME, W_SLACK])
    diag_entries = a * (a - 2 * B_PRIME)
    diag_sq_sum = np.sum(diag_entries ** 2)

    # Off-diagonal (i,j) for i≠j: entry = a_i * a_j
    # Σ_{i≠j} (a_i a_j)^2 = (Σ a_i^2)^2 - Σ a_i^4
    A_sum_sq = np.sum(a ** 2)
    A_sum_4 = np.sum(a ** 4)
    offdiag_sq_sum = A_sum_sq ** 2 - A_sum_4

    norm_V2 = np.sqrt(diag_sq_sum + offdiag_sq_sum)

    return norm_V1, norm_V2


def compute_natural_objective(x):
    """Compute f(x) = x^T Σ x - λ_R μ^T x for a binary vector x."""
    x = np.array(x, dtype=float)
    return float(x @ SIGMA @ x - LAMBDA_R * MU @ x)


def compute_sufficiency_floors():
    """
    Compute the sufficiency floor for each constraint.

    For the cardinality constraint:
      α₁_suff = max over k≠K of [f*(K) - f*(k)] / (k - K)²
      where f*(k) = best natural objective at cardinality k.

    For the budget constraint:
      α₂_suff ensures no budget-violating, cardinality-feasible
      portfolio has lower total energy than the best fully-feasible one.
    """
    # Best objective for each cardinality level
    best_obj_by_k = {}
    best_portfolio_by_k = {}

    for k in range(N_ASSETS + 1):
        if k == 0:
            best_obj_by_k[0] = 0.0
            best_portfolio_by_k[0] = ()
            continue

        best_obj = float('inf')
        best_port = None
        for combo in combinations(range(N_ASSETS), k):
            x = np.zeros(N_ASSETS)
            for idx in combo:
                x[idx] = 1.0
            obj = compute_natural_objective(x)
            if obj < best_obj:
                best_obj = obj
                best_port = combo
        best_obj_by_k[k] = best_obj
        best_portfolio_by_k[k] = tuple(TICKERS[i] for i in best_port)

    # Cardinality sufficiency floor
    f_star = best_obj_by_k[K]
    alpha1_suff = 0.0
    binding_k = K

    for k in range(N_ASSETS + 1):
        if k == K:
            continue
        violation_sq = (k - K) ** 2
        obj_profit = f_star - best_obj_by_k[k]
        if obj_profit > 0:
            required = obj_profit / violation_sq
            if required > alpha1_suff:
                alpha1_suff = required
                binding_k = k

    # Budget sufficiency floor
    best_feasible_obj = float('inf')
    best_violating_obj = float('inf')
    best_violating_cost = 0.0

    for combo in combinations(range(N_ASSETS), K):
        x = np.zeros(N_ASSETS)
        for idx in combo:
            x[idx] = 1.0
        obj = compute_natural_objective(x)
        cost = float(C_PRIME @ x)

        if cost <= B_PRIME + 1e-9:
            if obj < best_feasible_obj:
                best_feasible_obj = obj
        else:
            if obj < best_violating_obj:
                best_violating_obj = obj
                best_violating_cost = cost

    if best_violating_obj < best_feasible_obj:
        min_budget_violation_sq = (best_violating_cost - B_PRIME) ** 2
        budget_profit = best_feasible_obj - best_violating_obj
        alpha2_suff = budget_profit / min_budget_violation_sq
    else:
        alpha2_suff = 0.0

    return (alpha1_suff, binding_k, best_obj_by_k, best_portfolio_by_k,
            alpha2_suff, best_feasible_obj, best_violating_obj)


def derive_penalties():
    """
    Derive final penalty coefficients:
    α_c = max(peer_condition, sufficiency_floor)

    Peer condition (conditioning guarantee):
      α_c^peer = (I_c/I_total) × ||H_obj||_F / ||V_c||_F

    Sufficiency floor (feasibility guarantee):
      α_c^suff = max violation profit / violation²
    """
    H_obj = compute_objective_matrix()
    norm_H_obj = np.linalg.norm(H_obj, 'fro')

    norm_V1, norm_V2 = compute_structure_norms()

    alpha1_peer = (I_1 / I_TOTAL) * (norm_H_obj / norm_V1)
    alpha2_peer = (I_2 / I_TOTAL) * (norm_H_obj / norm_V2)

    (alpha1_suff, binding_k, best_obj_by_k, best_portfolio_by_k,
     alpha2_suff, best_feas_obj, best_viol_obj) = compute_sufficiency_floors()

    alpha1_final = max(alpha1_peer, alpha1_suff)
    alpha2_final = max(alpha2_peer, alpha2_suff)

    return {
        'norm_H_obj': norm_H_obj,
        'norm_V1': norm_V1,
        'norm_V2': norm_V2,
        'alpha1_peer': alpha1_peer,
        'alpha2_peer': alpha2_peer,
        'alpha1_suff': alpha1_suff,
        'alpha2_suff': alpha2_suff,
        'alpha1_final': alpha1_final,
        'alpha2_final': alpha2_final,
        'binding_k': binding_k,
        'best_obj_by_k': best_obj_by_k,
        'best_portfolio_by_k': best_portfolio_by_k,
        'best_feas_obj': best_feas_obj,
        'best_viol_obj': best_viol_obj,
    }


# ============================================================
# QUBO Matrix Construction — SYMMETRIC CONVENTION
# ============================================================

def build_qubo_matrix(alpha1: float, alpha2: float) -> np.ndarray:
    """
    Construct the full 13×13 QUBO matrix Q (SYMMETRIC convention).

    Energy: E(z) = z^T Q z  where z ∈ {0,1}^13

    ═══════════════════════════════════════════════════════════
    CRITICAL CONVENTION: Q is symmetric. The evaluation z^T Q z
    inherently doubles off-diagonal entries:
      z^T Q z = Σ Q[i,i] z_i + 2 Σ_{i<j} Q[i,j] z_i z_j

    Therefore off-diagonal entries store HALF the interaction:
      Q[i,j] = Σ[i,j] + α₁ + α₂ c'_i c'_j    (NOT 2α₁, NOT 2α₂)
    ═══════════════════════════════════════════════════════════
    """
    Q = np.zeros((N_TOTAL, N_TOTAL))

    # --- Decision block (8×8) ---
    for i in range(N_ASSETS):
        # Diagonal: (1-2K) from expanding (Σx_i)² - 2K(Σx_i)
        # The x_i² = x_i identity means diagonal absorbs linear terms.
        Q[i, i] = (SIGMA[i, i]
                    - LAMBDA_R * MU[i]
                    + alpha1 * (1 - 2 * K)
                    + alpha2 * C_PRIME[i] * (C_PRIME[i] - 2 * B_PRIME))

        for j in range(i + 1, N_ASSETS):
            # Off-diagonal: symmetric convention, no factor of 2
            val = (SIGMA[i, j]
                   + alpha1
                   + alpha2 * C_PRIME[i] * C_PRIME[j])
            Q[i, j] = val
            Q[j, i] = val

    # --- Slack diagonal ---
    for k in range(M_SLACK):
        idx = N_ASSETS + k
        Q[idx, idx] = alpha2 * W_SLACK[k] * (W_SLACK[k] - 2 * B_PRIME)

    # --- Slack–slack coupling ---
    for k in range(M_SLACK):
        for l in range(k + 1, M_SLACK):
            idx_k = N_ASSETS + k
            idx_l = N_ASSETS + l
            val = alpha2 * W_SLACK[k] * W_SLACK[l]
            Q[idx_k, idx_l] = val
            Q[idx_l, idx_k] = val

    # --- Decision–slack coupling ---
    for i in range(N_ASSETS):
        for k in range(M_SLACK):
            idx_k = N_ASSETS + k
            val = alpha2 * C_PRIME[i] * W_SLACK[k]
            Q[i, idx_k] = val
            Q[idx_k, i] = val

    return Q


def verify_qubo_energy(Q, z, alpha1, alpha2):
    """
    Independent verification: compute E from penalties directly
    and compare to z^T Q z.

    E_full = f_obj + α₁(Σx-K)² + α₂(budget_residual)²
    z^T Q z = E_full - α₁K² - α₂B'²  (constants dropped from QUBO)
    """
    x = z[:N_ASSETS]
    s = z[N_ASSETS:]

    # Objective
    f_obj = float(x @ SIGMA @ x - LAMBDA_R * MU @ x)

    # Cardinality
    card_violation = float(x.sum()) - K
    card_penalty = alpha1 * card_violation ** 2

    # Budget
    budget_expr = float(C_PRIME @ x + W_SLACK @ s - B_PRIME)
    budget_penalty = alpha2 * budget_expr ** 2

    E_full = f_obj + card_penalty + budget_penalty
    E_qubo = float(z @ Q @ z)

    # Constants dropped from QUBO
    const_offset = alpha1 * K ** 2 + alpha2 * B_PRIME ** 2
    E_expected = E_full - const_offset

    return {
        'f_obj': f_obj,
        'card_violation': card_violation,
        'card_penalty': card_penalty,
        'budget_expr': budget_expr,
        'budget_penalty': budget_penalty,
        'E_full': E_full,
        'E_qubo': E_qubo,
        'E_expected': E_expected,
        'const_offset': const_offset,
        'error': abs(E_qubo - E_expected),
    }


# ============================================================
# Exhaustive Enumeration
# ============================================================

class QUBOState(NamedTuple):
    bitstring: tuple
    decision: tuple
    slack: tuple
    energy: float
    portfolio: tuple
    n_selected: int
    cost: float
    slack_value: float
    budget_residual: float
    card_penalty: float
    budget_penalty: float
    obj_natural: float


def enumerate_all_states(Q: np.ndarray) -> List[QUBOState]:
    """Enumerate all 2^13 states and compute QUBO energy + diagnostics."""
    states = []

    for bits in range(2 ** N_TOTAL):
        z = np.array([(bits >> i) & 1 for i in range(N_TOTAL)], dtype=float)

        x = z[:N_ASSETS]
        s = z[N_ASSETS:]

        energy = float(z @ Q @ z)

        indices = tuple(i for i in range(N_ASSETS) if x[i] > 0.5)
        tickers = tuple(TICKERS[i] for i in indices)
        n_selected = int(x.sum())
        cost = float(C_PRIME @ x)
        slack_value = float(W_SLACK @ s)
        budget_residual = cost + slack_value - B_PRIME

        card_penalty = (n_selected - K) ** 2
        budget_penalty = budget_residual ** 2

        obj_natural = float(x @ SIGMA @ x - LAMBDA_R * MU @ x)

        states.append(QUBOState(
            bitstring=tuple(int(zi) for zi in z),
            decision=tuple(int(xi) for xi in x),
            slack=tuple(int(si) for si in s),
            energy=energy,
            portfolio=tickers,
            n_selected=n_selected,
            cost=cost,
            slack_value=slack_value,
            budget_residual=budget_residual,
            card_penalty=card_penalty,
            budget_penalty=budget_penalty,
            obj_natural=obj_natural,
        ))

    states.sort(key=lambda st: st.energy)
    return states


# ============================================================
# Ising Hamiltonian Mapping
# ============================================================

def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert symmetric QUBO matrix Q to Ising Hamiltonian.

    H_Ising = Σ_{i<j} J_ij s_i s_j + Σ_i h_i s_i + E_0
    where s_i ∈ {-1, +1}, x_i = (1 - s_i) / 2
    """
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)

    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = Q[i, j] / 2.0

    for i in range(n):
        row_sum = sum(Q[i, j] for j in range(n) if j != i)
        h[i] = -0.5 * (Q[i, i] + row_sum)

    diag_sum = np.trace(Q)
    off_diag_sum = Q.sum() - diag_sum
    E_0 = 0.5 * (diag_sum + 0.5 * off_diag_sum)

    return J, h, E_0


def verify_ising_mapping(Q, J, h, E_0, n_samples=100):
    """Verify Ising mapping against QUBO for random states."""
    np.random.seed(42)
    max_error = 0.0

    for _ in range(n_samples):
        x = np.random.randint(0, 2, size=N_TOTAL).astype(float)
        s_ising = 1 - 2 * x

        E_qubo = float(x @ Q @ x)

        E_ising = E_0
        for i in range(N_TOTAL):
            E_ising += h[i] * s_ising[i]
            for j in range(i + 1, N_TOTAL):
                E_ising += J[i, j] * s_ising[i] * s_ising[j]

        error = abs(E_qubo - E_ising)
        max_error = max(max_error, error)

    return max_error


# ============================================================
# Spectral Analysis
# ============================================================

def spectral_analysis(Q):
    eigvals_full = np.sort(np.linalg.eigvalsh(Q))
    abs_eigvals = np.abs(eigvals_full)
    kappa_full = abs_eigvals.max() / abs_eigvals[abs_eigvals > 1e-12].min()

    Q_dec = Q[:N_ASSETS, :N_ASSETS]
    eigvals_dec = np.sort(np.linalg.eigvalsh(Q_dec))
    abs_eigvals_dec = np.abs(eigvals_dec)
    kappa_dec = abs_eigvals_dec.max() / abs_eigvals_dec[abs_eigvals_dec > 1e-12].min()

    return {
        'eigvals_full': eigvals_full,
        'kappa_full': kappa_full,
        'eigvals_dec': eigvals_dec,
        'kappa_dec': kappa_dec,
        'frob_full': np.linalg.norm(Q, 'fro'),
        'frob_dec': np.linalg.norm(Q_dec, 'fro'),
    }


# ============================================================
# Main Execution
# ============================================================

def main():
    global _output_file

    _output_file = open(OUTPUT_FILENAME, 'w', encoding='utf-8')

    try:
        tee_print("=" * 80)
        tee_print("PHASE 3 — QUBO MATRIX VALIDATION & GROUND-STATE VERIFICATION")
        tee_print("  RIT Information-Theoretic Penalty Framework")
        tee_print("  Convention: SYMMETRIC Q, E = z^T Q z")
        if QC_SAFETY_MARGIN != 1.0:
            tee_print(f"  ⚡ QC DEPLOYMENT MODE: α₁ safety margin = {QC_SAFETY_MARGIN:.2f}×")
        tee_print("=" * 80)
        tee_print(f"\n  Output file: {os.path.abspath(OUTPUT_FILENAME)}")

        # ===========================================================
        # Step 0: Derive penalty coefficients
        # ===========================================================
        tee_print("\n" + "─" * 60)
        tee_print("STEP 0: PENALTY COEFFICIENT DERIVATION")
        tee_print("─" * 60)

        pen = derive_penalties()

        tee_print(f"\n  ── Objective Matrix ──")
        tee_print(f"    ||H_obj||_F = {pen['norm_H_obj']:.6f}")

        tee_print(f"\n  ── Penalty Structure Norms (symmetric convention) ──")
        tee_print(f"    ||V_1||_F (cardinality) = {pen['norm_V1']:.4f}")
        tee_print(f"    ||V_2||_F (budget)      = {pen['norm_V2']:.4f}")

        tee_print(f"\n  ── Information Costs ──")
        tee_print(f"    I_1 (cardinality) = {I_1:.6f} bits  ({I_1/I_TOTAL*100:.1f}%)")
        tee_print(f"    I_2 (budget)      = {I_2:.6f} bits  ({I_2/I_TOTAL*100:.1f}%)")
        tee_print(f"    I_total           = {I_TOTAL:.6f} bits")

        tee_print(f"\n  ── Peer Condition (conditioning guarantee) ──")
        tee_print(f"    α₁_peer = (I₁/I_tot) × ||H_obj||_F / ||V₁||_F = {pen['alpha1_peer']:.6f}")
        tee_print(f"    α₂_peer = (I₂/I_tot) × ||H_obj||_F / ||V₂||_F = {pen['alpha2_peer']:.6f}")

        tee_print(f"\n  ── Best Objective by Cardinality Level ──")
        for k in range(N_ASSETS + 1):
            obj = pen['best_obj_by_k'][k]
            port = pen['best_portfolio_by_k'][k]
            violation = (k - K) ** 2
            marker = " ◄ TARGET" if k == K else ""
            tee_print(f"    k={k}: f* = {obj:>+10.6f}  violation² = {violation:>2}"
                      f"  portfolio = {port}{marker}")

        tee_print(f"\n  ── Sufficiency Floor ──")
        f_star = pen['best_obj_by_k'][K]
        for k in range(N_ASSETS + 1):
            if k == K:
                continue
            obj_k = pen['best_obj_by_k'][k]
            profit = f_star - obj_k
            if profit > 0:
                violation_sq = (k - K) ** 2
                required = profit / violation_sq
                tee_print(f"      k={k}: profit = {profit:.6f}, violation² = {violation_sq}, "
                          f"required α₁ ≥ {required:.6f}")
        tee_print(f"    → α₁_suff = {pen['alpha1_suff']:.6f} (binding at k={pen['binding_k']})")

        tee_print(f"\n    Budget sufficiency floor:")
        tee_print(f"      Best budget-feasible k=4 obj:  {pen['best_feas_obj']:.6f}")
        tee_print(f"      Best budget-violating k=4 obj: {pen['best_viol_obj']:.6f}")
        tee_print(f"    → α₂_suff = {pen['alpha2_suff']:.6f}")

        ALPHA_1 = pen['alpha1_final']
        ALPHA_2 = pen['alpha2_final']

        tee_print(f"\n  ══ THEORETICAL PENALTY COEFFICIENTS ══")
        tee_print(f"    α₁ = max(peer={pen['alpha1_peer']:.6f}, "
                  f"suff={pen['alpha1_suff']:.6f}) = {ALPHA_1:.6f}"
                  f"  [{'SUFFICIENCY' if pen['alpha1_suff'] > pen['alpha1_peer'] else 'PEER'} BINDS]")
        tee_print(f"    α₂ = max(peer={pen['alpha2_peer']:.6f}, "
                  f"suff={pen['alpha2_suff']:.6f}) = {ALPHA_2:.6f}"
                  f"  [{'SUFFICIENCY' if pen['alpha2_suff'] > pen['alpha2_peer'] else 'PEER'} BINDS]")

        # ──────────────────────────────────────────────────────────
        # QC SAFETY MARGIN APPLICATION
        # ──────────────────────────────────────────────────────────
        # The theoretical sufficiency floor places the k=3 interloper
        # only 1.76×10⁻⁶ above the ground state. On real quantum
        # hardware (gate error ~10⁻³, readout error ~10⁻²), this gap
        # is indistinguishable from noise. The safety margin widens
        # the gap without changing the ground state identity.
        alpha1_pre_margin = ALPHA_1
        ALPHA_1 *= QC_SAFETY_MARGIN

        if QC_SAFETY_MARGIN != 1.0:
            tee_print(f"\n  ══ QC SAFETY MARGIN APPLIED ══")
            tee_print(f"    α₁ (theoretical):  {alpha1_pre_margin:.6f}")
            tee_print(f"    Safety multiplier:  {QC_SAFETY_MARGIN:.2f}×")
            tee_print(f"    α₁ (QC-hardened):   {ALPHA_1:.6f}")
            tee_print(f"    α₂ (unchanged):     {ALPHA_2:.6f}")
            tee_print(f"    Rationale: Widen energy gap from ~1.76×10⁻⁶ to ~10⁻³")
            tee_print(f"               for noise resilience on NISQ hardware.")
        else:
            tee_print(f"\n  ── QC safety margin: DISABLED (exact theoretical mode) ──")

        tee_print(f"\n  ══ FINAL DEPLOYED PENALTY COEFFICIENTS ══")
        tee_print(f"    α₁ = {ALPHA_1:.6f}")
        tee_print(f"    α₂ = {ALPHA_2:.6f}")

        pen_norm_1 = ALPHA_1 * pen['norm_V1']
        pen_norm_2 = ALPHA_2 * pen['norm_V2']
        total_pen_norm = pen_norm_1 + pen_norm_2
        peer_ratio = total_pen_norm / pen['norm_H_obj']

        tee_print(f"\n  ── Frobenius Norm Budget ──")
        tee_print(f"    ||α₁ V₁||_F = {pen_norm_1:.6f}")
        tee_print(f"    ||α₂ V₂||_F = {pen_norm_2:.6f}")
        tee_print(f"    Penalty/Obj ratio = {peer_ratio:.2f}×")

        # ----------------------------------------------------------
        # Step 1: Build the QUBO matrix
        # ----------------------------------------------------------
        tee_print("\n" + "─" * 60)
        tee_print("STEP 1: QUBO MATRIX CONSTRUCTION (symmetric convention)")
        tee_print("─" * 60)

        Q = build_qubo_matrix(ALPHA_1, ALPHA_2)

        tee_print(f"\nQUBO matrix dimensions: {Q.shape[0]} × {Q.shape[1]}")
        tee_print(f"  Decision variables: {N_ASSETS} (indices 0–7)")
        tee_print(f"  Slack variables:    {M_SLACK} (indices 8–12)")
        tee_print(f"  Total qubits:       {N_TOTAL}")

        # Decision block
        tee_print(f"\nDecision block (8×8):")
        tee_print("        ", "  ".join(f"{t:>8}" for t in TICKERS))
        for i in range(N_ASSETS):
            row = "  ".join(f"{Q[i,j]:>8.5f}" for j in range(N_ASSETS))
            tee_print(f"  {TICKERS[i]:>5}  {row}")

        # Scale verification
        tee_print(f"\n  Scale verification (diagonal entries):")
        tee_print(f"    {'Asset':<6} {'H_obj':>10} {'Card pen':>10} {'Bud pen':>10} {'Q_ii':>10}")
        for i in range(N_ASSETS):
            h_obj_ii = SIGMA[i, i] - LAMBDA_R * MU[i]
            card_ii = ALPHA_1 * (1 - 2 * K)
            bud_ii = ALPHA_2 * C_PRIME[i] * (C_PRIME[i] - 2 * B_PRIME)
            tee_print(f"    {TICKERS[i]:<6} {h_obj_ii:>+10.5f} {card_ii:>+10.5f} "
                      f"{bud_ii:>+10.5f} {Q[i,i]:>+10.5f}")

        # Slack diagonal
        tee_print(f"\n  Slack diagonal:")
        for k in range(M_SLACK):
            idx = N_ASSETS + k
            tee_print(f"    Q[s{k+1},s{k+1}] = {Q[idx,idx]:.6f}  (w_{k+1} = {W_SLACK[k]:.1f})")

        # Symmetry check
        sym_error = np.max(np.abs(Q - Q.T))
        tee_print(f"\n  Symmetry check: max|Q - Q^T| = {sym_error:.2e} "
                  f"{'✓ SYMMETRIC' if sym_error < 1e-12 else '✗ ASYMMETRIC'}")

        # ----------------------------------------------------------
        # Convention Verification
        # ----------------------------------------------------------
        tee_print(f"\n  ── CONVENTION VERIFICATION ──")
        # Test with a known state: x = all 1s (k=8)
        z_test = np.ones(N_TOTAL)
        ver = verify_qubo_energy(Q, z_test, ALPHA_1, ALPHA_2)
        tee_print(f"    Test state: all 1s (k=8, all slack on)")
        tee_print(f"      f_obj           = {ver['f_obj']:+.8f}")
        tee_print(f"      Card penalty    = α₁×{ver['card_violation']:.0f}² = {ver['card_penalty']:.8f}")
        tee_print(f"      Budget penalty  = α₂×{ver['budget_expr']:.4f}² = {ver['budget_penalty']:.8f}")
        tee_print(f"      E_full          = {ver['E_full']:+.8f}")
        tee_print(f"      E_full - offset = {ver['E_expected']:+.8f}")
        tee_print(f"      z^T Q z         = {ver['E_qubo']:+.8f}")
        tee_print(f"      |Error|         = {ver['error']:.2e} "
                  f"{'✓ CONVENTION CORRECT' if ver['error'] < 1e-10 else '✗ CONVENTION ERROR'}")

        # Test with target portfolio
        z_target = np.zeros(N_TOTAL)
        target_indices = [0, 4, 5, 6]  # NVDA, JNJ, PG, JPM
        for idx in target_indices:
            z_target[idx] = 1.0
        ver2 = verify_qubo_energy(Q, z_target, ALPHA_1, ALPHA_2)
        tee_print(f"\n    Test state: target portfolio (NVDA,JNJ,PG,JPM), no slack")
        tee_print(f"      f_obj           = {ver2['f_obj']:+.8f}")
        tee_print(f"      Card penalty    = α₁×{ver2['card_violation']:.0f}² = {ver2['card_penalty']:.8f}")
        tee_print(f"      Budget penalty  = α₂×{ver2['budget_expr']:.4f}² = {ver2['budget_penalty']:.8f}")
        tee_print(f"      E_full          = {ver2['E_full']:+.8f}")
        tee_print(f"      E_full - offset = {ver2['E_expected']:+.8f}")
        tee_print(f"      z^T Q z         = {ver2['E_qubo']:+.8f}")
        tee_print(f"      |Error|         = {ver2['error']:.2e} "
                  f"{'✓' if ver2['error'] < 1e-10 else '✗ CONVENTION ERROR'}")

        # ----------------------------------------------------------
        # Step 2: Exhaustive enumeration
        # ----------------------------------------------------------
        tee_print("\n" + "─" * 60)
        tee_print("STEP 2: EXHAUSTIVE ENUMERATION (2^13 = 8192 states)")
        tee_print("─" * 60)

        states = enumerate_all_states(Q)

        # Ground state
        gs = states[0]
        tee_print(f"\n★ QUBO GROUND STATE:")
        tee_print(f"  Full bitstring:  {gs.bitstring}")
        tee_print(f"  Decision bits:   {gs.decision}  → {gs.portfolio}")
        tee_print(f"  Slack bits:      {gs.slack}  → slack = {gs.slack_value:.4f}")
        tee_print(f"  QUBO energy:     {gs.energy:.8f}")
        tee_print(f"  Natural obj:     {gs.obj_natural:.8f}")
        tee_print(f"  Assets selected: {gs.n_selected}")
        tee_print(f"  Normalized cost: {gs.cost:.6f}")
        tee_print(f"  Budget residual: {gs.budget_residual:.6f}")
        tee_print(f"  Card. violation: ({gs.n_selected} - {K})² = {gs.card_penalty:.0f}")
        tee_print(f"  Budget residual²: {gs.budget_penalty:.8f}")

        # Verify ground state convention
        z_gs_np = np.array(gs.bitstring, dtype=float)
        gs_ver = verify_qubo_energy(Q, z_gs_np, ALPHA_1, ALPHA_2)
        tee_print(f"\n  Ground state convention check:")
        tee_print(f"    E_full (independent) = {gs_ver['E_full']:+.8f}")
        tee_print(f"    z^T Q z + offset     = {gs_ver['E_qubo'] + gs_ver['const_offset']:+.8f}")
        tee_print(f"    |Error| = {gs_ver['error']:.2e} "
                  f"{'✓' if gs_ver['error'] < 1e-10 else '✗'}")

        # Cross-validation against Phase 1
        expected_portfolio = {"NVDA", "JNJ", "PG", "JPM"}
        actual_portfolio = set(gs.portfolio)
        portfolio_match = actual_portfolio == expected_portfolio

        tee_print(f"\n  Phase 1 cross-validation:")
        tee_print(f"    Expected portfolio: {sorted(expected_portfolio)}")
        tee_print(f"    QUBO ground state:  {sorted(actual_portfolio)}")
        tee_print(f"    Match: {'✓ CONFIRMED' if portfolio_match else '✗ MISMATCH'}")

        # ── FIX: Corrected Phase 1 reference value ──
        # Was -0.018997 (transcription error); correct value per Phase 1 §1.7 is:
        phase1_obj = -0.018961
        obj_diff = abs(gs.obj_natural - phase1_obj)
        tee_print(f"    Phase 1 reference obj:  {phase1_obj:.6f}")
        tee_print(f"    QUBO ground state obj:  {gs.obj_natural:.6f}")
        tee_print(f"    |Difference|: {obj_diff:.2e} "
                  f"{'✓' if obj_diff < 1e-3 else '⚠ CHECK'}")

        # Top 20 states
        tee_print(f"\n  Top 20 QUBO states (by energy):")
        tee_print(f"  {'Rank':<5} {'Decision':>12} {'Slack':>8} {'Energy':>12} "
                  f"{'NatObj':>10} {'|x|':>4} {'Cost':>8} {'Residual':>9} {'F?':<3} {'Portfolio'}")
        tee_print("  " + "─" * 115)

        for rank, st in enumerate(states[:20], 1):
            dec_str = "".join(str(b) for b in st.decision)
            slk_str = "".join(str(b) for b in st.slack)
            port_str = ", ".join(st.portfolio) if st.portfolio else "∅"
            is_feas = st.n_selected == K and st.cost <= B_PRIME + 1e-6
            feas_mark = "✓" if is_feas else " "
            tee_print(f"  {rank:<5} {dec_str:>12} {slk_str:>8} {st.energy:>12.6f} "
                      f"{st.obj_natural:>10.6f} {st.n_selected:>4} {st.cost:>8.4f} "
                      f"{st.budget_residual:>+9.4f} {feas_mark:<3} {port_str}")

        # Energy gap
        gap_01 = states[1].energy - states[0].energy if len(states) > 1 else 0.0
        tee_print(f"\n  Energy gap (ground → 1st excited): {gap_01:.8f}")
        frob = np.linalg.norm(Q, 'fro')
        tee_print(f"  Relative gap / ||Q||_F: {gap_01 / frob * 100:.4f}%")

        if QC_SAFETY_MARGIN != 1.0:
            tee_print(f"  ⚡ Gap with safety margin ({QC_SAFETY_MARGIN:.2f}×): {gap_01:.8f}")
            tee_print(f"     (vs. ~1.76×10⁻⁶ at theoretical minimum)")

        # Feasibility statistics
        feasible_energies = [st.energy for st in states
                             if st.n_selected == K and st.cost <= B_PRIME + 1e-6]
        infeasible_energies = [st.energy for st in states
                               if st.n_selected != K or st.cost > B_PRIME + 1e-6]

        n_card_feasible = sum(1 for st in states if st.n_selected == K)

        tee_print(f"\n  State space analysis (8192 total):")
        tee_print(f"    Cardinality-feasible (|x| = 4): {n_card_feasible}  ({n_card_feasible/8192*100:.1f}%)")

        if feasible_energies:
            tee_print(f"\n  Energy statistics — feasible states (card+budget):")
            tee_print(f"    Count: {len(feasible_energies)}")
            tee_print(f"    Min:   {min(feasible_energies):.6f}")
            tee_print(f"    Max:   {max(feasible_energies):.6f}")
            tee_print(f"    Mean:  {np.mean(feasible_energies):.6f}")

        if infeasible_energies:
            tee_print(f"  Energy statistics — infeasible states:")
            tee_print(f"    Count: {len(infeasible_energies)}")
            tee_print(f"    Min:   {min(infeasible_energies):.6f}")
            tee_print(f"    Max:   {max(infeasible_energies):.6f}")
            tee_print(f"    Mean:  {np.mean(infeasible_energies):.6f}")

        # Penalty sufficiency check
        tee_print(f"\n  ── PENALTY SUFFICIENCY CHECK ──")
        gs_is_feasible = (gs.n_selected == K and gs.cost <= B_PRIME + 1e-6)
        tee_print(f"    Ground state feasible?  |x| = {gs.n_selected}, "
                  f"cost = {gs.cost:.4f}  → {'YES ✓' if gs_is_feasible else 'NO ✗'}")

        penalty_check_passed = False
        if feasible_energies and infeasible_energies:
            min_feasible = min(feasible_energies)
            min_infeasible = min(infeasible_energies)
            penalty_gap = min_infeasible - min_feasible

            tee_print(f"    Min feasible energy:    {min_feasible:.6f}")
            tee_print(f"    Min infeasible energy:  {min_infeasible:.6f}")
            tee_print(f"    Gap (infeas − feas):    {penalty_gap:+.6f}")

            if gs_is_feasible and penalty_gap > 0:
                tee_print(f"    ✓ PENALTY SUFFICIENT: All infeasible states above feasible minimum.")
                penalty_check_passed = True
            elif gs_is_feasible:
                tee_print(f"    ⚠ Ground state feasible but some infeasible states are lower.")
                penalty_check_passed = False
            else:
                tee_print(f"    ✗ PENALTY INSUFFICIENT: Ground state is INFEASIBLE.")
                penalty_check_passed = False

        # ----------------------------------------------------------
        # Step 3: Ising Hamiltonian
        # ----------------------------------------------------------
        tee_print("\n" + "─" * 60)
        tee_print("STEP 3: ISING HAMILTONIAN MAPPING")
        tee_print("─" * 60)

        J, h_ising, E_0 = qubo_to_ising(Q)

        tee_print(f"\n  Ising offset E_0 = {E_0:.8f}")
        tee_print(f"\n  Local fields h_i:")
        for i in range(N_TOTAL):
            label = TICKERS[i] if i < N_ASSETS else f"s{i - N_ASSETS + 1}"
            tee_print(f"    h[{label:>5}] = {h_ising[i]:>+10.6f}")

        tee_print(f"\n  Coupling strengths J_ij (decision block, top 5 by |J|):")
        j_entries = []
        for i in range(N_ASSETS):
            for j in range(i + 1, N_ASSETS):
                j_entries.append((abs(J[i, j]), J[i, j], i, j))
        j_entries.sort(reverse=True)
        for _, jval, i, j in j_entries[:5]:
            tee_print(f"    J[{TICKERS[i]:>5}, {TICKERS[j]:>5}] = {jval:>+10.6f}")

        tee_print(f"\n  Coupling strengths J_ij (weakest 3 in decision block):")
        for _, jval, i, j in j_entries[-3:]:
            tee_print(f"    J[{TICKERS[i]:>5}, {TICKERS[j]:>5}] = {jval:>+10.6f}")

        n_j_total = sum(1 for i in range(N_TOTAL) for j in range(i+1, N_TOTAL)
                        if abs(J[i, j]) > 1e-15)
        n_j_max = N_TOTAL * (N_TOTAL - 1) // 2
        tee_print(f"\n  Total couplings: {n_j_total} / {n_j_max} ({n_j_total/n_j_max*100:.1f}%)")

        max_error = verify_ising_mapping(Q, J, h_ising, E_0)
        tee_print(f"\n  Ising mapping verification (100 random states):")
        tee_print(f"    Max |E_QUBO - E_Ising| = {max_error:.2e} "
                  f"{'✓ VERIFIED' if max_error < 1e-10 else '✗ ERROR'}")

        # Ground state hand-trace
        tee_print(f"\n  ── GROUND STATE ISING HAND-TRACE ──")
        s_gs = 1.0 - 2.0 * z_gs_np

        E_ising_offset = E_0
        E_ising_linear = float(h_ising @ s_gs)
        E_ising_quad = 0.0
        for i in range(N_TOTAL):
            for j in range(i + 1, N_TOTAL):
                E_ising_quad += J[i, j] * s_gs[i] * s_gs[j]

        E_ising_total = E_ising_offset + E_ising_linear + E_ising_quad

        tee_print(f"    Spin config: {tuple(int(si) for si in s_gs)}")
        tee_print(f"    E_0 (offset):    {E_ising_offset:>+12.8f}")
        tee_print(f"    Σ h_i s_i:       {E_ising_linear:>+12.8f}")
        tee_print(f"    Σ J_ij s_i s_j:  {E_ising_quad:>+12.8f}")
        tee_print(f"    ────────────────────────────────")
        tee_print(f"    E_Ising total:   {E_ising_total:>+12.8f}")
        tee_print(f"    E_QUBO  (check): {gs.energy:>+12.8f}")
        ising_gs_error = abs(E_ising_total - gs.energy)
        tee_print(f"    |Difference|:    {ising_gs_error:.2e} {'✓' if ising_gs_error < 1e-10 else '✗'}")

        # ----------------------------------------------------------
        # Step 4: Spectral Analysis
        # ----------------------------------------------------------
        tee_print("\n" + "─" * 60)
        tee_print("STEP 4: SPECTRAL ANALYSIS")
        tee_print("─" * 60)

        spec = spectral_analysis(Q)

        tee_print(f"\n  Full 13×13 QUBO matrix:")
        tee_print(f"    Frobenius norm:    {spec['frob_full']:.6f}")
        tee_print(f"    Condition number:  {spec['kappa_full']:.2f}")
        tee_print(f"    Eigenvalue range:  [{spec['eigvals_full'][0]:.6f}, "
                  f"{spec['eigvals_full'][-1]:.6f}]")
        tee_print(f"    All eigenvalues:")
        for idx, ev in enumerate(spec['eigvals_full']):
            tee_print(f"      λ_{idx:>2} = {ev:>+12.6f}")

        n_neg = sum(1 for ev in spec['eigvals_full'] if ev < -1e-12)
        n_pos = sum(1 for ev in spec['eigvals_full'] if ev > 1e-12)
        n_zero = N_TOTAL - n_neg - n_pos
        tee_print(f"    Signature: {n_pos} positive, {n_neg} negative, {n_zero} zero")

        tee_print(f"\n  Decision block (8×8):")
        tee_print(f"    Frobenius norm:    {spec['frob_dec']:.6f}")
        tee_print(f"    Condition number:  {spec['kappa_dec']:.2f}")
        tee_print(f"    Eigenvalue range:  [{spec['eigvals_dec'][0]:.6f}, "
                  f"{spec['eigvals_dec'][-1]:.6f}]")
        tee_print(f"    All eigenvalues:")
        for idx, ev in enumerate(spec['eigvals_dec']):
            tee_print(f"      λ_{idx:>2} = {ev:>+12.6f}")

        n_neg_dec = sum(1 for ev in spec['eigvals_dec'] if ev < -1e-12)
        n_pos_dec = sum(1 for ev in spec['eigvals_dec'] if ev > 1e-12)
        n_zero_dec = N_ASSETS - n_neg_dec - n_pos_dec
        tee_print(f"    Signature: {n_pos_dec} positive, {n_neg_dec} negative, {n_zero_dec} zero")

        # ----------------------------------------------------------
        # Step 5: Sector Constraints
        # ----------------------------------------------------------
        tee_print("\n" + "─" * 60)
        tee_print("STEP 5: SECTOR CONSTRAINT ANALYSIS")
        tee_print("─" * 60)

        tee_print(f"\n  Sector definitions (M_S = {M_S}):")
        for sector, indices in SECTORS.items():
            members = [TICKERS[i] for i in sorted(indices)]
            tee_print(f"    {sector:<20} → {members}  (max {M_S})")

        gs_dec_idx = set(i for i in range(N_ASSETS) if gs.decision[i] == 1)
        tee_print(f"\n  Ground state sector check:")
        all_sectors_ok = True
        for sector, indices in SECTORS.items():
            count = len(gs_dec_idx & indices)
            ok = count <= M_S
            if not ok:
                all_sectors_ok = False
            tee_print(f"    {sector:<20}: {count}/{M_S}  {'✓' if ok else '✗ VIOLATED'}")
        tee_print(f"    Overall: {'✓ ALL COMPLIANT' if all_sectors_ok else '✗ VIOLATION'}")

        n_sector_violations = 0
        for st in states:
            dec_idx = set(i for i in range(N_ASSETS) if st.decision[i] == 1)
            for sector, indices in SECTORS.items():
                if len(dec_idx & indices) > M_S:
                    n_sector_violations += 1
                    break
        tee_print(f"\n  States violating sector constraints: "
                  f"{n_sector_violations} / 8192 ({n_sector_violations/8192*100:.1f}%)")

        # ----------------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------------
        tee_print("\n" + "=" * 80)
        tee_print("PHASE 3 — SUMMARY")
        tee_print("=" * 80)

        tee_print(f"\n  Convention:            Symmetric Q, E = z^T Q z")
        tee_print(f"  QUBO matrix:           {N_TOTAL}×{N_TOTAL}, symmetric ✓")
        tee_print(f"  Penalty derivation:    RIT peer + sufficiency floor")
        if QC_SAFETY_MARGIN != 1.0:
            tee_print(f"  ⚡ QC safety margin:    {QC_SAFETY_MARGIN:.2f}× on α₁")
        tee_print(f"  α₁ (cardinality):      {ALPHA_1:.6f}"
                  f"  [{'suff' if pen['alpha1_suff'] > pen['alpha1_peer'] else 'peer'}"
                  f"{f' × {QC_SAFETY_MARGIN}' if QC_SAFETY_MARGIN != 1.0 else ''}]")
        tee_print(f"  α₂ (budget):           {ALPHA_2:.6f}"
                  f"  [{'suff' if pen['alpha2_suff'] > pen['alpha2_peer'] else 'peer'}]")
        tee_print(f"  Penalty/Obj ratio:     {peer_ratio:.2f}×")
        tee_print(f"  Convention verified:    {'✓' if ver['error'] < 1e-10 and ver2['error'] < 1e-10 else '✗'}")
        tee_print(f"  States enumerated:     8192")
        tee_print(f"  Ground state:          {gs.portfolio}")
        tee_print(f"  Ground state energy:   {gs.energy:.8f}")
        tee_print(f"  Natural objective:     {gs.obj_natural:.8f}")
        tee_print(f"  Portfolio match:       {'✓' if portfolio_match else '✗'}")
        tee_print(f"  Penalty sufficiency:   {'✓' if penalty_check_passed else '✗'}")
        tee_print(f"  Ising mapping:         {'✓' if max_error < 1e-10 else '✗'}")
        tee_print(f"  Sector compliance:     {'✓' if all_sectors_ok else '✗'}")
        tee_print(f"  Energy gap:            {gap_01:.8f}")
        tee_print(f"  Condition number:      {spec['kappa_full']:.2f} (full), "
                  f"{spec['kappa_dec']:.2f} (decision)")

        tee_print(f"\n  Output saved to: {os.path.abspath(OUTPUT_FILENAME)}")
        tee_print("=" * 80)

    finally:
        if _output_file is not None:
            _output_file.close()
            _output_file = None


if __name__ == "__main__":
    main()
