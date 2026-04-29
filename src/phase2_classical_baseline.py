"""
Phase 2 — Classical Optimization Baseline
Constrained Combinatorial Portfolio Selection
RIT Information-Theoretic Penalty Framework

All numerical values locked to Phase 1 formulation.
"""

import numpy as np
from itertools import combinations
from typing import NamedTuple

# ============================================================
# 2.2.1  Problem Data (exact values from Phase 1)
# ============================================================

TICKERS = ["NVDA", "MSFT", "LLY", "XOM", "JNJ", "PG", "JPM", "BRK-B"]
N = len(TICKERS)
K = 4          # cardinality constraint
M_S = 2        # sector cap (per sector)
LAMBDA_R = 0.341503

# Normalized costs
C_PRIME = np.array([
    0.24947, 0.48927, 1.00000, 0.17067,
    0.25953, 0.17092, 0.35891, 0.54454
])

B_PRIME = 1.638036  # normalized budget

# Annualized mean returns
MU = np.array([
    0.69074, 0.11672, 0.27136, 0.10990,
    0.13740, 0.00855, 0.29607, 0.12231
])

# Annualized covariance matrix (symmetric, from Phase 1 §1.4.1)
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

# Sector partition (index sets)
SECTORS = {
    "Technology":       {0, 1},
    "Financials":       {6, 7},
    "Healthcare":       {2, 4},
    "Consumer Staples": {5},
    "Energy":           {3},
}


# ============================================================
# 2.2.2  Helper Functions
# ============================================================

class Portfolio(NamedTuple):
    indices: tuple
    tickers: tuple
    objective: float
    risk: float          # x^T Σ x  (portfolio variance)
    ret: float           # μ^T x    (sum of annualized returns)
    cost: float          # c'^T x   (normalized cost)
    budget_feasible: bool
    sector_feasible: bool
    cardinality_ok: bool


def evaluate(x: np.ndarray) -> Portfolio:
    """Evaluate a binary vector x ∈ {0,1}^n on all metrics."""
    indices = tuple(np.where(x > 0.5)[0])
    tickers = tuple(TICKERS[i] for i in indices)

    risk = float(x @ SIGMA @ x)
    ret = float(MU @ x)
    cost = float(C_PRIME @ x)
    objective = risk - LAMBDA_R * ret

    budget_ok = cost <= B_PRIME + 1e-9   # small tolerance for float
    card_ok = int(x.sum()) == K

    sector_ok = True
    for sector_name, sector_set in SECTORS.items():
        count = sum(x[i] for i in sector_set)
        if count > M_S:
            sector_ok = False
            break

    return Portfolio(
        indices=indices,
        tickers=tickers,
        objective=objective,
        risk=risk,
        ret=ret,
        cost=cost,
        budget_feasible=budget_ok,
        sector_feasible=sector_ok,
        cardinality_ok=card_ok,
    )


def is_fully_feasible(p: Portfolio) -> bool:
    return p.budget_feasible and p.sector_feasible and p.cardinality_ok


# ============================================================
# 2.2.3  Method 1: Brute Force Enumeration
# ============================================================

def brute_force_enumerate():
    """
    Enumerate all C(8,4) = 70 cardinality-feasible portfolios.
    Return sorted list of Portfolio objects (best objective first).
    """
    results = []
    for combo in combinations(range(N), K):
        x = np.zeros(N)
        for i in combo:
            x[i] = 1.0
        p = evaluate(x)
        results.append(p)

    # Sort by objective (lower = better)
    results.sort(key=lambda p: p.objective)
    return results


def brute_force_full_space():
    """
    Enumerate ALL 2^8 = 256 binary vectors.
    Used to verify budget feasibility count |Ω_g2| = 130.
    """
    budget_feasible_count = 0
    all_results = []
    for bits in range(2**N):
        x = np.array([(bits >> i) & 1 for i in range(N)], dtype=float)
        p = evaluate(x)
        all_results.append(p)
        if p.budget_feasible:
            budget_feasible_count += 1
    return all_results, budget_feasible_count


# ============================================================
# 2.2.4  Method 2: Greedy Marginal Improvement
# ============================================================

def greedy_marginal():
    """
    Greedy heuristic: iteratively select the asset that yields
    the best marginal improvement in the objective, subject to
    budget feasibility at each step.

    At each step k (selecting asset k+1 out of K):
      - For each unselected asset i, compute Δf = f(x ∪ {i}) - f(x)
      - Among those that keep the portfolio budget-feasible
        (with remaining picks accounted for), select min Δf.

    Note: This greedy does NOT guarantee global optimality.
    """
    selected = []
    remaining = list(range(N))

    for step in range(K):
        best_idx = None
        best_delta = float('inf')
        picks_remaining = K - step - 1

        for i in remaining:
            # Tentatively add asset i
            trial = selected + [i]
            x_trial = np.zeros(N)
            for j in trial:
                x_trial[j] = 1.0

            # Budget feasibility check:
            # Current cost + minimum possible cost for remaining picks
            current_cost = C_PRIME[trial].sum()

            if picks_remaining > 0:
                # Remaining candidates (excluding trial assets)
                future_candidates = [r for r in remaining if r != i]
                if len(future_candidates) < picks_remaining:
                    continue  # Can't fill remaining slots
                # Optimistic: assume cheapest remaining assets fill slots
                future_costs = sorted([C_PRIME[r] for r in future_candidates])
                min_future_cost = sum(future_costs[:picks_remaining])
                if current_cost + min_future_cost > B_PRIME + 1e-9:
                    continue  # Even optimistically, budget will be exceeded
            else:
                # This is the last pick
                if current_cost > B_PRIME + 1e-9:
                    continue

            # Sector feasibility check
            sector_ok = True
            for sector_name, sector_set in SECTORS.items():
                count = sum(1 for j in trial if j in sector_set)
                if count > M_S:
                    sector_ok = False
                    break
            if not sector_ok:
                continue

            # Marginal objective
            risk_trial = float(x_trial @ SIGMA @ x_trial)
            ret_trial = float(MU @ x_trial)
            obj_trial = risk_trial - LAMBDA_R * ret_trial

            if obj_trial < best_delta:
                best_delta = obj_trial
                best_idx = i

        if best_idx is None:
            # No feasible addition found — greedy fails
            print(f"  [Greedy] No feasible asset at step {step+1}. Aborting.")
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    # Final evaluation
    x_final = np.zeros(N)
    for i in selected:
        x_final[i] = 1.0

    return evaluate(x_final)


# ============================================================
# 2.2.5  Method 3: Penalty-Relaxed Continuous Optimization
# ============================================================

def continuous_relaxation():
    """
    Relax x ∈ {0,1}^n to x ∈ [0,1]^n.
    Solve the penalized continuous problem using projected gradient descent.

    min_x  x^T Σ x - λ_R μ^T x
           + α_1 (Σ x_i - K)^2
           + α_2 (c'^T x - B')^2   [one-sided: only penalize if > B']

    Then round the continuous solution to binary via top-K selection.
    """
    ALPHA_1 = 0.093940
    ALPHA_2 = 0.049093

    # Initialize at uniform
    x = np.full(N, K / N)  # = 0.5 for K=4, N=8

    lr = 0.005
    n_iter = 5000

    for iteration in range(n_iter):
        # Gradient of objective: 2Σx - λ_R μ
        grad_obj = 2 * SIGMA @ x - LAMBDA_R * MU

        # Gradient of cardinality penalty: 2α_1 (Σx_i - K) * 1
        card_violation = x.sum() - K
        grad_card = 2 * ALPHA_1 * card_violation * np.ones(N)

        # Gradient of budget penalty: 2α_2 (c'^T x - B') * c'
        budget_violation = C_PRIME @ x - B_PRIME
        # Only penalize overspend (inequality constraint)
        if budget_violation > 0:
            grad_budget = 2 * ALPHA_2 * budget_violation * C_PRIME
        else:
            grad_budget = np.zeros(N)

        grad = grad_obj + grad_card + grad_budget

        # Gradient descent step
        x = x - lr * grad

        # Project onto [0, 1]^n
        x = np.clip(x, 0.0, 1.0)

    # Rounding: select top-K by continuous value
    top_k_indices = np.argsort(x)[-K:]
    x_rounded = np.zeros(N)
    x_rounded[top_k_indices] = 1.0

    # Check feasibility of rounded solution
    p_rounded = evaluate(x_rounded)

    # If rounded solution is budget-infeasible, try next-best rounding
    if not p_rounded.budget_feasible:
        # Fallback: enumerate roundings by replacing most expensive
        # selected asset with next-best unselected
        best_fallback = p_rounded
        selected = sorted(top_k_indices, key=lambda i: -C_PRIME[i])
        unselected = [i for i in range(N) if i not in top_k_indices]
        unselected.sort(key=lambda i: x[i], reverse=True)

        for drop in selected:
            for add in unselected:
                trial = [i for i in top_k_indices if i != drop] + [add]
                x_trial = np.zeros(N)
                for i in trial:
                    x_trial[i] = 1.0
                p_trial = evaluate(x_trial)
                if is_fully_feasible(p_trial) and p_trial.objective < best_fallback.objective:
                    best_fallback = p_trial
        p_rounded = best_fallback

    return x, p_rounded


# ============================================================
# 2.2.6  Main Execution
# ============================================================

def main():
    print("=" * 80)
    print("PHASE 2 — CLASSICAL OPTIMIZATION BASELINE")
    print("=" * 80)

    # ----------------------------------------------------------
    # Method 1: Brute Force
    # ----------------------------------------------------------
    print("\n" + "─" * 60)
    print("METHOD 1: BRUTE FORCE ENUMERATION")
    print("─" * 60)

    portfolios = brute_force_enumerate()

    feasible = [p for p in portfolios if is_fully_feasible(p)]
    infeasible_budget = [p for p in portfolios if not p.budget_feasible]

    print(f"\nTotal cardinality-feasible portfolios: {len(portfolios)}")
    print(f"Budget-feasible: {len(feasible)}")
    print(f"Budget-infeasible: {len(infeasible_budget)}")
    print(f"Feasibility ratio: {len(feasible)/len(portfolios)*100:.1f}%")

    # Verify full-space budget count
    _, budget_count_256 = brute_force_full_space()
    print(f"\nFull space budget-feasible vectors: {budget_count_256} / 256")

    print(f"\n{'Rank':<5} {'Portfolio':<28} {'Cost':>7} {'Return':>8} "
          f"{'Variance':>9} {'Objective':>10} {'Feasible':>8}")
    print("─" * 85)

    for rank, p in enumerate(feasible[:10], 1):
        ticker_str = ", ".join(p.tickers)
        feas = "✓" if is_fully_feasible(p) else "✗"
        print(f"{rank:<5} {ticker_str:<28} {p.cost:>7.4f} {p.ret:>8.4f} "
              f"{p.risk:>9.4f} {p.objective:>10.5f} {feas:>8}")

    bf_best = feasible[0]
    print(f"\n★ BRUTE FORCE OPTIMUM: {bf_best.tickers}")
    print(f"  Objective: {bf_best.objective:.6f}")
    print(f"  Return:    {bf_best.ret:.5f}")
    print(f"  Variance:  {bf_best.risk:.5f}")
    print(f"  Cost:      {bf_best.cost:.5f}")

    # ----------------------------------------------------------
    # Method 2: Greedy
    # ----------------------------------------------------------
    print("\n" + "─" * 60)
    print("METHOD 2: GREEDY MARGINAL IMPROVEMENT")
    print("─" * 60)

    greedy_result = greedy_marginal()

    print(f"\n  Selected: {greedy_result.tickers}")
    print(f"  Objective: {greedy_result.objective:.6f}")
    print(f"  Return:    {greedy_result.ret:.5f}")
    print(f"  Variance:  {greedy_result.risk:.5f}")
    print(f"  Cost:      {greedy_result.cost:.5f}")
    print(f"  Budget feasible: {greedy_result.budget_feasible}")
    print(f"  Sector feasible: {greedy_result.sector_feasible}")

    # Find greedy rank in brute force
    greedy_rank = None
    for rank, p in enumerate(feasible, 1):
        if set(p.indices) == set(greedy_result.indices):
            greedy_rank = rank
            break

    if greedy_rank:
        print(f"  Rank among feasible portfolios: {greedy_rank} / {len(feasible)}")
        gap = (greedy_result.objective - bf_best.objective) / abs(bf_best.objective) * 100
        print(f"  Optimality gap vs brute force: {gap:+.2f}%")
    else:
        print(f"  ⚠ Greedy solution not in feasible set!")

    # ----------------------------------------------------------
    # Method 3: Continuous Relaxation
    # ----------------------------------------------------------
    print("\n" + "─" * 60)
    print("METHOD 3: CONTINUOUS RELAXATION + ROUNDING")
    print("─" * 60)

    x_continuous, relaxed_result = continuous_relaxation()

    print(f"\n  Continuous solution (pre-rounding):")
    for i in range(N):
        bar = "█" * int(x_continuous[i] * 30)
        print(f"    {TICKERS[i]:>6}: {x_continuous[i]:.4f}  {bar}")

    print(f"\n  Rounded selection: {relaxed_result.tickers}")
    print(f"  Objective: {relaxed_result.objective:.6f}")
    print(f"  Return:    {relaxed_result.ret:.5f}")
    print(f"  Variance:  {relaxed_result.risk:.5f}")
    print(f"  Cost:      {relaxed_result.cost:.5f}")
    print(f"  Budget feasible: {relaxed_result.budget_feasible}")

    relaxed_rank = None
    for rank, p in enumerate(feasible, 1):
        if set(p.indices) == set(relaxed_result.indices):
            relaxed_rank = rank
            break

    if relaxed_rank:
        print(f"  Rank among feasible portfolios: {relaxed_rank} / {len(feasible)}")
        gap = (relaxed_result.objective - bf_best.objective) / abs(bf_best.objective) * 100
        print(f"  Optimality gap vs brute force: {gap:+.2f}%")

    # ----------------------------------------------------------
    # Comparison Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'Method':<32} {'Portfolio':<28} {'Objective':>10} {'Rank':>6} {'Gap':>8}")
    print("─" * 90)
    print(f"{'Brute Force (exact)':<32} {str(bf_best.tickers):<28} "
          f"{bf_best.objective:>10.6f} {'1':>6} {'0.00%':>8}")

    g_rank_str = str(greedy_rank) if greedy_rank else "N/A"
    g_gap = (greedy_result.objective - bf_best.objective) / abs(bf_best.objective) * 100
    print(f"{'Greedy Marginal':<32} {str(greedy_result.tickers):<28} "
          f"{greedy_result.objective:>10.6f} {g_rank_str:>6} {g_gap:>+7.2f}%")

    r_rank_str = str(relaxed_rank) if relaxed_rank else "N/A"
    r_gap = (relaxed_result.objective - bf_best.objective) / abs(bf_best.objective) * 100
    print(f"{'Continuous Relaxation':<32} {str(relaxed_result.tickers):<28} "
          f"{relaxed_result.objective:>10.6f} {r_rank_str:>6} {r_gap:>+7.2f}%")

    # ----------------------------------------------------------
    # Phase 1 Cross-Validation
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 1 CROSS-VALIDATION")
    print("=" * 80)

    print(f"\n  Phase 1 claimed optimum: {{NVDA, JNJ, PG, JPM}}, f* = -0.018997")
    print(f"  Phase 2 computed optimum: {{{', '.join(bf_best.tickers)}}}, "
          f"f* = {bf_best.objective:.6f}")

    phase1_match = (set(bf_best.tickers) == {"NVDA", "JNJ", "PG", "JPM"})
    print(f"  Portfolio match: {'✓ CONFIRMED' if phase1_match else '✗ MISMATCH'}")

    obj_diff = abs(bf_best.objective - (-0.018997))
    print(f"  Objective difference: {obj_diff:.6e} "
          f"({'< 1e-4 ✓' if obj_diff < 1e-4 else '⚠ CHECK'})")

    print(f"\n  Phase 1 claimed |Ω_g2| = 130")
    print(f"  Phase 2 computed |Ω_g2| = {budget_count_256}")
    print(f"  Match: {'✓ CONFIRMED' if budget_count_256 == 130 else '✗ MISMATCH'}")

    print(f"\n  Phase 1 claimed feasible ratio: 35/70 = 50.0%")
    print(f"  Phase 2 computed: {len(feasible)}/70 = {len(feasible)/70*100:.1f}%")
    print(f"  Match: {'✓ CONFIRMED' if len(feasible) == 35 else '✗ MISMATCH'}")


if __name__ == "__main__":
    main()
