import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
from math import comb, log2, ceil

# ══════════════════════════════════════════════════════════════
# CONFIGURATION — matches Phase 1 formulation exactly
# ══════════════════════════════════════════════════════════════
tickers = ['NVDA', 'MSFT', 'LLY', 'XOM', 'JNJ', 'PG', 'JPM', 'BRK-B']
n = len(tickers)
K = 4                # cardinality constraint
M_s = 2              # sector concentration cap
delta = 0.1          # slack discretization resolution (§1.5)
target_feas_lo = 0.40  # target feasibility window lower bound
target_feas_hi = 0.60  # target feasibility window upper bound

sectors = {
    'Technology':       ['NVDA', 'MSFT'],
    'Healthcare':       ['LLY', 'JNJ'],
    'Energy':           ['XOM'],
    'Consumer Staples': ['PG'],
    'Financials':       ['JPM', 'BRK-B'],
}

sector_indices = {
    'Technology':       [0, 1],
    'Financials':       [6, 7],
    'Healthcare':       [2, 4],
    'Consumer Staples': [5],
    'Energy':           [3],
}

start = '2023-04-28'
end   = '2026-04-28'

# ══════════════════════════════════════════════════════════════
# RAW DATA PULL
# ══════════════════════════════════════════════════════════════
print("Pulling data from yfinance...")
data = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']

# Reindex columns to match our ticker order (yfinance sometimes alphabetizes)
data = data[tickers]

# ══════════════════════════════════════════════════════════════
# CORE COMPUTATIONS (§1.2 – §1.4)
# ══════════════════════════════════════════════════════════════

# Latest closing prices (for c_i and normalization)
latest_prices = data.iloc[-1]

# Normalized cost vector c_i' = c_i / max(c_j) — §1.3
c_max = latest_prices.max()
c_max_ticker = latest_prices.idxmax()
c_prime = latest_prices / c_max
c_prime_arr = c_prime.values
c_bar_prime = np.mean(c_prime_arr)

# Daily log-returns
log_returns = np.log(data / data.shift(1)).dropna()

# Annualized mean returns μ_i = 252 * mean(r_i) — §1.4.2
mu = log_returns.mean() * 252
mu_arr = mu.values

# Annualized covariance matrix Σ = Cov(r) * 252 — §1.4.1
cov_matrix = log_returns.cov() * 252
cov_arr = cov_matrix.values

# Correlation matrix
corr_matrix = log_returns.corr()

# Annualized volatilities σ_i = sqrt(Σ_ii)
sigma = np.sqrt(np.diag(cov_arr))

# Risk-return trade-off parameter λ_R = σ̄² / μ̄ — §1.4.3
sigma_bar_sq = np.mean(sigma ** 2)
mu_bar = np.mean(mu_arr)
lambda_R = sigma_bar_sq / mu_bar

# ══════════════════════════════════════════════════════════════
# INFORMATION COST — CARDINALITY (§1.8.2)
# ══════════════════════════════════════════════════════════════

# I_1 = log2(2^n / C(n,K))
omega_total = 2 ** n
omega_g1 = comb(n, K)
I_1 = log2(omega_total / omega_g1)

# ══════════════════════════════════════════════════════════════
# BUDGET CALIBRATION BY ENUMERATION (§1.5, §1.8.2)
# ══════════════════════════════════════════════════════════════

# Generate all C(8,4) = 70 cardinality-feasible subsets
all_K_subsets = list(combinations(range(n), K))
subset_costs = []
for subset in all_K_subsets:
    cost = sum(c_prime_arr[i] for i in subset)
    subset_costs.append((subset, cost))

# Sort by cost for calibration
subset_costs.sort(key=lambda x: x[1])
costs_only = np.array([sc[1] for sc in subset_costs])

# Sweep B' to find target feasibility window (40-60%)
# Start from the natural B' = K * c̄' / τ and try τ values
print("Calibrating budget B' by enumeration...")

budget_sweep = []
tau_values = np.arange(0.80, 1.50, 0.01)
for tau in tau_values:
    B_candidate = (K * c_bar_prime) / tau
    n_feasible = int(np.sum(costs_only <= B_candidate))
    frac = n_feasible / len(all_K_subsets)
    budget_sweep.append((tau, B_candidate, n_feasible, frac))

# Find best τ that gives closest to 50% feasibility
best_sweep = min(budget_sweep, key=lambda x: abs(x[3] - 0.50))
tau_star = best_sweep[0]
B_prime = best_sweep[1]
n_budget_feasible = best_sweep[2]
feas_fraction = best_sweep[3]

# Count |Ω_g2| over ALL 256 binary vectors (not just K=4 subsets)
omega_g2_full = 0
for bits in range(omega_total):
    x_vec = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
    if np.dot(c_prime_arr, x_vec) <= B_prime:
        omega_g2_full += 1

# I_2 = log2(256 / |Ω_g2|) — §1.8.2
I_2 = log2(omega_total / omega_g2_full)

# Joint feasibility: subsets satisfying BOTH cardinality AND budget
n_joint_feasible = int(np.sum(costs_only <= B_prime))

# ══════════════════════════════════════════════════════════════
# SECTOR CONSTRAINT VERIFICATION (§1.8.2)
# ══════════════════════════════════════════════════════════════

sector_info_costs = {}
for sname, indices in sector_indices.items():
    n_s = len(indices)
    feasible_count = sum(comb(n_s, k) for k in range(0, min(M_s, n_s) + 1))
    total_sector = 2 ** n_s
    if feasible_count >= total_sector:
        I_s = 0.0
    else:
        I_s = log2(total_sector / feasible_count)
    sector_info_costs[sname] = {
        'n_s': n_s,
        'M_s': M_s,
        'feasible_configs': feasible_count,
        'total_configs': total_sector,
        'I_s': I_s,
    }

# ══════════════════════════════════════════════════════════════
# H_obj FROBENIUS NORM & PENALTY COEFFICIENTS (§1.8.4)
# ══════════════════════════════════════════════════════════════

# Build H_obj matrix (objective part of QUBO)
# Diagonal: Σ_ii - λ_R * μ_i
# Off-diagonal: Σ_ij
H_obj = cov_arr.copy()
for i in range(n):
    H_obj[i, i] -= lambda_R * mu_arr[i]

H_obj_frob = np.linalg.norm(H_obj, 'fro')

# Total information cost — only cardinality + budget survive
I_total = I_1 + I_2

# Penalty coefficients — §1.8.4
alpha_1 = I_1 * (H_obj_frob / I_total)
alpha_2 = I_2 * (H_obj_frob / I_total)

# ══════════════════════════════════════════════════════════════
# SLACK VARIABLE SIZING (§1.5, §1.12)
# ══════════════════════════════════════════════════════════════

m_slack = ceil(log2(B_prime / delta + 1))
N_total_qubits = n + m_slack

# Also compute for δ = 0.05 for comparison
delta_fine = 0.05
m_slack_fine = ceil(log2(B_prime / delta_fine + 1))
N_total_qubits_fine = n + m_slack_fine

# ══════════════════════════════════════════════════════════════
# QUBO MATRIX ELEMENT PREVIEW (§1.9)
# ══════════════════════════════════════════════════════════════

# Decision variable diagonal: Q_ii = Σ_ii - λ_R*μ_i + α_1*(1 - 2K) + α_2*c_i'*(c_i' - 2B')
Q_diag = np.zeros(n)
for i in range(n):
    Q_diag[i] = (cov_arr[i, i]
                 - lambda_R * mu_arr[i]
                 + alpha_1 * (1 - 2 * K)
                 + alpha_2 * c_prime_arr[i] * (c_prime_arr[i] - 2 * B_prime))

# Decision variable off-diagonal: Q_ij = Σ_ij + 2α_1 + 2α_2*c_i'*c_j'
Q_offdiag = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        Q_offdiag[i, j] = cov_arr[i, j] + 2 * alpha_1 + 2 * alpha_2 * c_prime_arr[i] * c_prime_arr[j]
        Q_offdiag[j, i] = Q_offdiag[i, j]

# Build full decision-variable block of Q (8×8)
Q_decision = Q_offdiag.copy()
for i in range(n):
    Q_decision[i, i] = Q_diag[i]

# Condition number of the decision-variable block
eigenvalues = np.linalg.eigvalsh(Q_decision)
kappa_Q = abs(eigenvalues[-1]) / abs(eigenvalues[0]) if abs(eigenvalues[0]) > 1e-15 else float('inf')

# ══════════════════════════════════════════════════════════════
# FULL ENUMERATION TABLE — ALL 70 PORTFOLIOS RANKED (§1.8.2)
# ══════════════════════════════════════════════════════════════

portfolio_table = []
for subset in all_K_subsets:
    x_vec = np.zeros(n)
    for i in subset:
        x_vec[i] = 1.0

    # Portfolio risk: x^T Σ x
    risk = x_vec @ cov_arr @ x_vec

    # Portfolio return: μ^T x
    ret = mu_arr @ x_vec

    # Total normalized cost: c'^T x
    cost = c_prime_arr @ x_vec

    # Objective: risk - λ_R * return
    obj = risk - lambda_R * ret

    # Budget feasibility
    budget_ok = cost <= B_prime

    port_tickers = [tickers[i] for i in subset]
    portfolio_table.append({
        'subset': subset,
        'tickers': port_tickers,
        'cost': cost,
        'return': ret,
        'risk': risk,
        'objective': obj,
        'budget_feasible': budget_ok,
    })

# Sort by objective value
portfolio_table.sort(key=lambda x: x['objective'])

# ══════════════════════════════════════════════════════════════
# WRITE EVERYTHING TO FILE
# ══════════════════════════════════════════════════════════════

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', '{:.6f}'.format)

output_file = 'phase1_complete_data3.txt'

with open(output_file, 'w') as f:
    f.write(f"{'=' * 120}\n")
    f.write(f"PHASE 1 — COMPLETE DATA DUMP FOR PHASE 2 TRANSITION\n")
    f.write(f"Generated: {datetime.now()}\n")
    f.write(f"Tickers: {tickers}\n")
    f.write(f"Period: {start} to {end}\n")
    f.write(f"Trading days in dataset: {len(log_returns)}\n")
    f.write(f"{'=' * 120}\n\n")

    # ── §1.2 LATEST PRICES & NORMALIZED COSTS ──
    f.write("§1.2–1.3  LATEST CLOSING PRICES & NORMALIZED COST VECTOR\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Ticker':<8} {'Price (USD)':>14} {'c_i_prime':>14}\n")
    f.write("-" * 40 + "\n")
    for i, t in enumerate(tickers):
        f.write(f"{t:<8} {latest_prices.iloc[i]:>14.2f} {c_prime_arr[i]:>14.6f}\n")
    f.write(f"\nMax price: {c_max:.2f} ({c_max_ticker})\n")
    f.write(f"Mean normalized cost c̄' = {c_bar_prime:.6f}\n\n")

    # ── §1.4.2 ANNUALIZED RETURNS ──
    f.write("§1.4.2  ANNUALIZED MEAN RETURNS μ_i = 252 × mean(log-return_i)\n")
    f.write("-" * 80 + "\n")
    for i, t in enumerate(tickers):
        f.write(f"μ({t}) = {mu_arr[i]:.6f}\n")
    f.write(f"\nμ̄ (mean of μ_i) = {mu_bar:.6f}\n\n")

    # ── ANNUALIZED VOLATILITIES ──
    f.write("§1.4.1  ANNUALIZED VOLATILITIES σ_i = sqrt(Σ_ii)\n")
    f.write("-" * 80 + "\n")
    for i, t in enumerate(tickers):
        f.write(f"σ({t}) = {sigma[i]:.6f}    σ²({t}) = {sigma[i]**2:.6f}\n")
    f.write(f"\nσ̄² (mean of variances) = {sigma_bar_sq:.6f}\n\n")

    # ── §1.4.3 RISK-RETURN TRADE-OFF ──
    f.write("§1.4.3  RISK-RETURN TRADE-OFF PARAMETER\n")
    f.write("-" * 80 + "\n")
    f.write(f"λ_R = σ̄² / μ̄ = {sigma_bar_sq:.6f} / {mu_bar:.6f} = {lambda_R:.6f}\n\n")

    # ── §1.4.1 FULL COVARIANCE MATRIX ──
    f.write("§1.4.1  FULL ANNUALIZED COVARIANCE MATRIX Σ (Cov × 252)\n")
    f.write("-" * 80 + "\n")
    f.write(cov_matrix.to_string() + "\n\n")

    # ── FULL CORRELATION MATRIX ──
    f.write("FULL CORRELATION MATRIX ρ\n")
    f.write("-" * 80 + "\n")
    f.write(corr_matrix.to_string() + "\n\n")

    # ── ALL UNIQUE COV/CORR ENTRIES ──
    f.write("ALL UNIQUE COVARIANCE/CORRELATION ENTRIES (upper triangle)\n")
    f.write("-" * 80 + "\n")
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if j >= i:
                f.write(f"Σ({ti},{tj}) = {cov_arr[i,j]:.6f},  "
                        f"ρ({ti},{tj}) = {corr_matrix.iloc[i,j]:.6f}\n")
    f.write("\n")

    # ── §1.8.2 INFORMATION COSTS ──
    f.write(f"{'=' * 120}\n")
    f.write("§1.8.2  INFORMATION-THEORETIC SPECIFICATION COSTS\n")
    f.write("-" * 80 + "\n")
    f.write(f"I_1 (cardinality) = log2({omega_total} / {omega_g1}) = {I_1:.6f} bits\n")
    f.write(f"I_2 (budget)      = log2({omega_total} / {omega_g2_full}) = {I_2:.6f} bits\n")
    f.write(f"I_total           = I_1 + I_2 = {I_total:.6f} bits\n\n")

    f.write("SECTOR CONSTRAINT VERIFICATION (§1.8.2)\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Sector':<20} {'n_s':>4} {'M_s':>4} {'Feasible':>10} {'Total':>8} {'I_s (bits)':>12}\n")
    f.write("-" * 62 + "\n")
    for sname, info in sector_info_costs.items():
        f.write(f"{sname:<20} {info['n_s']:>4} {info['M_s']:>4} "
                f"{info['feasible_configs']:>10} {info['total_configs']:>8} "
                f"{info['I_s']:>12.6f}\n")
    f.write("\n→ ALL sector constraints carry 0 bits. Vacuous at n=8. α_3,s = 0 ∀ s.\n\n")

    # ── §1.5 BUDGET CALIBRATION ──
    f.write(f"{'=' * 120}\n")
    f.write("§1.5  BUDGET CALIBRATION BY ENUMERATION\n")
    f.write("-" * 80 + "\n")
    f.write(f"K × c̄' = {K} × {c_bar_prime:.6f} = {K * c_bar_prime:.6f}\n")
    f.write(f"Selected τ* = {tau_star:.2f}\n")
    f.write(f"B' = {K * c_bar_prime:.6f} / {tau_star:.2f} = {B_prime:.6f}\n")
    f.write(f"Budget-feasible K=4 subsets: {n_budget_feasible} / {len(all_K_subsets)} "
            f"= {feas_fraction:.1%}\n")
    f.write(f"|Ω_g2| (all 256 vectors): {omega_g2_full}\n\n")

    f.write("BUDGET SWEEP TABLE (τ scan)\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'τ':>6} {'B_prime':>12} {'# feasible (of 70)':>20} {'fraction':>10}\n")
    f.write("-" * 52 + "\n")
    for tau_v, bv, nf, frac in budget_sweep:
        marker = " ◄" if abs(frac - 0.50) < 0.01 else ""
        f.write(f"{tau_v:>6.2f} {bv:>12.6f} {nf:>20} {frac:>10.1%}{marker}\n")
    f.write("\n")

    # ── §1.8.4 PENALTY COEFFICIENTS ──
    f.write(f"{'=' * 120}\n")
    f.write("§1.8.4  PENALTY COEFFICIENTS (RIT-DERIVED)\n")
    f.write("-" * 80 + "\n")
    f.write(f"‖H_obj‖_F = {H_obj_frob:.6f}\n")
    f.write(f"I_total    = {I_total:.6f} bits\n")
    f.write(f"α_1 = I_1 × ‖H_obj‖_F / I_total = {I_1:.6f} × {H_obj_frob:.6f} / {I_total:.6f} = {alpha_1:.6f}\n")
    f.write(f"α_2 = I_2 × ‖H_obj‖_F / I_total = {I_2:.6f} × {H_obj_frob:.6f} / {I_total:.6f} = {alpha_2:.6f}\n")
    f.write(f"α_3,s = 0 ∀ s (sector constraints vacuous)\n\n")

    # ── §1.12 QUBIT COUNT ──
    f.write(f"{'=' * 120}\n")
    f.write("§1.12  SLACK VARIABLES & QUBIT COUNT\n")
    f.write("-" * 80 + "\n")
    f.write(f"δ = {delta} → m = ⌈log2({B_prime:.4f}/{delta} + 1)⌉ = {m_slack} slack bits\n")
    f.write(f"   N_total = {n} + {m_slack} = {N_total_qubits} qubits\n")
    f.write(f"δ = {delta_fine} → m = ⌈log2({B_prime:.4f}/{delta_fine} + 1)⌉ = {m_slack_fine} slack bits\n")
    f.write(f"   N_total = {n} + {m_slack_fine} = {N_total_qubits_fine} qubits\n\n")

    # ── §1.9 QUBO MATRIX PREVIEW ──
    f.write(f"{'=' * 120}\n")
    f.write("§1.9  QUBO MATRIX — DECISION VARIABLE BLOCK (8×8 preview)\n")
    f.write("-" * 80 + "\n")
    f.write("\nDiagonal elements Q_ii:\n")
    for i, t in enumerate(tickers):
        f.write(f"Q[{t},{t}] = Σ_ii({cov_arr[i,i]:.6f}) - λ_R·μ_i({lambda_R:.4f}×{mu_arr[i]:.6f}) "
                f"+ α_1·(1-2K)({alpha_1:.4f}×{1-2*K}) "
                f"+ α_2·c'(c'-2B')({alpha_2:.4f}×{c_prime_arr[i]:.4f}×{c_prime_arr[i]-2*B_prime:.4f}) "
                f"= {Q_diag[i]:.6f}\n")
    f.write("\nFull 8×8 Q decision block:\n")
    Q_df = pd.DataFrame(Q_decision, index=tickers, columns=tickers)
    f.write(Q_df.to_string() + "\n\n")

    f.write(f"Eigenvalues of Q_decision block: {np.sort(eigenvalues)}\n")
    f.write(f"Condition number κ(Q_decision) = {kappa_Q:.4f}\n\n")

    # ── H_obj MATRIX (for reference) ──
    f.write("H_obj MATRIX (objective-only QUBO, before penalties):\n")
    f.write("-" * 80 + "\n")
    H_obj_df = pd.DataFrame(H_obj, index=tickers, columns=tickers)
    f.write(H_obj_df.to_string() + "\n\n")

    # ── FULL 70-PORTFOLIO ENUMERATION ──
    f.write(f"{'=' * 120}\n")
    f.write("COMPLETE ENUMERATION — ALL 70 CARDINALITY-FEASIBLE PORTFOLIOS\n")
    f.write("(Sorted by objective value, best first)\n")
    f.write("-" * 120 + "\n")
    f.write(f"{'Rank':>4} {'Tickers':<28} {'Cost':>10} {'Return':>10} {'Risk':>10} "
            f"{'Objective':>12} {'Budget OK':>10}\n")
    f.write("-" * 120 + "\n")
    for rank, p in enumerate(portfolio_table, 1):
        tk_str = ', '.join(p['tickers'])
        bflag = "YES" if p['budget_feasible'] else "no"
        f.write(f"{rank:>4} {tk_str:<28} {p['cost']:>10.6f} {p['return']:>10.6f} "
                f"{p['risk']:>10.6f} {p['objective']:>12.6f} {bflag:>10}\n")
    f.write("\n")

    # Identify the optimal feasible portfolio
    feasible_portfolios = [p for p in portfolio_table if p['budget_feasible']]
    if feasible_portfolios:
        best = feasible_portfolios[0]
        f.write(f"OPTIMAL FEASIBLE PORTFOLIO: {', '.join(best['tickers'])}\n")
        f.write(f"  Cost     = {best['cost']:.6f}\n")
        f.write(f"  Return   = {best['return']:.6f}\n")
        f.write(f"  Risk     = {best['risk']:.6f}\n")
        f.write(f"  Objective = {best['objective']:.6f}\n")
    f.write("\n")

    # ── SUMMARY BLOCK FOR PHASE 2 ──
    f.write(f"{'=' * 120}\n")
    f.write("PHASE 2 HANDOFF — ALL LOCKED PARAMETERS\n")
    f.write(f"{'=' * 120}\n")
    f.write(f"n              = {n}\n")
    f.write(f"K              = {K}\n")
    f.write(f"B'             = {B_prime:.6f}\n")
    f.write(f"δ              = {delta}\n")
    f.write(f"m (slack bits) = {m_slack}\n")
    f.write(f"N_total        = {N_total_qubits} qubits\n")
    f.write(f"λ_R            = {lambda_R:.6f}\n")
    f.write(f"I_1            = {I_1:.6f} bits\n")
    f.write(f"I_2            = {I_2:.6f} bits\n")
    f.write(f"I_total        = {I_total:.6f} bits\n")
    f.write(f"‖H_obj‖_F     = {H_obj_frob:.6f}\n")
    f.write(f"α_1            = {alpha_1:.6f}\n")
    f.write(f"α_2            = {alpha_2:.6f}\n")
    f.write(f"α_3,s          = 0 (all sectors)\n")
    f.write(f"κ(Q_decision)  = {kappa_Q:.4f}\n")
    f.write(f"Feasible portfolios: {len(feasible_portfolios)} / {len(all_K_subsets)}\n")
    if feasible_portfolios:
        f.write(f"Classical optimum: {', '.join(feasible_portfolios[0]['tickers'])} "
                f"(obj = {feasible_portfolios[0]['objective']:.6f})\n")
    f.write(f"\nμ vector: {mu_arr}\n")
    f.write(f"σ vector: {sigma}\n")
    f.write(f"c' vector: {c_prime_arr}\n")
    f.write(f"\nAll data dependencies from §1.13 are LOCKED. Ready for Phase 2.\n")

print(f"\nDone. All Phase 1 data written to: {output_file}")
