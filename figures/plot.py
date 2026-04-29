"""
Project Q — Research Figures Generation
Information-Theoretic Penalty Calibration for QAOA Portfolio Optimization

Generates three figures:
1. Efficient Frontier Plot
2. QAOA Convergence Plot  
3. Solution Comparison Bar Chart

Author: Ankur Chakraborty
Date: 29 April 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# =============================================================================
# SECTION 1: DATA FROM PAPER
# =============================================================================

# Asset data (Table from Section 2.1)
TICKERS = ['NVDA', 'MSFT', 'LLY', 'XOM', 'JNJ', 'PG', 'JPM', 'BRK-B']
SECTORS = ['Tech', 'Tech', 'Health', 'Energy', 'Health', 'ConsStap', 'Fin', 'Fin']

# Normalized costs
C_PRIME = np.array([0.24947, 0.48927, 1.00000, 0.17067, 0.25953, 0.17092, 0.35891, 0.54454])

# Annualized returns
MU = np.array([0.69074, 0.11672, 0.27136, 0.10990, 0.13740, 0.00855, 0.29607, 0.12231])

# Annualized volatilities
SIGMA = np.array([0.48330, 0.23456, 0.35097, 0.22873, 0.17098, 0.17018, 0.22697, 0.15771])

# Full covariance matrix (from paper Appendix)
COVARIANCE = np.array([
    [0.23358, 0.05762, 0.03029, -0.00295, -0.02060, -0.01361, 0.02952, 0.00432],
    [0.05762, 0.05502, 0.01078, -0.00270, -0.00482, -0.00153, 0.01446, 0.00621],
    [0.03029, 0.01078, 0.12318, -0.00071, 0.01085, 0.01121, 0.01273, 0.01151],
    [-0.00295, -0.00270, -0.00071, 0.05232, 0.00538, 0.00386, 0.01381, 0.01065],
    [-0.02060, -0.00482, 0.01085, 0.00538, 0.02923, 0.01077, 0.00434, 0.00808],
    [-0.01361, -0.00153, 0.01121, 0.00386, 0.01077, 0.02896, 0.00188, 0.00810],
    [0.02952, 0.01446, 0.01273, 0.01381, 0.00434, 0.00188, 0.05151, 0.01927],
    [0.00432, 0.00621, 0.01151, 0.01065, 0.00808, 0.00810, 0.01927, 0.02487]
])

# Parameters from paper
K = 4  # Cardinality constraint
B_PRIME = 1.638036  # Budget threshold
LAMBDA_R = 0.341503  # Risk-return tradeoff

# QAOA results (Table from Section 4.1)
QAOA_RESULTS = {
    'depth': [1, 2, 3, 5],
    'cvar_obj': [-0.60146, -0.61037, -0.61276, -0.61433],
    'expected_H': [-0.47861, -0.49193, -0.51927, -0.53145],
    'approx_ratio': [0.7774, 0.7991, 0.8435, 0.8633],
    'p_gs': [0.000398, 0.001232, 0.002647, 0.006048],
    'enhancement': [3.3, 10.1, 21.7, 49.5],
    'p_bottom_1pct': [0.0361, 0.0943, 0.1584, 0.2422],
    'time_s': [10.9, 65.4, 210.1, 1260.0],
    'target_found': [True, True, True, True]  # All found via min-E extraction
}

# Classical baseline results (Table from Section 3.1)
CLASSICAL_RESULTS = {
    'Brute Force': {'objective': -0.018961, 'rank': 1, 'gap_pct': 0.0, 'correct': True},
    'Greedy': {'objective': -0.018961, 'rank': 1, 'gap_pct': 0.0, 'correct': True},
    'Cont. Relaxation': {'objective': 0.004188, 'rank': 6, 'gap_pct': 122.09, 'correct': False}
}

# Ground state energy
E0_EXACT = -0.61562627
P_UNIFORM = 1/8192  # Uniform probability for 13-qubit system

# =============================================================================
# SECTION 2: HELPER FUNCTIONS
# =============================================================================

def compute_portfolio_metrics(indices):
    """Compute risk, return, and objective for a portfolio given asset indices."""
    x = np.zeros(8)
    x[list(indices)] = 1
    
    # Portfolio return (sum of individual returns)
    port_return = np.dot(MU, x)
    
    # Portfolio variance (x^T Sigma x)
    port_variance = x @ COVARIANCE @ x
    port_risk = np.sqrt(port_variance)
    
    # Objective function: variance - lambda_R * return
    objective = port_variance - LAMBDA_R * port_return
    
    # Budget cost
    cost = np.dot(C_PRIME, x)
    
    return {
        'return': port_return,
        'risk': port_risk,
        'variance': port_variance,
        'objective': objective,
        'cost': cost,
        'indices': indices,
        'tickers': [TICKERS[i] for i in indices]
    }

def enumerate_all_portfolios():
    """Enumerate all K=4 portfolios and classify by feasibility."""
    all_portfolios = []
    
    for combo in combinations(range(8), K):
        metrics = compute_portfolio_metrics(combo)
        metrics['budget_feasible'] = metrics['cost'] <= B_PRIME
        all_portfolios.append(metrics)
    
    return all_portfolios

# =============================================================================
# SECTION 3: FIGURE 1 — EFFICIENT FRONTIER PLOT
# =============================================================================

def plot_efficient_frontier(save_path='fig1_efficient_frontier.png'):
    """
    Generate the efficient frontier plot showing:
    - All feasible portfolios
    - Budget-infeasible portfolios (greyed out)
    - The optimal portfolio highlighted
    - Individual assets for reference
    """
    
    portfolios = enumerate_all_portfolios()
    
    # Separate feasible and infeasible
    feasible = [p for p in portfolios if p['budget_feasible']]
    infeasible = [p for p in portfolios if not p['budget_feasible']]
    
    # Find optimal portfolio
    optimal = min(feasible, key=lambda p: p['objective'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot infeasible portfolios (grey, smaller)
    if infeasible:
        ax.scatter(
            [p['risk'] for p in infeasible],
            [p['return'] for p in infeasible],
            c='lightgray', s=40, alpha=0.5, marker='o',
            label=f'Budget-infeasible (n={len(infeasible)})', zorder=2
        )
    
    # Plot feasible portfolios
    feasible_risks = [p['risk'] for p in feasible]
    feasible_returns = [p['return'] for p in feasible]
    feasible_objs = [p['objective'] for p in feasible]
    
    # Color by objective value
    scatter = ax.scatter(
        feasible_risks, feasible_returns,
        c=feasible_objs, cmap='RdYlGn_r', s=80, 
        edgecolors='black', linewidths=0.5,
        label=f'Budget-feasible (n={len(feasible)})', zorder=3
    )
    
    # Highlight optimal portfolio with a star
    ax.scatter(
        optimal['risk'], optimal['return'],
        c='gold', s=400, marker='*', edgecolors='black', linewidths=1.5,
        label=f'Optimal: {{{", ".join(optimal["tickers"])}}}', zorder=5
    )
    
    # Add annotation for optimal
    ax.annotate(
        f'  f* = {optimal["objective"]:.6f}\n  σ = {optimal["risk"]:.4f}\n  μ = {optimal["return"]:.4f}',
        xy=(optimal['risk'], optimal['return']),
        xytext=(optimal['risk'] + 0.05, optimal['return'] - 0.15),
        fontsize=9, ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='gray')
    )
    
    # Plot individual assets
    ax.scatter(
        SIGMA, MU, c='navy', s=100, marker='D', 
        edgecolors='white', linewidths=1,
        label='Individual assets', zorder=4
    )
    
    # Label individual assets
    for i, ticker in enumerate(TICKERS):
        offset = {'NVDA': (0.01, 0.03), 'MSFT': (0.01, -0.05), 'LLY': (0.01, 0.02),
                  'XOM': (0.01, -0.04), 'JNJ': (-0.06, 0.01), 'PG': (0.01, -0.04),
                  'JPM': (0.01, 0.02), 'BRK-B': (0.01, -0.04)}
        dx, dy = offset.get(ticker, (0.01, 0.01))
        ax.annotate(ticker, (SIGMA[i], MU[i]), xytext=(SIGMA[i]+dx, MU[i]+dy),
                   fontsize=8, fontweight='bold', color='navy')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Objective f(x) = Var - λ·Return', fontsize=11)
    
    # Compute and plot approximate efficient frontier
    # Sort feasible by return and find Pareto-optimal points
    sorted_feasible = sorted(feasible, key=lambda p: p['return'])
    frontier_points = []
    min_risk = float('inf')
    for p in reversed(sorted_feasible):
        if p['risk'] < min_risk:
            frontier_points.append(p)
            min_risk = p['risk']
    
    if len(frontier_points) > 1:
        frontier_risks = [p['risk'] for p in frontier_points]
        frontier_returns = [p['return'] for p in frontier_points]
        # Sort by risk for plotting
        sorted_indices = np.argsort(frontier_risks)
        ax.plot(
            np.array(frontier_risks)[sorted_indices],
            np.array(frontier_returns)[sorted_indices],
            'b--', linewidth=2, alpha=0.7, label='Efficient frontier', zorder=1
        )
    
    # Labels and formatting
    ax.set_xlabel('Portfolio Risk (Annualized Volatility σ)', fontsize=12)
    ax.set_ylabel('Portfolio Return (Annualized μ)', fontsize=12)
    ax.set_title('Efficient Frontier: 8-Asset Constrained Portfolio Selection\n'
                 f'K={K} assets, Budget ≤ {B_PRIME:.3f}, λᵣ = {LAMBDA_R:.4f}', fontsize=13)
    
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(0.1, 0.55)
    ax.set_ylim(-0.1, 1.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✓ Efficient frontier plot saved to: {save_path}")
    print(f"  - Total portfolios (K=4): {len(portfolios)}")
    print(f"  - Budget-feasible: {len(feasible)}")
    print(f"  - Optimal: {optimal['tickers']} with f* = {optimal['objective']:.6f}")
    
    return fig

# =============================================================================
# SECTION 4: FIGURE 2 — QAOA CONVERGENCE PLOT
# =============================================================================

def plot_qaoa_convergence(save_path='fig2_qaoa_convergence.png'):
    """
    Generate multi-panel QAOA convergence plot showing:
    - Panel A: Approximation ratio vs depth
    - Panel B: Ground state probability enhancement
    - Panel C: CVaR objective convergence
    - Panel D: Bottom 1% probability concentration
    """
    
    depths = QAOA_RESULTS['depth']
    
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)
    
    # Color scheme
    colors = {'main': '#2E86AB', 'accent': '#A23B72', 'target': '#F18F01', 'grid': '#E8E8E8'}
    
    # -------------------------------------------------------------------------
    # Panel A: Approximation Ratio
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    ax1.plot(depths, QAOA_RESULTS['approx_ratio'], 'o-', 
             color=colors['main'], linewidth=2.5, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
    ax1.axhline(y=1.0, color=colors['target'], linestyle='--', linewidth=2, label='Exact (r = 1.0)')
    
    # Fill area to show gap
    ax1.fill_between(depths, QAOA_RESULTS['approx_ratio'], 1.0, alpha=0.15, color=colors['main'])
    
    # Annotate values
    for i, (d, r) in enumerate(zip(depths, QAOA_RESULTS['approx_ratio'])):
        ax1.annotate(f'{r:.3f}', (d, r), textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Circuit Depth p')
    ax1.set_ylabel('Approximation Ratio r = ⟨H⟩/E₀')
    ax1.set_title('(A) Approximation Ratio vs. Circuit Depth', fontweight='bold')
    ax1.set_xticks(depths)
    ax1.set_ylim(0.7, 1.05)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel B: Ground State Probability Enhancement
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Bar chart with enhancement values
    bars = ax2.bar(depths, QAOA_RESULTS['enhancement'], color=colors['accent'], 
                   edgecolor='black', linewidth=1, alpha=0.85)
    
    # Add value labels on bars
    for bar, enh in zip(bars, QAOA_RESULTS['enhancement']):
        height = bar.get_height()
        ax2.annotate(f'{enh:.1f}×',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Reference line for uniform
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, label='Uniform baseline (1×)')
    
    ax2.set_xlabel('Circuit Depth p')
    ax2.set_ylabel('Enhancement over Uniform')
    ax2.set_title('(B) Ground State Probability Enhancement', fontweight='bold')
    ax2.set_xticks(depths)
    ax2.set_yscale('log')
    ax2.set_ylim(0.8, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')
    
    # -------------------------------------------------------------------------
    # Panel C: CVaR Objective Convergence
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.plot(depths, QAOA_RESULTS['cvar_obj'], 's-', 
             color=colors['main'], linewidth=2.5, markersize=10, 
             markeredgecolor='white', markeredgewidth=1.5, label='CVaR₀.₁₀ objective')
    ax3.plot(depths, QAOA_RESULTS['expected_H'], '^-', 
             color=colors['accent'], linewidth=2, markersize=9,
             markeredgecolor='white', markeredgewidth=1.5, label='⟨H⟩ expected energy')
    ax3.axhline(y=E0_EXACT, color=colors['target'], linestyle='--', linewidth=2, 
                label=f'E₀ = {E0_EXACT:.5f}')
    
    ax3.set_xlabel('Circuit Depth p')
    ax3.set_ylabel('Energy')
    ax3.set_title('(C) Energy Convergence', fontweight='bold')
    ax3.set_xticks(depths)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel D: Bottom 1% Probability Concentration
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.plot(depths, np.array(QAOA_RESULTS['p_bottom_1pct']) * 100, 'D-', 
             color=colors['main'], linewidth=2.5, markersize=10,
             markeredgecolor='white', markeredgewidth=1.5)
    
    # Reference: uniform would give 1%
    ax4.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, label='Uniform baseline (1%)')
    
    # Fill area
    ax4.fill_between(depths, np.array(QAOA_RESULTS['p_bottom_1pct']) * 100, 
                     alpha=0.2, color=colors['main'])
    
    # Annotate
    for d, p in zip(depths, QAOA_RESULTS['p_bottom_1pct']):
        ax4.annotate(f'{p*100:.1f}%', (d, p*100), textcoords='offset points', 
                    xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('Circuit Depth p')
    ax4.set_ylabel('Probability in Bottom 1% (%)')
    ax4.set_title('(D) Low-Energy Tail Concentration', fontweight='bold')
    ax4.set_xticks(depths)
    ax4.set_ylim(0, 30)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('QAOA Performance: CVaR-Optimized Portfolio Selection (13 Qubits)\n'
                 'Multi-Start Powell Optimization, 50 Restarts per Depth', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✓ QAOA convergence plot saved to: {save_path}")
    
    return fig

# =============================================================================
# SECTION 5: FIGURE 3 — SOLUTION COMPARISON BAR CHART
# =============================================================================

def plot_solution_comparison(save_path='fig3_solution_comparison.png'):
    """
    Generate solution comparison bar chart showing:
    - Objective values achieved by different methods
    - Whether each method found the correct portfolio
    - QAOA results at different depths
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # -------------------------------------------------------------------------
    # Panel A: Objective Function Values
    # -------------------------------------------------------------------------
    
    # Prepare data
    methods = ['Brute\nForce', 'Greedy', 'Cont.\nRelax.', 'QAOA\np=1', 'QAOA\np=2', 'QAOA\np=3', 'QAOA\np=5']
    
    # For QAOA, we use the natural objective of the extracted solution
    # All QAOA depths found the correct portfolio, so objective is f* = -0.018961
    f_star = -0.018961
    objectives = [
        f_star,  # Brute Force
        f_star,  # Greedy  
        0.004188,  # Continuous Relaxation
        f_star,  # QAOA p=1 (found via min-E)
        f_star,  # QAOA p=2 (found via min-E)
        f_star,  # QAOA p=3
        f_star   # QAOA p=5
    ]
    
    # Colors based on correctness
    colors = ['#2ECC71', '#2ECC71', '#E74C3C', '#3498DB', '#3498DB', '#3498DB', '#3498DB']
    edge_colors = ['darkgreen', 'darkgreen', 'darkred', 'darkblue', 'darkblue', 'darkblue', 'darkblue']
    
    bars = ax1.bar(methods, objectives, color=colors, edgecolor=edge_colors, linewidth=2)
    
    # Reference line at optimal
    ax1.axhline(y=f_star, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal f* = {f_star:.6f}')
    
    # Annotate bars
    for bar, obj in zip(bars, objectives):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 0.002 if height >= 0 else -0.002
        ax1.annotate(f'{obj:.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5 if height >= 0 else -12),
                    textcoords='offset points',
                    ha='center', va=va, fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('Objective Value f(x)', fontsize=12)
    ax1.set_title('(A) Solution Quality: Objective Function Value', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.set_ylim(-0.03, 0.015)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add legend for colors
    legend_elements = [
        Patch(facecolor='#2ECC71', edgecolor='darkgreen', label='Correct portfolio'),
        Patch(facecolor='#E74C3C', edgecolor='darkred', label='Incorrect portfolio'),
        Patch(facecolor='#3498DB', edgecolor='darkblue', label='QAOA (correct via extraction)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # -------------------------------------------------------------------------
    # Panel B: Method Comparison Metrics
    # -------------------------------------------------------------------------
    
    # Comparison of key metrics across methods
    method_names = ['Brute Force', 'Greedy', 'QAOA p=5']
    
    x = np.arange(len(method_names))
    width = 0.35
    
    # Metric 1: Gap to optimum (%)
    gaps = [0.0, 0.0, 0.0]  # All find correct answer
    
    # Metric 2: Computational cost (log scale, arbitrary units showing relative cost)
    # Brute force: O(C(8,4)) = 70 evaluations
    # Greedy: O(K*n) = 32 evaluations  
    # QAOA: 50 * 2000 = 100,000 evaluations (but would be different on quantum hardware)
    compute_cost = [70, 32, 100000]  
    
    # Create grouped bar chart
    ax2_1 = ax2
    ax2_2 = ax2.twinx()
    
    # We'll show: time comparison and correctness
    times = [0.001, 0.001, 1260]  # seconds (approximate)
    
    bars1 = ax2_1.bar(x - width/2, [0, 0, 0], width, label='Gap to Optimum (%)', color='#27AE60', alpha=0.8)
    
    # Instead, let's do a more informative comparison
    ax2.clear()
    
    # Create a table-style comparison
    comparison_data = {
        'Method': ['Brute Force', 'Greedy', 'Cont. Relaxation', 'QAOA p=5'],
        'Correct': ['✓', '✓', '✗', '✓'],
        'Rank': [1, 1, 6, 1],
        'Gap (%)': ['0.0%', '0.0%', '+122.1%', '0.0%'],
        'Time': ['<1 ms', '<1 ms', '<1 ms', '~21 min']
    }
    
    # Create stacked horizontal bar showing components
    methods_compare = ['Brute Force\n(classical)', 'Greedy\n(classical)', 'Relaxation\n(classical)', 'QAOA p=5\n(quantum sim.)']
    
    # Metrics to show
    # 1. Solution quality (1 = optimal, 0 = worst)
    quality_scores = [1.0, 1.0, 0.0, 1.0]  # Binary: found optimal or not
    
    # 2. Approximation ratio (for meaningful comparison with QAOA)
    approx_ratios = [1.0, 1.0, -0.22, 0.863]  # Cont relax is negative, QAOA is 86.3%
    
    y_pos = np.arange(len(methods_compare))
    
    # Quality bars
    bars = ax2.barh(y_pos, quality_scores, height=0.6, color=['#27AE60', '#27AE60', '#E74C3C', '#3498DB'],
                    edgecolor='black', linewidth=1.5)
    
    # Add text annotations
    for i, (method, correct, rank, gap) in enumerate(zip(
        methods_compare, 
        comparison_data['Correct'],
        comparison_data['Rank'],
        comparison_data['Gap (%)']
    )):
        # Status text
        status = '✓ Optimal' if correct == '✓' else f'✗ Rank {rank}'
        color = 'darkgreen' if correct == '✓' else 'darkred'
        ax2.annotate(status, (0.5, i), ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='white')
        
        # Gap annotation on right
        ax2.annotate(f'Gap: {gap}', (1.05, i), ha='left', va='center', fontsize=10)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods_compare, fontsize=11)
    ax2.set_xlim(0, 1.5)
    ax2.set_xlabel('Solution Quality (1 = Optimal Found)', fontsize=12)
    ax2.set_title('(B) Method Comparison: Correctness & Quality', fontweight='bold', fontsize=13)
    ax2.axvline(x=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Hide x-axis ticks (it's essentially binary)
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xticklabels(['Infeasible/Wrong', '', 'Optimal'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✓ Solution comparison plot saved to: {save_path}")
    
    return fig

# =============================================================================
# SECTION 6: MAIN EXECUTION
# =============================================================================

def generate_all_figures():
    """Generate all three figures for the research paper."""
    
    print("="*60)
    print("Project Q — Research Figure Generation")
    print("="*60)
    print()
    
    # Figure 1: Efficient Frontier
    print("Generating Figure 1: Efficient Frontier Plot...")
    fig1 = plot_efficient_frontier('fig1_efficient_frontier.png')
    print()
    
    # Figure 2: QAOA Convergence
    print("Generating Figure 2: QAOA Convergence Plot...")
    fig2 = plot_qaoa_convergence('fig2_qaoa_convergence.png')
    print()
    
    # Figure 3: Solution Comparison
    print("Generating Figure 3: Solution Comparison Bar Chart...")
    fig3 = plot_solution_comparison('fig3_solution_comparison.png')
    print()
    
    print("="*60)
    print("All figures generated successfully!")
    print("="*60)
    
    return fig1, fig2, fig3

if __name__ == "__main__":
    generate_all_figures()
