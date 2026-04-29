# main.py - runs entire pipeline
from src import phase1_data_preparation
from src import phase2_classical_baseline  
from src import phase3_qubo_validation
from src import phase4_qaoa_optimization

if __name__ == "__main__":
    print("Running full Quantum Portfolio Optimization pipeline...")
    # Each module has its own main() already
