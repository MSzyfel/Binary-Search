#!/usr/bin/env python3
"""
generate_all_datasets.py

Generuje kompletny zestaw danych ze wszystkimi kombinacjami typów drzew i rozkładów.
Wszystkie dane są zapisywane w folderze data_set/.

Użycie:
    python generate_all_datasets.py --seed 42
    python generate_all_datasets.py --force  # nadpisz istniejące
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ============================================================
# KONFIGURACJA
# ============================================================

BASE_DIR = "data_set"

# Rozkłady do testowania
DISTRIBUTIONS = {
    "uniform": "Rozkład jednostajny",
    "normal": "Rozkład normalny", 
    "exponential": "Rozkład wykładniczy",
    "binomial": "Rozkład dwumianowy",
}

# Konfiguracje typów drzew
TREE_TYPES = {
    "probabilistic_trees": {
        "method": "random_probabilistic",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "params": [],
    },
    
    "stars": {
        "method": "random_star",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "params": [],
    },
    
    "caterpillars": {
        "method": "random_caterpillar",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "params": [],
    },
    
    "spiders": {
        "method": "random_spider",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "params": ["legs=4"],
    },
    
    "bounded_degree_3": {
        "method": "random_bounded_degree",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "params": ["delta=3"],
    },
    
    "bounded_degree_5": {
        "method": "random_bounded_degree",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "params": ["delta=5"],
    },
    
    "bounded_diameter_6": {
        "method": "random_bounded_diameter",
        "n_min": 10,
        "n_max": 50,
        "instances": 10,
        "params": ["D=6"],
    },
    
    "bounded_diameter_10": {
        "method": "random_bounded_diameter",
        "n_min": 10,
        "n_max": 50,
        "instances": 10,
        "params": ["D=10"],
    },
    
    "large_degree": {
        "method": "random_large_degree",
        "n_min": 20,
        "n_max": 100,
        "instances": 10,
        "params": ["delta=10"],
    },
    
    "large_diameter": {
        "method": "random_large_diameter",
        "n_min": 20,
        "n_max": 100,
        "instances": 10,
        "params": ["D=15"],
    },
    
    "full_binary": {
        "method": "random_full_mary",
        "n_min": 7,
        "n_max": 63,
        "instances": 5,
        "params": ["m=2"],
    },
    
    "full_ternary": {
        "method": "random_full_mary",
        "n_min": 10,
        "n_max": 40,
        "instances": 5,
        "params": ["m=3"],
    },
    
    "balanced_binary": {
        "method": "random_balanced",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "params": ["m=2"],
    },
    
    "brooms": {
        "method": "random_broom",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "params": [],
    },
    
    "lobsters": {
        "method": "random_lobster",
        "n_min": 15,
        "n_max": 100,
        "instances": 10,
        "params": [],
    },
    
    "probabilistic": {
        "method": "random_probabilistic",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "params": [],
    },
    
    "recursive": {
        "method": "random_recursive",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "params": [],
    },
    
    "preferential": {
        "method": "random_preferential",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "params": [],
    },
}


# ============================================================
# FUNKCJE
# ============================================================

def generate_dataset(tree_type: str, config: dict, dist_c: str, dist_w: str, 
                     base_seed: int, force: bool, fmt: str):
    """
    Generuje jeden zbiór danych dla danej kombinacji typu drzewa i rozkładów.
    """
    # Nazwa katalogu wyjściowego
    outdir = f"{BASE_DIR}/{tree_type}_c{dist_c}_w{dist_w}"
    
    print(f"\n{'='*70}")
    print(f"[GEN] Generating: {tree_type}")
    print(f"   Distribution C (node costs): {dist_c}")
    print(f"   Distribution W (edge weights): {dist_w}")
    print(f"   Output: {outdir}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable,
        "input_generation.py",
        "--method", config["method"],
        "--n-min", str(config["n_min"]),
        "--n-max", str(config["n_max"]),
        "--instances", str(config["instances"]),
        "--outdir", outdir,
        "--format", fmt,
        "--seed", str(base_seed),
        "--distribution-c", dist_c,
        "--distribution-w", dist_w,
    ]
    
    if force:
        cmd.append("--force")
    
    if config.get("params"):
        cmd.extend(["--params"] + config["params"])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"[OK] Success: {tree_type} (c={dist_c}, w={dist_w})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed: {tree_type} (c={dist_c}, w={dist_w})")
        print(f"Error: {e}")
        return False


def generate_summary(results: dict, output_file: Path):
    """Generuje podsumowanie wygenerowanych zbiorów danych."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Complete Dataset Generation Summary\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Results\n\n")
        success_count = sum(1 for v in results.values() if v)
        f.write(f"- Total datasets: {len(results)}\n")
        f.write(f"- Successful: {success_count}\n")
        f.write(f"- Failed: {len(results) - success_count}\n\n")
        
        f.write(f"## Distribution Combinations\n\n")
        f.write(f"Each tree type was generated with all combinations of:\n")
        for dist_name, dist_desc in DISTRIBUTIONS.items():
            f.write(f"- {dist_name}: {dist_desc}\n")
        f.write(f"\n")
        
        f.write(f"## Details by Tree Type\n\n")
        
        # Grupuj wyniki po typie drzewa
        by_tree_type = {}
        for key, success in results.items():
            tree_type, dist_info = key.split("_c")
            if tree_type not in by_tree_type:
                by_tree_type[tree_type] = []
            by_tree_type[tree_type].append((dist_info, success))
        
        for tree_type in sorted(by_tree_type.keys()):
            config = TREE_TYPES[tree_type]
            f.write(f"### {tree_type}\n")
            f.write(f"- Method: {config['method']}\n")
            f.write(f"- Size range: {config['n_min']} - {config['n_max']}\n")
            f.write(f"- Instances per size: {config['instances']}\n")
            if config.get('params'):
                f.write(f"- Parameters: {', '.join(config['params'])}\n")
            
            f.write(f"\n  Distribution combinations:\n")
            for dist_info, success in by_tree_type[tree_type]:
                status = "SUCCESS" if success else "FAILED"
                f.write(f"  - c{dist_info}: {status}\n")
            f.write(f"\n")
    
    print(f"\nSummary saved to: {output_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generuje kompletny zestaw danych ze wszystkimi kombinacjami rozkładów",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bazowy seed dla generatora (default: 42)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Nadpisz istniejące pliki"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["graphml", "gexf", "gpickle", "json"],
        default="graphml",
        help="Format plików wyjściowych (default: graphml)"
    )
    
    parser.add_argument(
        "--summary",
        type=str,
        default="dataset_summary.md",
        help="Nazwa pliku z podsumowaniem (default: dataset_summary.md)"
    )
    
    parser.add_argument(
        "--tree-types",
        nargs="+",
        choices=list(TREE_TYPES.keys()),
        help="Wybierz konkretne typy drzew (domyślnie: wszystkie)"
    )
    
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=list(DISTRIBUTIONS.keys()),
        help="Wybierz konkretne rozkłady (domyślnie: wszystkie)"
    )
    
    args = parser.parse_args()
    
    # Wybierz typy drzew i rozkłady
    tree_types_to_generate = args.tree_types if args.tree_types else list(TREE_TYPES.keys())
    distributions_to_use = args.distributions if args.distributions else list(DISTRIBUTIONS.keys())
    
    # Oblicz całkowitą liczbę zbiorów do wygenerowania
    total_datasets = len(tree_types_to_generate) * len(distributions_to_use) * len(distributions_to_use)
    
    # Informacje początkowe
    print(f"\n{'='*70}")
    print(f"[START] Complete Dataset Generation Started")
    print(f"{'='*70}")
    print(f"[INFO] Tree types: {len(tree_types_to_generate)}")
    print(f"[INFO] Distributions: {', '.join(distributions_to_use)}")
    print(f"[INFO] Total datasets to generate: {total_datasets}")
    print(f"[INFO] Base seed: {args.seed}")
    print(f"[INFO] Format: {args.format}")
    print(f"[INFO] Base directory: {BASE_DIR}/")
    print(f"[INFO] Force overwrite: {args.force}")
    print(f"{'='*70}\n")
    
    # Stwórz katalog bazowy
    Path(BASE_DIR).mkdir(exist_ok=True)
    
    # Generuj wszystkie kombinacje
    results = {}
    count = 0
    
    for tree_type in tree_types_to_generate:
        config = TREE_TYPES[tree_type]
        
        for dist_c in distributions_to_use:
            for dist_w in distributions_to_use:
                count += 1
                print(f"\n[{count}/{total_datasets}] Processing...")
                
                # Unikalny seed dla każdej kombinacji
                seed = args.seed + hash(f"{tree_type}_{dist_c}_{dist_w}") % 100000
                
                success = generate_dataset(
                    tree_type=tree_type,
                    config=config,
                    dist_c=dist_c,
                    dist_w=dist_w,
                    base_seed=seed,
                    force=args.force,
                    fmt=args.format
                )
                
                key = f"{tree_type}_c{dist_c}_w{dist_w}"
                results[key] = success
    
    # Podsumowanie
    print(f"\n{'='*70}")
    print(f"[SUMMARY] Final Summary")
    print(f"{'='*70}")
    
    success_count = sum(1 for v in results.values() if v)
    print(f"[OK] Successful: {success_count}/{len(results)}")
    print(f"[FAIL] Failed: {len(results) - success_count}/{len(results)}")
    
    # Zapisz podsumowanie
    summary_path = Path(args.summary)
    generate_summary(results, summary_path)
    
    print(f"\n[DONE] All done!\n")
    print(f"[INFO] All datasets are in: {BASE_DIR}/")
    print(f"[INFO] Summary: {summary_path}")


if __name__ == "__main__":
    main()
