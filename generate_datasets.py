#!/usr/bin/env python3
"""
generate_datasets.py

Zbiorczy skrypt do generowania wszystkich potrzebnych zbiorów danych.
Automatyzuje wywołania input_generation.py dla różnych konfiguracji.

Użycie:
    python generate_datasets.py --preset all
    python generate_datasets.py --preset basic
    python generate_datasets.py --preset experimental
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ============================================================
# KONFIGURACJA ZBIORÓW DANYCH
# ============================================================

# Rozkłady dla wag węzłów i krawędzi
DISTRIBUTIONS = ["uniform", "normal", "exponential"]

def create_datasets_with_distributions():
    """Tworzy konfiguracje dla wszystkich kombinacji typów drzew i rozkładów"""
    datasets = {}
    
    base_configs = {
        # Drzewa losowe - podstawowe
        "random_trees": {
            "method": "random_labeled_tree",
            "n_min": 10,
            "n_max": 100,
            "instances": 10,
        },
    
    # Gwiazdy (star graphs)
    "stars": {
        "method": "random_star",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_stars",
        "params": [],
    },
    
    # Gąsienice (caterpillars)
    "caterpillars": {
        "method": "random_caterpillar",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_caterpillars",
        "params": [],
    },
    
    # Pająki (spiders)
    "spiders": {
        "method": "random_spider",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_spiders",
        "params": ["legs=4"],
    },
    
    # Drzewa z ograniczonym stopniem (bounded degree)
    "bounded_degree_3": {
        "method": "random_bounded_degree",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_bounded_degree_3",
        "params": ["delta=3"],
    },
    
    "bounded_degree_5": {
        "method": "random_bounded_degree",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_bounded_degree_5",
        "params": ["delta=5"],
    },
    
    # Drzewa z ograniczoną średnicą (bounded diameter)
    "bounded_diameter_6": {
        "method": "random_bounded_diameter",
        "n_min": 10,
        "n_max": 50,
        "instances": 10,
        "outdir": "generated_bounded_diameter_6",
        "params": ["D=6"],
    },
    
    "bounded_diameter_10": {
        "method": "random_bounded_diameter",
        "n_min": 10,
        "n_max": 50,
        "instances": 10,
        "outdir": "generated_bounded_diameter_10",
        "params": ["D=10"],
    },
    
    # Drzewa z dużym stopniem (large degree)
    "large_degree": {
        "method": "random_large_degree",
        "n_min": 20,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_large_degree",
        "params": ["delta=10"],
    },
    
    # Drzewa z dużą średnicą (large diameter)
    "large_diameter": {
        "method": "random_large_diameter",
        "n_min": 20,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_large_diameter",
        "params": ["D=15"],
    },
    
    # Drzewa pełne m-arne (full m-ary trees)
    "full_binary": {
        "method": "random_full_mary",
        "n_min": 7,
        "n_max": 63,
        "instances": 5,
        "outdir": "generated_full_binary",
        "params": ["m=2"],
    },
    
    "full_ternary": {
        "method": "random_full_mary",
        "n_min": 10,
        "n_max": 40,
        "instances": 5,
        "outdir": "generated_full_ternary",
        "params": ["m=3"],
    },
    
    # Drzewa zrównoważone (balanced trees)
    "balanced_binary": {
        "method": "random_balanced",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_balanced_binary",
        "params": ["m=2"],
    },
    
    # Miotły (brooms)
    "brooms": {
        "method": "random_broom",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_brooms",
        "params": [],
    },
    
    # Homary (lobsters)
    "lobsters": {
        "method": "random_lobster",
        "n_min": 15,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_lobsters",
        "params": [],
    },
    
    # Drzewa probabilistyczne
    "probabilistic": {
        "method": "random_probabilistic",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_probabilistic",
        "params": [],
    },
    
    # Drzewa rekurencyjne
    "recursive": {
        "method": "random_recursive",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_recursive",
        "params": [],
    },
    
    # Drzewa preferencyjne (preferential attachment)
    "preferential": {
        "method": "random_preferential",
        "n_min": 10,
        "n_max": 100,
        "instances": 10,
        "outdir": "generated_preferential",
        "params": [],
    },
    
    # Małe drzewa - wszystkie nieizomorficzne (do testów)
    "all_small_trees": {
        "method": "all_nonisomorphic_trees",
        "n_min": 4,
        "n_max": 10,
        "instances": 1,
        "outdir": "generated_all_small",
        "params": [],
    },
}


# Presety - grupy zbiorów danych
PRESETS = {
    "basic": [
        "random_trees",
        "stars",
        "caterpillars",
        "bounded_degree_3",
        "bounded_diameter_6",
    ],
    
    "special_structures": [
        "stars",
        "caterpillars",
        "spiders",
        "brooms",
        "lobsters",
    ],
    
    "bounded": [
        "bounded_degree_3",
        "bounded_degree_5",
        "bounded_diameter_6",
        "bounded_diameter_10",
        "large_degree",
        "large_diameter",
    ],
    
    "balanced": [
        "full_binary",
        "full_ternary",
        "balanced_binary",
    ],
    
    "random_models": [
        "probabilistic",
        "recursive",
        "preferential",
    ],
    
    "experimental": [
        "stars",
        "caterpillars",
        "spiders",
        "bounded_degree_3",
    ],
    
    "small": [
        "all_small_trees",
    ],
    
    "all": list(DATASETS.keys()),
}


# ============================================================
# FUNKCJE
# ============================================================

def run_generation(dataset_name: str, config: dict, base_seed: int, force: bool, fmt: str):
    """Uruchamia input_generation.py dla danego zbioru danych."""
    print(f"\n{'='*70}")
    print(f"🌳 Generating: {dataset_name}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable,
        "input_generation.py",
        "--method", config["method"],
        "--n-min", str(config["n_min"]),
        "--n-max", str(config["n_max"]),
        "--instances", str(config["instances"]),
        "--outdir", config["outdir"],
        "--format", fmt,
        "--seed", str(base_seed),
    ]
    
    if force:
        cmd.append("--force")
    
    if config.get("params"):
        cmd.extend(["--params"] + config["params"])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Success: {dataset_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {dataset_name}")
        print(f"Error: {e}")
        return False


def generate_summary(results: dict, output_file: Path):
    """Generuje podsumowanie wygenerowanych zbiorów danych."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Dataset Generation Summary\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Results\n\n")
        success_count = sum(1 for v in results.values() if v)
        f.write(f"- Total datasets: {len(results)}\n")
        f.write(f"- Successful: {success_count}\n")
        f.write(f"- Failed: {len(results) - success_count}\n\n")
        
        f.write(f"## Details\n\n")
        for name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            config = DATASETS[name]
            f.write(f"### {name} - {status}\n")
            f.write(f"- Method: {config['method']}\n")
            f.write(f"- Size range: {config['n_min']} - {config['n_max']}\n")
            f.write(f"- Instances per size: {config['instances']}\n")
            f.write(f"- Output dir: {config['outdir']}\n")
            if config.get('params'):
                f.write(f"- Parameters: {', '.join(config['params'])}\n")
            f.write(f"\n")
    
    print(f"\nSummary saved to: {output_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Zbiorczy generator zbiorów danych do testowania algorytmów",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    preset_help = "\n".join(f"  {name}: {', '.join(datasets)}" 
                           for name, datasets in PRESETS.items())
    
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()),
        help=f"Preset zbiorów danych do wygenerowania:\n{preset_help}"
    )
    
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Wybierz konkretne zbiory danych (zamiast presetu)"
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
        default="generation_summary.md",
        help="Nazwa pliku z podsumowaniem (default: generation_summary.md)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="Wyświetl dostępne zbiory danych i wyjdź"
    )
    
    args = parser.parse_args()
    
    # Wyświetl listę dostępnych zbiorów
    if args.list:
        print("\n📋 Dostępne zbiory danych:\n")
        for name, config in sorted(DATASETS.items()):
            print(f"  {name:25} - {config['method']:30} (n={config['n_min']}-{config['n_max']}, k={config['instances']})")
        
        print("\n📦 Dostępne presety:\n")
        for preset, datasets in PRESETS.items():
            print(f"  {preset:20} - {len(datasets)} zbiorów: {', '.join(datasets[:3])}...")
        
        return
    
    # Wybierz zbiory danych do wygenerowania
    if args.preset:
        datasets_to_generate = PRESETS[args.preset]
        print(f"\n🎯 Using preset: {args.preset}")
    elif args.datasets:
        datasets_to_generate = args.datasets
        print(f"\n🎯 Custom selection: {len(datasets_to_generate)} datasets")
    else:
        parser.error("Musisz podać --preset lub --datasets")
    
    # Informacje początkowe
    print(f"\n{'='*70}")
    print(f"🚀 Dataset Generation Started")
    print(f"{'='*70}")
    print(f"📊 Datasets to generate: {len(datasets_to_generate)}")
    print(f"🎲 Base seed: {args.seed}")
    print(f"📁 Format: {args.format}")
    print(f"🔄 Force overwrite: {args.force}")
    print(f"{'='*70}\n")
    
    # Generuj wszystkie wybrane zbiory
    results = {}
    for dataset_name in datasets_to_generate:
        config = DATASETS[dataset_name]
        success = run_generation(
            dataset_name=dataset_name,
            config=config,
            base_seed=args.seed,
            force=args.force,
            fmt=args.format
        )
        results[dataset_name] = success
    
    # Podsumowanie
    print(f"\n{'='*70}")
    print(f"📊 Final Summary")
    print(f"{'='*70}")
    
    success_count = sum(1 for v in results.values() if v)
    print(f"✅ Successful: {success_count}/{len(results)}")
    print(f"❌ Failed: {len(results) - success_count}/{len(results)}")
    
    # Zapisz podsumowanie
    summary_path = Path(args.summary)
    generate_summary(results, summary_path)
    
    print(f"\n✨ All done!\n")


if __name__ == "__main__":
    main()
