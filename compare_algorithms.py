#!/usr/bin/env python3
"""
compare_algorithms.py

Porównuje koszty rozwiązań dwóch algorytmów (testowanego vs referencyjnego).
Dla każdej pary instancji oblicza współczynnik cost_algorithm / cost_reference,
a następnie dla każdego rozmiaru liczy średnią tych współczynników.

Użycie:
    python compare_algorithms.py algorithm_results.csv reference_results.csv
    python compare_algorithms.py algorithm_results.csv reference_results.csv --output comparison.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict
import sys


def load_results(csv_path):
    """
    Wczytuje wyniki z pliku CSV.
    
    Returns:
        dict: {filename: {'size': int, 'cost': float}}
    """
    results = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    filename = row['Filename']
                    num_nodes = int(row['NumNodes'])
                    cost_str = row['SolutionCost(crit)']
                    
                    if cost_str and cost_str.strip():
                        cost = float(cost_str)
                        results[filename] = {
                            'size': num_nodes,
                            'cost': cost
                        }
                    
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping row due to error: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    return results


def compute_ratios(algorithm_results, reference_results):
    """
    Oblicza współczynniki cost_algorithm / cost_reference dla wspólnych instancji.
    
    Returns:
        dict: {size: [ratio1, ratio2, ...]}
    """
    ratios_by_size = defaultdict(list)
    
    # Znajdź wspólne instancje
    common_files = set(algorithm_results.keys()) & set(reference_results.keys())
    
    if not common_files:
        print("[WARNING] No common instances found between the two result files!")
        return None
    
    print(f"[INFO] Found {len(common_files)} common instances")
    
    for filename in common_files:
        alg_data = algorithm_results[filename]
        ref_data = reference_results[filename]
        
        # Sprawdź czy rozmiary się zgadzają
        if alg_data['size'] != ref_data['size']:
            print(f"[WARNING] Size mismatch for {filename}: {alg_data['size']} vs {ref_data['size']}")
            continue
        
        # Oblicz współczynnik
        if ref_data['cost'] == 0:
            print(f"[WARNING] Reference cost is 0 for {filename}, skipping")
            continue
        
        ratio = alg_data['cost'] / ref_data['cost']
        ratios_by_size[alg_data['size']].append(ratio)
    
    return ratios_by_size


def compute_average_ratios(ratios_by_size):
    """
    Oblicza średnie współczynniki dla każdego rozmiaru.
    
    Returns:
        list: [(size, avg_ratio, count), ...]
    """
    results = []
    
    for size in sorted(ratios_by_size.keys()):
        ratios = ratios_by_size[size]
        avg_ratio = sum(ratios) / len(ratios)
        count = len(ratios)
        
        results.append((size, avg_ratio, count))
    
    return results


def print_table(results, title="Algorithm Comparison"):
    """Wyświetla wyniki w ładnej tabeli."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    print(f"{'Size':<10} {'Avg Ratio (Alg/Ref)':<30} {'Instances':<10}")
    print(f"{'-'*70}")
    
    for size, avg_ratio, count in results:
        print(f"{size:<10} {avg_ratio:<30.6f} {count:<10}")
    
    print(f"{'='*70}\n")
    
    # Statystyki globalne
    all_ratios = [avg_ratio for _, avg_ratio, _ in results]
    if all_ratios:
        global_avg = sum(all_ratios) / len(all_ratios)
        print(f"Global average ratio: {global_avg:.6f}")
        print(f"Min ratio: {min(all_ratios):.6f}")
        print(f"Max ratio: {max(all_ratios):.6f}")
    print()


def save_to_csv(results, output_path):
    """Zapisuje wyniki do pliku CSV."""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Size', 'AvgRatio', 'InstanceCount'])
            
            for size, avg_ratio, count in results:
                writer.writerow([size, f"{avg_ratio:.6f}", count])
        
        print(f"[OK] Comparison saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare algorithm costs against a reference algorithm.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("algorithm_csv", 
                        help="Path to the algorithm results CSV file")
    parser.add_argument("reference_csv",
                        help="Path to the reference algorithm results CSV file")
    parser.add_argument("--output", "-o", type=str,
                        help="Save comparison results to CSV file")
    
    args = parser.parse_args()
    
    # Wczytaj wyniki
    print(f"\n[INFO] Loading algorithm results from: {args.algorithm_csv}")
    algorithm_results = load_results(args.algorithm_csv)
    if algorithm_results is None:
        return 1
    print(f"[INFO] Loaded {len(algorithm_results)} algorithm results")
    
    print(f"[INFO] Loading reference results from: {args.reference_csv}")
    reference_results = load_results(args.reference_csv)
    if reference_results is None:
        return 1
    print(f"[INFO] Loaded {len(reference_results)} reference results")
    
    # Oblicz współczynniki
    ratios_by_size = compute_ratios(algorithm_results, reference_results)
    if ratios_by_size is None:
        return 1
    
    if not ratios_by_size:
        print("[ERROR] No valid ratios computed")
        return 1
    
    # Oblicz średnie
    results = compute_average_ratios(ratios_by_size)
    
    # Wyświetl wyniki
    alg_name = Path(args.algorithm_csv).stem
    ref_name = Path(args.reference_csv).stem
    print_table(results, title=f"Comparison: {alg_name} vs {ref_name}")
    
    # Zapisz do pliku
    if args.output:
        save_to_csv(results, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
