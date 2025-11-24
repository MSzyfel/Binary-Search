#!/usr/bin/env python3
"""
analyze_results.py

Analizuje wyniki z pliku CSV i oblicza średnie czasy dla każdego rozmiaru grafu.

Użycie:
    python analyze_results.py results.csv
    python analyze_results.py --all-datasets
    python analyze_results.py results.csv --output analysis.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict
import sys


def analyze_csv(csv_path):
    """
    Analizuje plik CSV i zwraca statystyki pogrupowane według rozmiaru.
    
    Returns:
        dict: {size: {'wall_times': [], 'cpu_times': [], 'costs': [], 'count': int}}
    """
    stats = defaultdict(lambda: {
        'wall_times': [],
        'cpu_times': [],
        'costs': [],
        'count': 0
    })
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    num_nodes = int(row['NumNodes'])
                    wall_time = float(row['WallTime_s'])
                    cpu_time = float(row['CPUTime_s'])
                    cost = float(row['SolutionCost(crit)']) if row['SolutionCost(crit)'] else None
                    
                    stats[num_nodes]['wall_times'].append(wall_time)
                    stats[num_nodes]['cpu_times'].append(cpu_time)
                    if cost is not None:
                        stats[num_nodes]['costs'].append(cost)
                    stats[num_nodes]['count'] += 1
                    
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping row due to error: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    return stats


def compute_averages(stats):
    """
    Oblicza średnie wartości dla każdego rozmiaru.
    
    Returns:
        list: [(size, avg_cpu, count), ...]
    """
    results = []
    
    for size in sorted(stats.keys()):
        data = stats[size]
        
        avg_cpu = sum(data['cpu_times']) / len(data['cpu_times']) if data['cpu_times'] else 0
        count = data['count']
        
        results.append((size, avg_cpu, count))
    
    return results


def print_table(results, title="Analysis Results"):
    """Wyświetla wyniki w ładnej tabeli."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    print(f"{'Size':<10} {'Avg CPU Time (s)':<25} {'Count':<10}")
    print(f"{'-'*60}")
    
    for size, avg_cpu, count in results:
        print(f"{size:<10} {avg_cpu:<25.4f} {count:<10}")
    
    print(f"{'='*60}\n")


def save_to_csv(results, output_path):
    """Zapisuje wyniki do pliku CSV."""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Size', 'AvgCPUTime_s', 'InstanceCount'])
            
            for size, avg_cpu, count in results:
                writer.writerow([size, f"{avg_cpu:.6f}", count])
        
        print(f"[OK] Results saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        return False


def find_all_result_files(base_dir="data_set"):
    """Znajduje wszystkie pliki results.csv w podfolderach."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    result_files = []
    for result_file in base_path.rglob("results.csv"):
        result_files.append(result_file)
    
    return sorted(result_files)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze algorithm execution results from CSV files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("csv_file", nargs='?', 
                        help="Path to the results CSV file")
    parser.add_argument("--output", "-o", type=str,
                        help="Save analysis results to CSV file")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Analyze all results.csv files in data_set/ subfolders")
    
    args = parser.parse_args()
    
    if args.all_datasets:
        # Analizuj wszystkie pliki results.csv w data_set
        print("\n[INFO] Searching for results.csv files in data_set/...")
        result_files = find_all_result_files()
        
        if not result_files:
            print("[ERROR] No results.csv files found in data_set/")
            return 1
        
        print(f"[INFO] Found {len(result_files)} result files\n")
        
        for result_file in result_files:
            dataset_name = result_file.parent.name
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"File: {result_file}")
            print(f"{'='*80}")
            
            stats = analyze_csv(result_file)
            if stats is None:
                continue
            
            if not stats:
                print("[WARNING] No data found in this file")
                continue
            
            results = compute_averages(stats)
            print_table(results, title=f"Analysis for {dataset_name}")
            
            if args.output:
                output_name = f"analysis_{dataset_name}.csv"
                save_to_csv(results, output_name)
    
    elif args.csv_file:
        # Analizuj pojedynczy plik
        csv_path = Path(args.csv_file)
        
        if not csv_path.exists():
            print(f"[ERROR] File not found: {csv_path}")
            return 1
        
        print(f"\n[INFO] Analyzing: {csv_path}")
        
        stats = analyze_csv(csv_path)
        if stats is None:
            return 1
        
        if not stats:
            print("[WARNING] No data found in the file")
            return 1
        
        results = compute_averages(stats)
        print_table(results)
        
        if args.output:
            save_to_csv(results, args.output)
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
