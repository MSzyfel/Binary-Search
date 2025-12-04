#!/usr/bin/env python3
"""
analyze_results.py

Analizuje wyniki z pliku CSV i oblicza średnie czasy dla każdego rozmiaru grafu.

Użycie:
    python analyze_results.py results.csv
    python analyze_results.py --all-datasets
    python analyze_results.py --directory data_set
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
                    
                    # Sprawdź, która kolumna z kosztem istnieje
                    cost = None
                    if 'SolutionCost(crit)' in row and row['SolutionCost(crit)']:
                        cost = float(row['SolutionCost(crit)'])
                    elif 'SolutionCost(worst)' in row and row['SolutionCost(worst)']:
                        cost = float(row['SolutionCost(worst)'])
                    
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
    Oblicza średnie i maksymalne wartości dla każdego rozmiaru.
    
    Returns:
        list: [(size, avg_cpu, max_cpu, avg_cost, max_cost, count), ...]
    """
    results = []
    
    for size in sorted(stats.keys()):
        data = stats[size]
        
        avg_cpu = sum(data['cpu_times']) / len(data['cpu_times']) if data['cpu_times'] else 0
        max_cpu = max(data['cpu_times']) if data['cpu_times'] else 0
        
        avg_cost = sum(data['costs']) / len(data['costs']) if data['costs'] else 0
        max_cost = max(data['costs']) if data['costs'] else 0
        
        count = data['count']
        
        results.append((size, avg_cpu, max_cpu, avg_cost, max_cost, count))
    
    return results


def print_table(results, title="Analysis Results"):
    """Wyświetla wyniki w ładnej tabeli."""
    print(f"\n{'='*100}")
    print(f"{title:^100}")
    print(f"{'='*100}")
    print(f"{'Size':<6} {'Avg CPU (s)':<13} {'Max CPU (s)':<13} {'Avg Cost':<12} {'Max Cost':<12} {'Count':<10}")
    print(f"{'-'*100}")
    
    for size, avg_cpu, max_cpu, avg_cost, max_cost, count in results:
        print(f"{size:<6} {avg_cpu:<13.4f} {max_cpu:<13.4f} {avg_cost:<12.2f} {max_cost:<12} {count:<10}")
    
    print(f"{'='*100}\n")


def save_to_csv(results, output_path):
    """Zapisuje wyniki do pliku CSV."""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Size', 'AvgCPUTime_s', 'MaxCPUTime_s', 'AvgCost', 'MaxCost', 'InstanceCount'])
            
            for size, avg_cpu, max_cpu, avg_cost, max_cost, count in results:
                writer.writerow([size, f"{avg_cpu:.6f}", f"{max_cpu:.6f}", f"{avg_cost:.2f}", int(max_cost), count])
        
        print(f"[OK] Results saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        return False


def find_all_result_files(base_dir):
    """Znajduje wszystkie pliki CSV w podfolderach."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"[ERROR] Directory not found: {base_dir}")
        return []
    
    result_files = []
    # Znajdź wszystkie pliki CSV (nie tylko results.csv)
    for csv_file in base_path.rglob("*.csv"):
        result_files.append(csv_file)
    
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
    parser.add_argument("--directory", "-d", type=str,
                        help="Analyze all results.csv files in specified directory")
    
    args = parser.parse_args()
    
    if args.all_datasets or args.directory:
        # Analizuj wszystkie pliki results.csv w podanym katalogu
        base_dir = args.directory if args.directory else "data_set"
        
        print(f"\n[INFO] Searching for results.csv files in {base_dir}/...")
        result_files = find_all_result_files(base_dir)
        
        if not result_files:
            print(f"[ERROR] No results.csv files found in {base_dir}/")
            return 1
        
        print(f"[INFO] Found {len(result_files)} result files\n")
        
        for result_file in result_files:
            # Użyj nazwy pliku bez rozszerzenia jako nazwy datasetu
            dataset_name = result_file.stem  # filename bez .csv
            
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
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_name = output_dir / f"analysis_{dataset_name}.csv"
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
