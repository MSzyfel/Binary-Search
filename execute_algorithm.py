import os
import time
import csv
import argparse
import networkx as nx
import traceback

from algorithms import *  # Tutaj importuj wszystkie algorytmy
from tree import Tree
from decision_tree import DecisionTree

def read_graph_as_tree(filepath):
    """Wczytuje graf bezpośrednio jako Tree z poprawnymi atrybutami"""
    if not filepath.endswith(".graphml"):
        raise ValueError(f"Unsupported format: {filepath}")
    
    # Wczytaj jako NetworkX
    G = nx.read_graphml(filepath)
    
    # Konwertuj nazwy węzłów na int
    mapping = {v: int(v) for v in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    
    # WAŻNE: Konwertuj atrybuty c i w na float PRZED przekazaniem do Tree
    for node in G.nodes():
        if 'c' in G.nodes[node]:
            G.nodes[node]['c'] = float(G.nodes[node]['c'])
        if 'w' in G.nodes[node]:
            G.nodes[node]['w'] = float(G.nodes[node]['w'])
    
    # Stwórz Tree z grafu NetworkX (Tree skopiuje atrybuty)
    return Tree(G)

def is_file_processed(filepath):
    """Sprawdza czy plik został już przetworzony"""
    processed_marker = filepath + ".processed"
    error_marker = filepath + ".error"
    return os.path.exists(processed_marker) or os.path.exists(error_marker)

def mark_file_processed(filepath):
    """Oznacza plik jako pomyślnie przetworzony"""
    marker = filepath + ".processed"
    with open(marker, 'w') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

def mark_file_error(filepath, error_msg):
    """Oznacza plik jako zawierający błąd"""
    marker = filepath + ".error"
    with open(marker, 'w') as f:
        f.write(f"Error occurred at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Error: {error_msg}\n")

# ---------------- Argumenty wiersza poleceń ----------------
parser = argparse.ArgumentParser(description="Uruchamianie algorytmów na grafach w katalogu")
parser.add_argument("--graph_dir", type=str, default="generated_trees_d6", help="Katalog z grafami")
parser.add_argument("--algorithm", type=str, default="ranking_based_dt", help="Algorytm do uruchomienia")
parser.add_argument("--crit", type=str, default="worst", choices=["worst", "average"], help="Kryterium minimalizacji kosztu")
parser.add_argument("--force", action="store_true", help="Usuń markery .processed/.error i przelicz wszystkie instancje od nowa")
args = parser.parse_args()

GRAPH_DIR = args.graph_dir
ALGORITHM_NAME = args.algorithm
CRIT = args.crit
OUTPUT_FILE = os.path.join(GRAPH_DIR, "results.csv")

# Jeśli --force, usuń wszystkie markery i plik wynikowy
if args.force:
    print(f"\n{'='*60}")
    print(f"FORCE MODE: Usuwam wszystkie markery i wyniki")
    print(f"{'='*60}")
    
    # Usuń plik CSV z wynikami
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"[OK] Usunięto {OUTPUT_FILE}")
    
    # Usuń wszystkie markery .processed i .error
    markers_removed = 0
    for filename in os.listdir(GRAPH_DIR):
        filepath = os.path.join(GRAPH_DIR, filename)
        if filename.endswith(".processed") or filename.endswith(".error"):
            os.remove(filepath)
            markers_removed += 1

    print(f"[OK] Usunięto {markers_removed} markerów")
    print(f"{'='*60}\n")

# ---------------- Mapowanie nazw do funkcji ----------------
ALGORITHMS = {
    "dp_tree": lambda tree: dp_tree(tree),
    "ranking_based_dt": lambda tree: ranking_based_dt(tree),
    "k_up_modularity_algorithm": lambda tree: k_up_modularity_algorithm(tree),
    "qptas_dereniowski": lambda tree: qptas_dereniowski_inspired(tree),
    "dereniowski_inspired": lambda tree: dereniowski_inspired(tree),
    "cicalese_inspired": lambda tree: cicalese_inspired(tree),
    "centroid_dt": lambda tree: centroid_dt(tree),
    "average_case_trees_fptas": lambda tree: average_case_trees_fptas(tree),
    "average_case_PTAS": lambda tree: average_case_PTAS(tree),
    "average_case_FPTAS": lambda tree: average_case_FPTAS(tree),
}

if ALGORITHM_NAME not in ALGORITHMS:
    raise ValueError(f"Algorytm {ALGORITHM_NAME} nie jest zdefiniowany w mapie ALGORITHMS")

algorithm_func = ALGORITHMS[ALGORITHM_NAME]

# ---------------- Zbierz pliki do przetworzenia ----------------
files_to_process = []
for filename in os.listdir(GRAPH_DIR):
    if not filename.endswith(".graphml"):
        continue
    filepath = os.path.join(GRAPH_DIR, filename)
    if is_file_processed(filepath):
        print(f"Skipping {filename} (already processed)")
        continue
    
    # Wyciągnij rozmiar z nazwy pliku (np. random_star_n10_inst0.graphml -> 10)
    try:
        import re
        match = re.search(r'_n(\d+)_', filename)
        num_nodes = int(match.group(1)) if match else float('inf')
        files_to_process.append((filename, filepath, num_nodes))
    except Exception as e:
        print(f"Warning: Could not parse size from {filename}: {e}")
        files_to_process.append((filename, filepath, float('inf')))

# Sortuj pliki według liczby wierzchołków (najmniejsze najpierw)
files_to_process.sort(key=lambda x: x[2])

print(f"\nFound {len(files_to_process)} files to process (sorted by number of nodes, smallest first)")

# ---------------- Przygotowanie pliku CSV ----------------
file_exists = os.path.exists(OUTPUT_FILE)
with open(OUTPUT_FILE, mode="a", newline="") as csv_file:
    writer = csv.writer(csv_file)
    
    # Zapisz nagłówek tylko jeśli plik nie istnieje
    if not file_exists:
        writer.writerow(["Filename", "NumNodes", "NumEdges", "WallTime_s", "CPUTime_s", f"SolutionCost({CRIT})", "Status"])

    for filename, filepath, num_nodes in files_to_process:
        try:
            print(f"\n[DEBUG] Processing {filename} (n={num_nodes})...")
            print(f"[DEBUG] Reading graph from {filepath}")
            tree_obj = read_graph_as_tree(filepath)
            print(f"[DEBUG] Tree loaded: {tree_obj.n} nodes")
            print(f"[DEBUG] Starting algorithm...")

            start_wall_time = time.time()
            start_cpu_time = time.process_time()
            dt = algorithm_func(tree_obj)
            end_cpu_time = time.process_time()
            end_wall_time = time.time()
            
            print(f"[DEBUG] Algorithm finished!")
            
            wall_time = end_wall_time - start_wall_time
            cpu_time = end_cpu_time - start_cpu_time

            solution_cost = dt.cost(crit=CRIT) if isinstance(dt, DecisionTree) else None

            writer.writerow([filename, tree_obj.n, tree_obj.n - 1, wall_time, cpu_time, solution_cost, "SUCCESS"])
            csv_file.flush()  # Zapisz na dysk natychmiast
            
            mark_file_processed(filepath)
            print(f"[OK] Completed {filename}: wall_time={wall_time:.4f}s, cpu_time={cpu_time:.4f}s, cost({CRIT})={solution_cost}")
            
        except Exception as e:
            error_msg = str(e)
            trace = traceback.format_exc()
            print(f"✗ Error processing {filename}: {error_msg}")
            print(f"Traceback:\n{trace}")
            
            writer.writerow([filename, None, None, None, None, f"ERROR: {error_msg}"])
            csv_file.flush()
            
            mark_file_error(filepath, trace)

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Results saved to: {OUTPUT_FILE}")
print(f"{'='*60}")
