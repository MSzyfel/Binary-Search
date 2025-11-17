import os
import time
import csv
import argparse
import networkx as nx
import traceback

from algorithms import *  # Tutaj importuj wszystkie algorytmy
from tree import Tree
from decision_tree import DecisionTree

def read_graph_as_int_nodes(filepath):
    """Wczytuje graf i konwertuje wierzchołki na int"""
    if filepath.endswith(".graphml"):
        G = nx.read_graphml(filepath)
    else:
        raise ValueError(f"Unsupported format: {filepath}")

    mapping = {v: int(v) for v in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G

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
args = parser.parse_args()

GRAPH_DIR = args.graph_dir
ALGORITHM_NAME = args.algorithm
CRIT = args.crit
OUTPUT_FILE = os.path.join(GRAPH_DIR, "results.csv")

# ---------------- Mapowanie nazw do funkcji ----------------
ALGORITHMS = {
    "dp_tree": lambda g: dp_tree(Tree(g)),
    "ranking_based_dt": lambda g: ranking_based_dt(Tree(g)),
    "k_up_modularity_algorithm": lambda g: k_up_modularity_algorithm(Tree(g)),
    "qptas_dereniowski": lambda g: qptas_dereniowski_inspired(Tree(g)),
    "dereniowski_inspired": lambda g: dereniowski_inspired(Tree(g)),
    "cicalese_inspired": lambda g: cicalese_inspired(Tree(g)),
    "centroid_dt": lambda g: centroid_dt(Tree(g)),
    "average_case_trees_fptas": lambda g: average_case_trees_fptas(Tree(g)),
    "average_case_PTAS": lambda g: average_case_PTAS(Tree(g)),
    "average_case_FPTAS": lambda g: average_case_FPTAS(Tree(g)),
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
    
    # Wczytaj graf, aby uzyskać liczbę wierzchołków
    try:
        G = read_graph_as_int_nodes(filepath)
        num_nodes = G.number_of_nodes()
        files_to_process.append((filename, filepath, num_nodes))
    except Exception as e:
        print(f"Warning: Could not read {filename} for sorting: {e}")
        files_to_process.append((filename, filepath, float('inf')))  # Dodaj na koniec

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
            print(f"\nProcessing {filename} (n={num_nodes})...")
            G = read_graph_as_int_nodes(filepath)
            tree = Tree(G)

            start_wall_time = time.time()
            start_cpu_time = time.process_time()
            dt = algorithm_func(G)
            end_cpu_time = time.process_time()
            end_wall_time = time.time()
            
            wall_time = end_wall_time - start_wall_time
            cpu_time = end_cpu_time - start_cpu_time

            solution_cost = dt.cost(crit=CRIT) if isinstance(dt, DecisionTree) else None

            writer.writerow([filename, G.number_of_nodes(), G.number_of_edges(), wall_time, cpu_time, solution_cost, "SUCCESS"])
            csv_file.flush()  # Zapisz na dysk natychmiast
            
            mark_file_processed(filepath)
            print(f"✓ Completed {filename}: wall_time={wall_time:.4f}s, cpu_time={cpu_time:.4f}s, cost({CRIT})={solution_cost}")
            
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
