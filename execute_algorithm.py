import os
import time
import csv
import argparse
import networkx as nx

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
    "centroid_dt": lambda g: centroid_dt(Tree(g)),
    "average_case_trees_fptas": lambda g: average_case_trees_fptas(Tree(g)),
    "average_case_PTAS": lambda g: average_case_PTAS(Tree(g)),
    "average_case_FPTAS": lambda g: average_case_FPTAS(Tree(g)),
}

if ALGORITHM_NAME not in ALGORITHMS:
    raise ValueError(f"Algorytm {ALGORITHM_NAME} nie jest zdefiniowany w mapie ALGORITHMS")

algorithm_func = ALGORITHMS[ALGORITHM_NAME]

# ---------------- Przygotowanie pliku CSV ----------------
with open(OUTPUT_FILE, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Filename", "NumNodes", "NumEdges", "ExecutionTime_s", f"SolutionCost({CRIT})"])

    for filename in os.listdir(GRAPH_DIR):
        if not filename.endswith(".graphml"):
            continue

        filepath = os.path.join(GRAPH_DIR, filename)
        try:
            G = read_graph_as_int_nodes(filepath)
        except Exception as e:
            continue

        tree = Tree(G)

        start_time = time.time()
        dt = algorithm_func(G)
        end_time = time.time()
        exec_time = end_time - start_time

        solution_cost = dt.cost(crit=CRIT) if isinstance(dt, DecisionTree) else None

        writer.writerow([filename, G.number_of_nodes(), G.number_of_edges(), exec_time, solution_cost])
        print(f"Processed {filename}: time={exec_time:.4f}s, cost({CRIT})={solution_cost}")
