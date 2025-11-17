from math import log2, sqrt
import time
import matplotlib.pyplot as plt
import networkx as nx

from tree import Tree
from algorithms import average_case_FPTAS, average_case_PTAS, tree_search_cicalese_inspired, qptas_dereniowski_inspired  # w razie potrzeby dodaj inne algorytmy

# Lista n od 1 do 7
n_values = list(range(20, 21))
execution_times = []

for n in n_values:
    # Tworzymy drzewo
    T = Tree(n=n, seed=2)
    #T.draw(attribute='c', type=float)
    print(n)

    # Mierzymy czas działania algorytmu
    start_time = time.perf_counter()

    t = int(2 ** (sqrt(log2(len(T)))))
    dt1 = tree_search_cicalese_inspired(T, t, qptas_dereniowski_inspired)

    end_time = time.perf_counter()

    execution_times.append(end_time - start_time)


plt.figure(figsize=(8, 5))
plt.plot(n_values, execution_times, marker='o', linestyle='-', color='blue')
plt.title("Czas działania average_case_FPTAS dla różnych n")
plt.xlabel("Liczba węzłów n")
plt.ylabel("Czas wykonania [s]")
plt.grid(True)
plt.show()
