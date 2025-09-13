import networkx as nx
import random

from tree import Tree

def generate_random_tree(n, seed=None, distribution=None):
    """
    Generuje losowe drzewo o n wierzchołkach z losowym atrybutem 'c' ∈ (0, 1) dla każdego wierzchołka.

    Parametry:
    - n (int): liczba wierzchołków (n >= 1)
    - seed (int, optional): ziarno generatora losowego (dla deterministyczności)

    Zwraca:
    - G (networkx.Graph): nieskierowane drzewo z atrybutami 'c' przypisanymi do wierzchołków
    """
    # Ziarno dla obu generatorów
    if distribution is None:
        distribution = lambda: random.Random(seed).uniform(0, 1)

    # Generowanie drzewa
    t = nx.generators.random_labeled_tree(n=n, seed=seed)

    # Dodanie atrybutów 'c' ∈ (0, 1)
    for node in t.nodes():
        t.nodes[node]['c'] = distribution()

    return t
