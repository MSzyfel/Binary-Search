# Szybka Generacja Zbiorów Danych

Skrypt `generate_datasets.py` automatyzuje generowanie wielu zbiorów danych testowych.

## Szybki Start

### 1. Zobacz dostępne opcje:
```bash
python generate_datasets.py --list
```

### 2. Wygeneruj podstawowe zbiory:
```bash
python generate_datasets.py --preset basic
```

### 3. Wygeneruj wszystkie zbiory:
```bash
python generate_datasets.py --preset all --seed 42
```

## Dostępne Presety

- **`basic`** - podstawowe zbiory do testów (5 typów)
- **`special_structures`** - specjalne struktury (stars, caterpillars, spiders, brooms, lobsters)
- **`bounded`** - drzewa z ograniczeniami (stopień, średnica)
- **`balanced`** - drzewa zrównoważone (binary, ternary)
- **`random_models`** - modele losowe (probabilistic, recursive, preferential)
- **`experimental`** - zbiory eksperymentalne (mniejsze rozmiary)
- **`small`** - małe drzewa (wszystkie nieizomorficzne)
- **`all`** - wszystkie zbiory

## Przykłady Użycia

### Wygeneruj konkretne zbiory:
```bash
python generate_datasets.py --datasets stars caterpillars bounded_degree_3
```

### Nadpisz istniejące pliki:
```bash
python generate_datasets.py --preset basic --force
```

### Użyj innego formatu:
```bash
python generate_datasets.py --preset basic --format gexf
```

### Zmień seed:
```bash
python generate_datasets.py --preset all --seed 12345
```

### Własne podsumowanie:
```bash
python generate_datasets.py --preset basic --summary my_summary.md
```

## Wygenerowane Struktury Katalogów

Po uruchomieniu skryptu zostaną utworzone katalogi:
```
generated_random_trees/
generated_stars/
generated_caterpillars/
generated_bounded_degree_3/
...
```

Każdy katalog zawiera:
- `*.graphml` - pliki grafów
- `*.meta.json` - metadane każdej instancji
- ewentualnie `results.csv` jeśli uruchomiono algorytmy

## Dostosowywanie

Aby dodać własny zbiór danych, edytuj `generate_datasets.py` i dodaj wpis do słownika `DATASETS`:

```python
"my_custom_dataset": {
    "method": "random_bounded_degree",
    "n_min": 20,
    "n_max": 50,
    "instances": 15,
    "outdir": "generated_my_custom",
    "params": ["delta=4"],
},
```

Następnie dodaj do odpowiedniego presetu lub użyj bezpośrednio:
```bash
python generate_datasets.py --datasets my_custom_dataset
```

## Pełny Workflow

```bash
# 1. Wygeneruj wszystkie zbiory danych
python generate_datasets.py --preset all

# 2. Uruchom algorytmy na wybranym zbiorze
python execute_algorithm.py --graph_dir generated_stars --algorithm ranking_based_dt --crit worst

# 3. Sprawdź wyniki
cat generated_stars/results.csv
```

## Parametry Generacji

Każdy zbiór może mieć następujące parametry:
- `method` - nazwa metody generującej z klasy Tree
- `n_min`, `n_max` - zakres liczby wierzchołków
- `instances` - liczba instancji dla każdego rozmiaru
- `outdir` - katalog wyjściowy
- `params` - dodatkowe parametry (np. `delta=3`, `D=6`, `legs=4`)

## Dostępne Metody Generacji

Zobacz plik `tree.py` dla pełnej dokumentacji. Główne metody:
- `random_labeled_tree` - losowe drzewa
- `random_star` - gwiazdy
- `random_caterpillar` - gąsienice
- `random_spider` - pająki
- `random_bounded_degree` - ograniczony stopień
- `random_bounded_diameter` - ograniczona średnica
- `random_full_mary` - pełne drzewa m-arne
- `random_balanced` - zrównoważone
- `random_broom` - miotły
- `random_lobster` - homary
- `random_probabilistic` - model probabilistyczny
- `random_recursive` - model rekurencyjny
- `random_preferential` - preferential attachment
- `all_nonisomorphic_trees` - wszystkie nieizomorficzne

## Porady

1. **Testuj najpierw na małych zbiorach**: Użyj presetu `experimental` przed `all`
2. **Zapisuj seed**: Użyj tego samego seeda dla powtarzalności
3. **Sprawdzaj podsumowanie**: Plik `generation_summary.md` zawiera szczegóły
4. **Nie nadpisuj bez potrzeby**: Domyślnie pliki nie są nadpisywane
