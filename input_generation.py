#!/usr/bin/env python3
"""
input_generation.py

Tree generator using class methods Tree.*(...).
Saves generated trees in graph formats:
 - graphml (.graphml)
 - gexf    (.gexf)
 - gpickle (.gpickle)
 - json    (node-link, .json)

Requires: networkx, numpy.

Make sure to import your Tree class from the correct module.
"""
import argparse
import inspect
import json
from pathlib import Path
from datetime import datetime
import networkx as nx
import numpy as np

# === IMPORT YOUR TREE CLASS ===
from tree import Tree
# ==============================


# ---------- DISTRIBUTIONS ----------

def get_distributions():
    """Return available random distributions and their descriptions."""
    rng = np.random.default_rng

    distributions = {
        "uniform": lambda seed=None: (lambda size=None, low=0.0, high=1.0: rng(seed).uniform(low=low, high=high, size=size)),
        "normal": lambda seed=None: (lambda size=None, mean=0.0, std=1.0: rng(seed).normal(loc=mean, scale=std, size=size)),
        "exponential": lambda seed=None: (lambda size=None, scale=1.0: rng(seed).exponential(scale=scale, size=size)),
        "constant": lambda seed=None: (lambda size=None, value=1.0: np.full(size, value)),
    }

    descriptions = {
        "uniform": "Uniform(0, 1) distribution",
        "normal": "Normal(0, 1) distribution",
        "exponential": "Exponential(scale=1.0) distribution",
        "constant": "Constant value distribution (all weights equal)",
    }

    return distributions, descriptions


# ---------- HELPER FUNCTIONS ----------

def available_methods():
    """Return a mapping of available Tree generation methods."""
    methods = {}
    for name, member in inspect.getmembers(Tree):
        if callable(member) and (
            name.startswith("random_")
            or name in ("all_nonisomorphic_trees", "enumerate_attribute_combinations")
        ):
            methods[name] = member
    return methods


def _save_graph(G: nx.Graph, outpath: Path, fmt: str):
    """Save a NetworkX graph to file."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "graphml":
        nx.write_graphml(G, str(outpath))
    elif fmt == "gexf":
        nx.write_gexf(G, str(outpath))
    elif fmt == "gpickle":
        nx.write_gpickle(G, str(outpath))
    elif fmt == "json":
        data = nx.readwrite.json_graph.node_link_data(G)
        outpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unknown format: {fmt}")


def tree_to_networkx(tree):
    """Convert Tree object to NetworkX graph if necessary."""
    if isinstance(tree, (nx.Graph, nx.DiGraph)):
        return tree
    if hasattr(tree, "to_networkx"):
        G = tree.to_networkx()
        if not isinstance(G, (nx.Graph, nx.DiGraph)):
            raise TypeError("to_networkx() did not return a networkx graph")
        return G
    try:
        return nx.DiGraph(tree)
    except Exception as e:
        raise TypeError("Cannot convert tree to networkx graph.") from e


def save_metadata(meta: dict, outdir: Path, basename: str):
    """Save metadata JSON for each instance."""
    meta_path = outdir / f"{basename}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def parse_kv_params(param_list):
    """Parse --params key=value pairs into a dict with type inference."""
    out = {}
    if not param_list:
        return out
    for token in param_list:
        if "=" not in token:
            raise ValueError(f"Invalid parameter: {token} (expected key=value)")
        k, v = token.split("=", 1)
        if v.lower() == "none":
            parsed = None
        else:
            try:
                parsed = int(v)
            except ValueError:
                try:
                    parsed = float(v)
                except ValueError:
                    parsed = v
        out[k] = parsed
    return out


# ---------- GENERATION ----------

def generate_and_save(method_name: str, method_callable, n: int, instances: int,
                      outdir: Path, fmt: str, extra_kwargs: dict, seed_base: int | None, force: bool):
    """Generate and save instances for given method and size."""
    saved = 0
    for i in range(instances):
        seed = (seed_base * 1000003 + n * 1009 + i) % (2**31 - 1) if seed_base is not None else None
        kwargs_local = dict(extra_kwargs, seed=seed) if seed is not None else dict(extra_kwargs)
        basename = f"{method_name}_n{n}_inst{i}"
        outpath = outdir / f"{basename}.{fmt if fmt != 'json' else 'json'}"

        if outpath.exists() and not force:
            print(f"[SKIP] {outpath} exists (use --force to overwrite).")
            continue

        try:
            t = method_callable(n=n, **kwargs_local)
            G = tree_to_networkx(t)
            _save_graph(G, outpath, fmt)

            meta = {
                "method": method_name,
                "n": n,
                "instance": i,
                "params": kwargs_local,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            save_metadata(meta, outdir, basename)
            saved += 1
        except Exception as e:
            print(f"[WARN] Failed to generate/save (n={n}, inst={i}): {e}")

    return saved


# ---------- CLI ----------

def main():
    methods = available_methods()
    distributions, distribution_docs = get_distributions()

    methods_list = "\n".join(f"  - {name}" for name in sorted(methods.keys())) or "  (No generation methods found)"
    dist_list = "\n".join(f"  - {name}: {desc}" for name, desc in distribution_docs.items())

    parser = argparse.ArgumentParser(
        description=(
            "Tree instance generator using class methods from Tree.\n\n"
            "Available generation methods:\n"
            f"{methods_list}\n\n"
            "Available distributions:\n"
            f"{dist_list}\n\n"
            "Each method may accept additional parameters via --params key=value."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--method", "-m", required=True, choices=sorted(methods.keys()),
                        help="Name of the generation method (one from the list above).")
    parser.add_argument("--n-min", type=int, required=True, help="Minimum number of nodes (>=1).")
    parser.add_argument("--n-max", type=int, required=True, help="Maximum number of nodes (>= n-min).")
    parser.add_argument("--instances", "-k", type=int, default=1,
                        help="Number of instances per size.")
    parser.add_argument("--outdir", "-o", type=str, default="generated_graphs",
                        help="Output directory (default: generated_graphs/).")
    parser.add_argument("--format", "-f", choices=["graphml", "gexf", "gpickle", "json"],
                        default="graphml", help="Graph output format (default: graphml).")
    parser.add_argument("--params", "-p", nargs="*", help="Extra parameters in key=value format (e.g. delta=3 D=4).")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (optional).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files if they exist.")
    parser.add_argument("--distribution-c", choices=distributions.keys(),
                        help="Distribution used for node costs.")
    parser.add_argument("--distribution-w", choices=distributions.keys(),
                        help="Distribution used for edge weights.")

    args = parser.parse_args()

    # Parse parameters
    params = parse_kv_params(args.params)

    # Map distributions to callables
    if args.distribution_c:
        params["distribution_c"] = get_distributions()[0][args.distribution_c](args.seed)
    if args.distribution_w:
        params["distribution_w"] = get_distributions()[0][args.distribution_w](args.seed)

    method_callable = methods[args.method]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    total = 0
    for n in range(args.n_min, args.n_max + 1):
        count = generate_and_save(
            method_name=args.method,
            method_callable=method_callable,
            n=n,
            instances=args.instances,
            outdir=outdir,
            fmt=args.format,
            extra_kwargs=params,
            seed_base=args.seed,
            force=args.force,
        )
        total += count

    print(f"\n✅ Done. Generated {total} graphs in {outdir.resolve()}\n")


if __name__ == "__main__":
    main()
