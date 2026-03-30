import time
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_SIZES = [100, 200, 400, 600, 800, 1200]
RUNS        = 3
EDGE_PROB   = 0.3
INF         = 1e9
COLOUR_CPU  = "#2E86AB"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_graph(n: int, edge_prob: float = 0.3, seed: int = 42) -> np.ndarray:
    """
    Random weighted directed graph as an N×N float32 adjacency matrix.
    Missing edges = INF, diagonal = 0, edge weights uniform in [1, 100].
    """
    rng  = np.random.default_rng(seed)
    mask = rng.random((n, n)) < edge_prob
    w    = rng.uniform(1, 100, (n, n)).astype(np.float32)
    dist = np.where(mask, w, INF).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    return dist


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CPU — NumPy Floyd-Warshall
# ─────────────────────────────────────────────────────────────────────────────

def floyd_warshall_cpu(dist: np.ndarray) -> np.ndarray:
    """
    Architecture mapping — CPU
    --------------------------
    All computation stays on the host CPU. NumPy collapses the two inner
    loops (i, j) into a single broadcasted minimum using SIMD/AVX, so each
    outer iteration k is O(N²) with a low constant.
    """
    d = dist.copy()
    n = d.shape[0]
    for k in range(n):
        d = np.minimum(d, d[:, k:k+1] + d[k:k+1, :])
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BENCHMARKING
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(fn, dist: np.ndarray, runs: int = 3) -> float:
    """
    Run fn(dist) runs+1 times, discard the first as warm-up,
    return the mean wall-clock time in seconds via time.perf_counter().
    """
    times = []
    for i in range(runs + 1):
        t0 = time.perf_counter()
        fn(dist)
        t1 = time.perf_counter()
        if i > 0:
            times.append(t1 - t0)
    return float(np.mean(times))


def run_benchmarks() -> pd.DataFrame:
    records = []

    print("\n" + "=" * 45)
    print(f"  CPU Benchmark  |  runs = {RUNS}  |  density = {EDGE_PROB}")
    print("=" * 45)
    print(f"{'Size':>6}  {'CPU Time (s)':>14}")
    print("-" * 45)

    for n in GRAPH_SIZES:
        dist     = generate_graph(n, edge_prob=EDGE_PROB)
        cpu_time = benchmark_one(floyd_warshall_cpu, dist, runs=RUNS)
        records.append({"Graph Size": n, "CPU Time (s)": cpu_time})
        print(f"{n:>6}  {cpu_time:>14.4f}")

    print("=" * 45)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_cpu_only(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#F7F9FB")

    ax.plot(df["Graph Size"], df["CPU Time (s)"], "-o",
            color=COLOUR_CPU, linewidth=2.2, markersize=8, label="CPU (NumPy)")

    for _, row in df.iterrows():
        ax.annotate(f"{row['CPU Time (s)']:.3f}s",
                    xy=(row["Graph Size"], row["CPU Time (s)"]),
                    xytext=(5, 6), textcoords="offset points",
                    fontsize=8, color=COLOUR_CPU)

    ax.set_yscale("log")
    ax.set_xlabel("Graph Size N (nodes)", fontsize=12)
    ax.set_ylabel("Execution Time (seconds, log scale)", fontsize=12)
    ax.set_title("Floyd–Warshall: CPU Execution Time vs Graph Size",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("cpu_results.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 45)
    print("  Floyd–Warshall Benchmark  |  CPU Only")
    print(f"  Python {sys.version.split()[0]}  |  NumPy {np.__version__}")
    print(f"  Platform: {platform.processor()}")
    print("=" * 45)

    df = run_benchmarks()

    print("\n[DATA] Results:")
    print(df.to_string(index=False))

    plot_cpu_only(df)

    df.to_csv("cpu_results.csv", index=False)

if __name__ == "__main__":
    main()
