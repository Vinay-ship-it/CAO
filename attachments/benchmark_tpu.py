import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (keep graph sizes identical to benchmark_cpu.py)
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_SIZES = [100, 200, 400, 600, 800, 1200]
RUNS        = 3
EDGE_PROB   = 0.3
INF         = 1e9
COLOUR_TPU  = "#3BB273"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_graph(n: int, edge_prob: float = 0.3, seed: int = 42) -> np.ndarray:
    """
    Random weighted directed graph as an N×N float32 adjacency matrix.
    Missing edges = INF, diagonal = 0, edge weights uniform in [1, 100].
    Same seed as the local script so graphs are identical across machines.
    """
    rng  = np.random.default_rng(seed)
    mask = rng.random((n, n)) < edge_prob
    w    = rng.uniform(1, 100, (n, n)).astype(np.float32)
    dist = np.where(mask, w, INF).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    return dist


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TPU — JAX Floyd-Warshall
# ─────────────────────────────────────────────────────────────────────────────

def floyd_warshall_tpu(dist: np.ndarray) -> np.ndarray:
    """
    Architecture mapping — TPU (JAX + XLA)
    ----------------------------------------
    jax.jit + lax.fori_loop compiles the entire k-loop into a single XLA
    program so the TPU sees the full algorithm as one fused graph.
    lax.dynamic_slice replaces d[k:k+1, :] because standard Python slice
    syntax does not accept dynamic (traced) indices inside JIT.
    jax.block_until_ready() waits for async TPU execution to finish
    before the timer stops.
    """
    import jax
    import jax.numpy as jnp

    n = dist.shape[0]

    @jax.jit
    def _run(d):
        def body_fn(k, d):
            row_k = jax.lax.dynamic_slice(d, (k, 0), (1, n))  # shape (1, n)
            col_k = jax.lax.dynamic_slice(d, (0, k), (n, 1))  # shape (n, 1)
            return jnp.minimum(d, col_k + row_k)
        return jax.lax.fori_loop(0, n, body_fn, d)

    d      = jnp.array(dist)
    result = _run(d)
    jax.block_until_ready(result)
    return np.array(result)


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
    print(f"  TPU Benchmark  |  runs = {RUNS}  |  density = {EDGE_PROB}")
    print("=" * 45)
    print(f"{'Size':>6}  {'TPU Time (s)':>14}")
    print("-" * 45)

    for n in GRAPH_SIZES:
        dist     = generate_graph(n, edge_prob=EDGE_PROB)
        tpu_time = benchmark_one(floyd_warshall_tpu, dist, runs=RUNS)
        records.append({"Graph Size": n, "TPU Time (s)": tpu_time})
        print(f"{n:>6}  {tpu_time:>14.4f}")

    print("=" * 45)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VISUALISATION  (Colab preview)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tpu_only(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#F7F9FB")

    ax.plot(df["Graph Size"], df["TPU Time (s)"], "-^",
            color=COLOUR_TPU, linewidth=2.2, markersize=8, label="TPU (JAX)")

    for _, row in df.iterrows():
        ax.annotate(f"{row['TPU Time (s)']:.3f}s",
                    xy=(row["Graph Size"], row["TPU Time (s)"]),
                    xytext=(5, 6), textcoords="offset points",
                    fontsize=8, color=COLOUR_TPU)

    ax.set_yscale("log")
    ax.set_xlabel("Graph Size N (nodes)", fontsize=12)
    ax.set_ylabel("Execution Time (seconds, log scale)", fontsize=12)
    ax.set_title("Floyd–Warshall: TPU Execution Time vs Graph Size",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("tpu_results.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import jax

    print("\n" + "=" * 45)
    print("  Floyd–Warshall  |  TPU Only  (Google Colab)")
    print(f"  Python {sys.version.split()[0]}  |  NumPy {np.__version__}")
    print("=" * 45)

    try:
        devices = jax.devices("tpu")
        print(f"[ENV] TPU devices : {devices} ✓")
    except Exception as e:
        print(f"[ERROR] No TPU found: {e}")
        print("        Go to: Runtime → Change runtime type → TPU v5e-1")
        return

    df = run_benchmarks()

    print("\n[DATA] TPU Results:")
    print(df.to_string(index=False))

    plot_tpu_only(df)

    df.to_csv("tpu_results.csv", index=False)

if __name__ == "__main__":
    main()
