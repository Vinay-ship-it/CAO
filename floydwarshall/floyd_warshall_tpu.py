"""
Floyd-Warshall  |  TPU ONLY  (Google Colab)
============================================
SETUP  -- run these two cells in Colab before executing this script:

  Cell 1 (install):
      !pip install "jax[tpu]" -f \
        https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
      !pip install numpy pandas matplotlib tabulate -q

  Cell 2 (confirm TPU is live):
      import jax; print(jax.devices())
      # should print: [TpuDevice(id=0, process_index=0, coords=(0,0,0), ...)]

  Runtime: Runtime -> Change runtime type -> TPU v5e-1

Outputs
    tpu_results.csv          <-- download this, used by combine_results.py
    tpu_heatmap_N10.png      <-- 4-panel heatmap for the 10-node demo
    tpu_benchmark.png        <-- benchmark chart
"""

import sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib, matplotlib.pyplot as plt

# Colab inline display (no-op when running outside a notebook)
try:
    from IPython.display import display as nb_display
    IN_NB = True
except ImportError:
    IN_NB = False
    def nb_display(*a, **kw): pass

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers=(), tablefmt="simple", **kw):
        ws = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in ws)
        lines = [fmt.format(*[str(h) for h in headers]),
                 "  ".join("-"*w for w in ws)]
        for row in rows:
            lines.append(fmt.format(*[str(v) for v in row]))
        return "\n".join(lines)

# JAX / TPU detection
try:
    import jax
    import jax.numpy as jnp
    TPU_DEVICES = jax.devices("tpu")
    HAS_TPU = True
    print(f"[ENV] TPU detected: {TPU_DEVICES}")
except Exception as e:
    HAS_TPU = False
    jax = None
    print(f"[ENV] No TPU found ({e})")
    print("      Go to: Runtime -> Change runtime type -> TPU v5e-1")

# ── constants ─────────────────────────────────────────────────────────────────
INF         = 1e9
GRAPH_SIZES = [10, 20, 50, 100]
RUNS        = 3
EDGE_PROB   = 0.5
SEED        = 42
AMBER       = "#E07B39"    # TPU colour throughout


# ==============================================================================
#  1. GRAPH GENERATION  (same seed as CPU script -> identical graphs)
# ==============================================================================

def generate_graph(n, edge_prob=EDGE_PROB, w_lo=1, w_hi=20, seed=SEED):
    """
    N x N float32 adjacency matrix for a random directed weighted graph.
    Uses the same seed as floyd_warshall_cpu.py so graphs are identical.
        adj[i][j] = edge weight  |  INF if no edge  |  0 on diagonal
    """
    rng  = np.random.default_rng(seed)
    mask = rng.random((n, n)) < edge_prob
    wt   = rng.integers(w_lo, w_hi + 1, (n, n)).astype(np.float32)
    adj  = np.where(mask, wt, INF).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


# ==============================================================================
#  2. FLOYD-WARSHALL  (TPU / JAX)
# ==============================================================================

def floyd_warshall_tpu(adj):
    """
    All-pairs shortest path compiled as a single XLA program via JAX.

    How it maps to TPU hardware
    ---------------------------
    jax.jit + lax.fori_loop compiles the entire N-iteration k-loop into one
    static XLA computation graph.  On every iteration the TPU's Matrix
    Multiply Unit (MXU) evaluates all N^2 cell updates simultaneously --
    this is the "parallel broadcast" that gives the TPU its advantage for
    large N.

    lax.dynamic_slice replaces adj[k:k+1, :] because Python slicing with a
    traced (dynamic) variable is not allowed inside jit.

    jax.block_until_ready() waits for async TPU execution to finish before
    the timer stops.

    Returns
    -------
    d_np    : final N x N distance matrix (numpy)
    elapsed : wall-clock seconds (dispatch + compute + host wait)
    """
    if not HAS_TPU:
        raise RuntimeError("No TPU available. Change runtime to TPU v5e-1.")

    n = adj.shape[0]

    @jax.jit
    def _run(d):
        def body(k, d):
            col_k = jax.lax.dynamic_slice(d, (0, k), (n, 1))   # (N, 1)
            row_k = jax.lax.dynamic_slice(d, (k, 0), (1, n))   # (1, N)
            return jnp.minimum(d, col_k + row_k)               # broadcast N,N
        return jax.lax.fori_loop(0, n, body, d)

    d_jax = jax.device_put(jnp.array(adj, dtype=jnp.float32))

    t0      = time.perf_counter()
    result  = _run(d_jax)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - t0

    return np.array(result, dtype=np.float64), elapsed


def _build_pred(adj):
    """
    Reconstruct predecessor matrix on CPU (used only for the small demo).
    The benchmark timing uses floyd_warshall_tpu() exclusively.
    """
    n    = adj.shape[0]
    pred = np.full((n, n), -1, dtype=np.int32)
    np.fill_diagonal(pred, np.arange(n))
    r, c = np.where(
        (adj < INF * 0.9) & (np.arange(n)[:, None] != np.arange(n)[None, :])
    )
    pred[r, c] = c
    # Replay Floyd-Warshall on CPU to get correct predecessor values
    d = adj.copy().astype(np.float64)
    for k in range(n):
        via = d[:, k:k+1] + d[k:k+1, :]
        ri, ci = np.where(via < d - 1e-9)
        if len(ri):
            pred[ri, ci] = pred[ri, k]
        d = np.minimum(d, via)
    return pred


# ==============================================================================
#  3. DISPLAY HELPERS
# ==============================================================================

def _dstr(v): return "inf" if v >= INF * 0.9 else f"{v:.0f}"
def _pstr(v): return "--"  if v < 0           else str(int(v))

def _print_matrix(mat, title, pred=False):
    n  = mat.shape[0]
    fn = _pstr if pred else _dstr
    hdr = "     " + "  ".join(f"[{j:2d}]" for j in range(n))
    print(f"\n  {title}")
    print("  " + "-" * len(hdr))
    print("  " + hdr)
    for i, row in enumerate(mat):
        print(f"  [{i:2d}]  " + " ".join(f"  {fn(v):>4s} " for v in row))

def print_table(mat, title, pred=False):
    n  = mat.shape[0]
    fn = _pstr if pred else _dstr
    df = pd.DataFrame(
        [[fn(mat[i][j]) for j in range(n)] for i in range(n)],
        index=[f"{i}>" for i in range(n)],
        columns=[f">{j}" for j in range(n)]
    )
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(df.to_string())


# ==============================================================================
#  4. HEATMAP ANALYSIS
# ==============================================================================

def draw_heatmap(adj, dist, pred, n, fname=None):
    """
    4-panel heatmap:
      A  Initial adjacency   B  Final distances
      C  Improvement delta   D  Predecessor matrix
    """
    def clip(m):
        c = m.astype(float).copy()
        c[c >= INF * 0.9] = np.nan
        return c

    d0     = clip(adj)
    df_m   = clip(dist)
    delta  = np.where(np.isnan(d0) | np.isnan(df_m), np.nan, d0 - df_m)
    pred_f = pred.astype(float)
    pred_f[pred_f < 0] = np.nan

    cm_hot = matplotlib.cm.get_cmap("YlOrRd").copy(); cm_hot.set_bad("#DDDDDD")
    cm_grn = matplotlib.cm.get_cmap("Greens").copy(); cm_grn.set_bad("#EEEEEE")
    cm_tab = matplotlib.cm.get_cmap("tab20b").copy(); cm_tab.set_bad("#EEEEEE")

    panels = [
        (d0,     cm_hot, "A  Initial Adjacency Matrix",     "Edge weight"),
        (df_m,   cm_hot, "B  Final Distance Matrix",        "Shortest path"),
        (delta,  cm_grn, "C  Improvement  (init - final)",  "Distance saved"),
        (pred_f, cm_tab, "D  Predecessor Matrix",           "Next-hop node"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor("#FAFAF8")
    fig.suptitle(f"Floyd-Warshall Matrix Analysis  |  TPU  |  N={n}",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (dat, cmap, title, clbl) in zip(axes.flat, panels):
        im = ax.imshow(dat, cmap=cmap, interpolation="nearest", aspect="auto")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(clbl, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Node j", fontsize=8)
        ax.set_ylabel("Node i", fontsize=8)
        ax.tick_params(labelsize=7)
        vmax = np.nanmax(dat) if not np.all(np.isnan(dat)) else 1
        for i in range(n):
            for j in range(n):
                v = dat[i][j]
                if not np.isnan(v):
                    ax.text(j, i, str(int(v)), ha="center", va="center",
                            fontsize=max(5, 9 - n // 4),
                            color="white" if v > vmax * 0.65 else "#333")

    plt.tight_layout()
    out = fname or f"tpu_heatmap_N{n}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    nb_display(fig)
    plt.show()
    print(f"[SAVED] {out}")


# ==============================================================================
#  5. PATH TABLE
# ==============================================================================

def _path(pred, src, dst):
    path, cur = [src], src
    while cur != dst:
        nxt = int(pred[cur][dst])
        if nxt < 0 or nxt == cur:
            return []
        path.append(nxt)
        cur = nxt
        if len(path) > len(pred):
            return []
    return path

def print_paths(dist, pred, max_rows=25):
    n, rows = dist.shape[0], []
    for src in range(n):
        for dst in range(n):
            if src == dst or dist[src][dst] >= INF * 0.9:
                continue
            p = _path(pred, src, dst)
            if p:
                rows.append([src, dst,
                             " > ".join(map(str, p)),
                             f"{dist[src][dst]:.0f}"])
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    print(f"\n{'='*60}\n  Shortest Paths  (first {len(rows)} shown)\n{'='*60}")
    print(tabulate(rows,
                   headers=["From", "To", "Path", "Distance"],
                   tablefmt="simple"))


# ==============================================================================
#  6. VERIFY  (compare TPU result against CPU reference)
# ==============================================================================

def verify(adj, d_tpu):
    """Check TPU output matches a CPU reference computation."""
    n  = adj.shape[0]
    d  = adj.copy().astype(np.float64)
    for k in range(n):
        d = np.minimum(d, d[:, k:k+1] + d[k:k+1, :])
    err = float(np.max(np.abs(d_tpu - d)))
    ok  = err < 1e-2
    print(f"  Verification vs CPU reference:  max |error| = {err:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ==============================================================================
#  7. BENCHMARK
# ==============================================================================

def benchmark():
    print(f"\n{'='*54}")
    print(f"  TPU Benchmark  |  runs={RUNS}  |  edge_prob={EDGE_PROB}")
    print(f"{'='*54}")
    print(f"  {'N':>5}  {'Mean Time (s)':>14}  {'Ops  N^3':>12}  {'Mops/s':>10}")
    print(f"  {'-'*50}")

    rows = []
    for n in GRAPH_SIZES:
        adj  = generate_graph(n, seed=SEED)
        ops  = n ** 3

        # One extra warm-up run to trigger JIT compilation
        floyd_warshall_tpu(adj)
        floyd_warshall_tpu(adj)

        times = []
        for _ in range(RUNS):
            _, t = floyd_warshall_tpu(adj)
            times.append(t)

        mean_t = float(np.mean(times))
        mops   = ops / mean_t / 1e6
        rows.append({
            "N":             n,
            "TPU Time (s)":  round(mean_t, 6),
            "Ops (N^3)":     ops,
            "TPU Mops/s":    round(mops, 2),
        })
        print(f"  {n:>5}  {mean_t:>14.6f}  {ops:>12,}  {mops:>10.2f}")

    print(f"  {'-'*50}")
    return pd.DataFrame(rows)


def plot_benchmark(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#FAFAF8")

    xs, ts, ms = (df["N"].tolist(),
                  df["TPU Time (s)"].tolist(),
                  df["TPU Mops/s"].tolist())

    axes[0].plot(xs, ts, "-^", color=AMBER, lw=2.2, ms=8)
    for x, t in zip(xs, ts):
        axes[0].annotate(f"{t:.4f}s", xy=(x, t), xytext=(4, 6),
                         textcoords="offset points", fontsize=8, color=AMBER)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Graph Size N", fontsize=11)
    axes[0].set_ylabel("Time (s, log scale)", fontsize=11)
    axes[0].set_title("TPU Execution Time vs N", fontsize=12, fontweight="bold")
    axes[0].grid(True, which="both", ls="--", alpha=0.4)

    axes[1].bar(xs, ms, color=AMBER, alpha=0.75, edgecolor="white",
                width=max(xs) // 10)
    axes[1].set_xlabel("Graph Size N", fontsize=11)
    axes[1].set_ylabel("Million ops / second", fontsize=11)
    axes[1].set_title("TPU Throughput (Mops/s)", fontsize=12, fontweight="bold")
    axes[1].grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("tpu_benchmark.png", dpi=140, bbox_inches="tight")
    nb_display(fig)
    plt.show()
    print("[SAVED] tpu_benchmark.png")


# ==============================================================================
#  8. DEMO
# ==============================================================================

def run_demo(n=10, edge_prob=EDGE_PROB, seed=SEED):
    sep = "=" * 68
    print(f"\n{sep}\n  Floyd-Warshall Demo  |  TPU  |  N={n}\n{sep}")

    print("\n[1/6]  Generating graph ...")
    adj    = generate_graph(n, edge_prob=edge_prob, seed=seed)
    nedges = int(np.sum((adj < INF * 0.9) & (adj > 0)))
    print(f"       nodes={n}  edges={nedges}  density={nedges/(n*(n-1)):.1%}")
    print_table(adj, "Initial Adjacency Matrix  (inf = no edge)")

    print("\n[2/6]  Running Floyd-Warshall on TPU ...")
    dist, elapsed = floyd_warshall_tpu(adj)
    print(f"       Done in {elapsed:.6f}s")

    print("\n[3/6]  Reconstructing predecessor matrix (CPU) ...")
    pred = _build_pred(adj)

    print("\n[4/6]  Final matrices ...")
    _print_matrix(dist, "Final Distance Matrix  d[i][j]")
    _print_matrix(pred, "Final Predecessor Matrix  pred[i][j]", pred=True)
    print_table(dist, "Final Distance Matrix  d[i][j]")
    print_table(pred, "Final Predecessor Matrix  pred[i][j]", pred=True)

    print("\n[5/6]  Heatmap analysis ...")
    draw_heatmap(adj, dist, pred, n, fname=f"tpu_heatmap_N{n}.png")

    print("\n[6/6]  Shortest paths + verification ...")
    print_paths(dist, pred, max_rows=30)
    verify(adj, dist)

    print(f"\n{sep}\n  Demo done  |  TPU elapsed: {elapsed:.6f}s\n{sep}\n")


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    if not HAS_TPU:
        print("\n[ERROR] No TPU detected. Cannot run this script.")
        print("        Runtime -> Change runtime type -> TPU v5e-1")
        return

    print(f"\n{'='*54}")
    print(f"  Floyd-Warshall  |  TPU Only")
    print(f"  Python {sys.version.split()[0]}  |  JAX {jax.__version__}")
    print(f"  Devices: {TPU_DEVICES}")
    print(f"{'='*54}")

    run_demo(n=10, edge_prob=0.5, seed=SEED)

    print(f"\n{'='*68}\n  BENCHMARK\n{'='*68}")
    df = benchmark()
    print(f"\n{df.to_string(index=False)}")
    plot_benchmark(df)
    df.to_csv("tpu_results.csv", index=False)
    print("\n[SAVED] tpu_results.csv")
    print("[NEXT]  Download tpu_results.csv,")
    print("        place it next to cpu_results.csv,")
    print("        then run:  py combine_results.py")


if __name__ == "__main__":
    main()
