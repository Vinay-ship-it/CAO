"""
Floyd-Warshall  |  CPU ONLY
============================
Requirements
    py -m pip install numpy pandas matplotlib networkx tabulate

Run
    py floyd_warshall_cpu.py

Outputs
    cpu_results.csv          <-- benchmark timings (used by combine_results.py)
    cpu_graph_N10.png        <-- networkx graph + shortest-path overlay
    cpu_heatmap_N10.png      <-- 4-panel heatmap analysis
    cpu_animation_N10.gif    <-- animated matrix evolution
    cpu_benchmark.png        <-- benchmark chart
"""

import sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib, matplotlib.pyplot as plt
import matplotlib.animation as anim_mod

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[WARN] networkx not installed  --  graph drawing skipped")

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

# ── constants ─────────────────────────────────────────────────────────────────
INF         = 1e9          # sentinel for "no edge"
GRAPH_SIZES = [10, 20, 50, 100]
RUNS        = 3            # timed runs per graph size (+ 1 warm-up discarded)
EDGE_PROB   = 0.5
SEED        = 42
BLUE        = "#2E86AB"    # CPU colour throughout


# ==============================================================================
#  1. GRAPH GENERATION
# ==============================================================================

def generate_graph(n, edge_prob=EDGE_PROB, w_lo=1, w_hi=20, seed=SEED):
    """
    Return an N x N float32 adjacency matrix for a random directed
    weighted graph.
        adj[i][j] = edge weight   (i != j, edge exists)
        adj[i][j] = INF           (no edge)
        adj[i][i] = 0             (diagonal)
    """
    rng  = np.random.default_rng(seed)
    mask = rng.random((n, n)) < edge_prob
    wt   = rng.integers(w_lo, w_hi + 1, (n, n)).astype(np.float32)
    adj  = np.where(mask, wt, INF).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


# ==============================================================================
#  2. FLOYD-WARSHALL  (CPU / NumPy)
# ==============================================================================

def _build_pred(adj):
    """
    Initial predecessor (next-hop) matrix.
        pred[i][j] = j    direct edge i -> j exists
        pred[i][j] = i    self (diagonal)
        pred[i][j] = -1   no path known yet
    """
    n    = adj.shape[0]
    pred = np.full((n, n), -1, dtype=np.int32)
    np.fill_diagonal(pred, np.arange(n))
    r, c = np.where(
        (adj < INF * 0.9) & (np.arange(n)[:, None] != np.arange(n)[None, :])
    )
    pred[r, c] = c
    return pred


def floyd_warshall_cpu(adj, verbose=False, snapshots=False):
    """
    All-pairs shortest path on CPU using NumPy.

    For each relay node k = 0 .. N-1:
        via[i,j] = d[i,k] + d[k,j]          (broadcast: N,1 + 1,N -> N,N)
        where via[i,j] < d[i,j]:
            d[i,j]    <- via[i,j]
            pred[i,j] <- pred[i,k]

    Complexity: O(N^3) time, O(N^2) space.

    Returns
    -------
    d        : final N x N distance matrix
    pred     : final N x N predecessor matrix
    elapsed  : wall-clock seconds
    snaps    : list of (k, d_copy, pred_copy) when snapshots=True
    """
    n    = adj.shape[0]
    d    = adj.copy().astype(np.float64)
    pred = _build_pred(adj)
    snaps = []

    t0 = time.perf_counter()

    for k in range(n):
        via      = d[:, k:k+1] + d[k:k+1, :]   # (N,1) + (1,N) -> (N,N)
        improved = via < d - 1e-9
        ri, ci   = np.where(improved)
        if len(ri):
            pred[ri, ci] = pred[ri, k]
        d = np.minimum(d, via)

        if verbose:
            _print_k(k, n, d, pred)
        if snapshots:
            snaps.append((k, d.copy(), pred.copy()))

    return d, pred, time.perf_counter() - t0, snaps


# ==============================================================================
#  3. DISPLAY HELPERS
# ==============================================================================

def _dstr(v):  return "inf" if v >= INF * 0.9 else f"{v:.0f}"
def _pstr(v):  return "--"  if v < 0           else str(int(v))

def _print_matrix(mat, title, pred=False):
    n   = mat.shape[0]
    fn  = _pstr if pred else _dstr
    hdr = "     " + "  ".join(f"[{j:2d}]" for j in range(n))
    print(f"\n  {title}")
    print("  " + "-" * len(hdr))
    print("  " + hdr)
    for i, row in enumerate(mat):
        print(f"  [{i:2d}]  " + " ".join(f"  {fn(v):>4s} " for v in row))

def _print_k(k, n, d, pred):
    print(f"\n  {'='*66}")
    print(f"  Iteration k = {k}   relay node {k}   ({k+1}/{n} done)")
    print(f"  {'='*66}")
    _print_matrix(d,    "Distance Matrix    d[i][j]")
    _print_matrix(pred, "Predecessor Matrix pred[i][j]", pred=True)

def print_table(mat, title, pred=False):
    n   = mat.shape[0]
    fn  = _pstr if pred else _dstr
    df  = pd.DataFrame(
        [[fn(mat[i][j]) for j in range(n)] for i in range(n)],
        index=[f"{i}>" for i in range(n)],
        columns=[f">{j}" for j in range(n)]
    )
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(df.to_string())


# ==============================================================================
#  4. NETWORKX GRAPH VISUALIZATION
# ==============================================================================

def _path(pred, src, dst):
    """Reconstruct shortest path src -> dst from predecessor matrix."""
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

def draw_graph(adj, dist=None, pred=None, src=0, n=10, fname=None):
    """
    Left panel  : raw directed graph with edge weights
    Right panel : same graph with shortest-path tree from node `src` highlighted
                  (only shown when dist and pred are provided)
    """
    if not HAS_NX:
        return

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    edges, elabels = [], {}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] < INF * 0.9:
                G.add_edge(i, j)
                edges.append((i, j))
                elabels[(i, j)] = int(adj[i][j])

    sp = set()
    if dist is not None and pred is not None:
        for dst in range(n):
            if dst == src or dist[src][dst] >= INF * 0.9:
                continue
            p = _path(pred, src, dst)
            for a, b in zip(p[:-1], p[1:]):
                sp.add((a, b))

    pos   = nx.circular_layout(G)
    ncols = 2 if dist is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    fig.patch.set_facecolor("#F8F8F6")
    if ncols == 1:
        axes = [axes]

    for ax, show_sp in zip(axes, [False, True][:ncols]):
        nc   = [BLUE] * n if not show_sp else \
               ["#E07B39" if i == src else BLUE for i in range(n)]
        base = [(i, j) for (i, j) in edges if not show_sp or (i, j) not in sp]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc,
                               node_size=600, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_color="white",
                                font_size=9, font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=base,
                               edge_color="#CCCCCC", arrows=True,
                               arrowsize=12, width=1.2, alpha=0.5,
                               connectionstyle="arc3,rad=0.08")
        if show_sp and sp:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(sp),
                                   edge_color="#E07B39", arrows=True,
                                   arrowsize=16, width=2.5, alpha=0.9,
                                   connectionstyle="arc3,rad=0.08")
        nx.draw_networkx_edge_labels(G, pos, elabels, ax=ax, font_size=7)
        ttl = f"Shortest-Path Tree from Node {src}" if show_sp else \
              f"Random Directed Graph  N={n}"
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    out = fname or f"cpu_graph_N{n}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"[SAVED] {out}")


# ==============================================================================
#  5. HEATMAP ANALYSIS
# ==============================================================================

def draw_heatmap(adj, dist, pred, n, fname=None):
    """
    4-panel heatmap figure:
      A  Initial adjacency matrix   B  Final distance matrix
      C  Improvement delta          D  Predecessor matrix
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

    cm_hot = matplotlib.cm.get_cmap("YlOrRd").copy()
    cm_hot.set_bad("#DDDDDD")
    cm_grn = matplotlib.cm.get_cmap("Greens").copy()
    cm_grn.set_bad("#EEEEEE")
    cm_tab = matplotlib.cm.get_cmap("tab20b").copy()
    cm_tab.set_bad("#EEEEEE")

    panels = [
        (d0,    cm_hot, "A  Initial Adjacency Matrix",      "Edge weight"),
        (df_m,  cm_hot, "B  Final Distance Matrix",         "Shortest path"),
        (delta, cm_grn, "C  Improvement  (init - final)",   "Distance saved"),
        (pred_f,cm_tab, "D  Predecessor Matrix",            "Next-hop node"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor("#F8F8F6")
    fig.suptitle(f"Floyd-Warshall Matrix Analysis  |  CPU  |  N={n}",
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
    out = fname or f"cpu_heatmap_N{n}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"[SAVED] {out}")


# ==============================================================================
#  6. ANIMATION
# ==============================================================================

def animate(adj, snaps, n, fname=None):
    """Save a GIF of the distance and predecessor matrices evolving over k."""
    def clip(m):
        c = m.astype(float).copy()
        c[c >= INF * 0.9] = np.nan
        return c

    frames = [(adj, None, "Initial  (before k=0)")] + \
             [(d, p, f"k = {k}   relay node {k}") for k, d, p in snaps]

    finite = clip(snaps[-1][1])
    finite = finite[np.isfinite(finite)]
    vmax   = float(np.percentile(finite, 98)) if len(finite) else 100.0

    cm_d = matplotlib.cm.get_cmap("YlOrRd").copy()
    cm_d.set_bad("#DDDDDD")
    cm_p = matplotlib.cm.get_cmap("tab20b").copy()
    cm_p.set_bad("#EEEEEE")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#F8F8F6")

    im_d = axes[0].imshow(clip(adj), cmap=cm_d, vmin=0, vmax=vmax,
                          interpolation="nearest", aspect="auto")
    fig.colorbar(im_d, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("Distance Matrix", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Node j"); axes[0].set_ylabel("Node i")

    p0 = _build_pred(adj).astype(float)
    p0[p0 < 0] = np.nan
    im_p = axes[1].imshow(p0, cmap=cm_p, vmin=0, vmax=n - 1,
                          interpolation="nearest", aspect="auto")
    fig.colorbar(im_p, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Predecessor Matrix", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Node j"); axes[1].set_ylabel("Node i")

    ttl = fig.suptitle("", fontsize=12, fontweight="bold")

    def update(idx):
        df, pf, lbl = frames[idx]
        im_d.set_data(clip(df))
        if pf is not None:
            pp = pf.astype(float)
            pp[pp < 0] = np.nan
            im_p.set_data(pp)
        ttl.set_text(f"Floyd-Warshall  |  CPU  |  {lbl}")
        return im_d, im_p

    a = anim_mod.FuncAnimation(fig, update, frames=len(frames),
                                interval=650, blit=False)
    out = fname or f"cpu_animation_N{n}.gif"
    a.save(out, writer="pillow", fps=1.4)
    plt.close()
    print(f"[SAVED] {out}")


# ==============================================================================
#  7. PATH TABLE
# ==============================================================================

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
#  8. BENCHMARK
# ==============================================================================

def benchmark():
    print(f"\n{'='*54}")
    print(f"  CPU Benchmark  |  runs={RUNS}  |  edge_prob={EDGE_PROB}")
    print(f"{'='*54}")
    print(f"  {'N':>5}  {'Mean Time (s)':>14}  {'Ops  N^3':>12}  {'Mops/s':>10}")
    print(f"  {'-'*50}")

    rows = []
    for n in GRAPH_SIZES:
        adj   = generate_graph(n, seed=SEED)
        ops   = n ** 3
        times = []

        # 1 warm-up run (not timed)
        floyd_warshall_cpu(adj)

        for _ in range(RUNS):
            t0 = time.perf_counter()
            floyd_warshall_cpu(adj)
            times.append(time.perf_counter() - t0)

        mean_t = float(np.mean(times))
        mops   = ops / mean_t / 1e6
        rows.append({
            "N":             n,
            "CPU Time (s)":  round(mean_t, 6),
            "Ops (N^3)":     ops,
            "CPU Mops/s":    round(mops, 2),
        })
        print(f"  {n:>5}  {mean_t:>14.6f}  {ops:>12,}  {mops:>10.2f}")

    print(f"  {'-'*50}")
    return pd.DataFrame(rows)


def plot_benchmark(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#F8F8F6")

    xs, ts, ms = df["N"].tolist(), df["CPU Time (s)"].tolist(), df["CPU Mops/s"].tolist()

    axes[0].plot(xs, ts, "-o", color=BLUE, lw=2.2, ms=8)
    for x, t in zip(xs, ts):
        axes[0].annotate(f"{t:.4f}s", xy=(x, t), xytext=(4, 6),
                         textcoords="offset points", fontsize=8, color=BLUE)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Graph Size N", fontsize=11)
    axes[0].set_ylabel("Time (s, log scale)", fontsize=11)
    axes[0].set_title("CPU Execution Time vs N", fontsize=12, fontweight="bold")
    axes[0].grid(True, which="both", ls="--", alpha=0.4)

    axes[1].bar(xs, ms, color=BLUE, alpha=0.75, edgecolor="white",
                width=max(xs) // 10)
    axes[1].set_xlabel("Graph Size N", fontsize=11)
    axes[1].set_ylabel("Million ops / second", fontsize=11)
    axes[1].set_title("CPU Throughput (Mops/s)", fontsize=12, fontweight="bold")
    axes[1].grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("cpu_benchmark.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("[SAVED] cpu_benchmark.png")


# ==============================================================================
#  9. DEMO  (full step-by-step on a small graph)
# ==============================================================================

def run_demo(n=10, edge_prob=EDGE_PROB, seed=SEED, do_animate=True):
    sep = "=" * 68
    print(f"\n{sep}\n  Floyd-Warshall Demo  |  CPU  |  N={n}\n{sep}")

    # Step 1 — generate
    print("\n[1/7]  Generating graph ...")
    adj     = generate_graph(n, edge_prob=edge_prob, seed=seed)
    nedges  = int(np.sum((adj < INF * 0.9) & (adj > 0)))
    print(f"       nodes={n}  edges={nedges}  density={nedges/(n*(n-1)):.1%}")
    print_table(adj, "Initial Adjacency Matrix  (inf = no edge)")

    # Step 2 — draw raw graph
    print("\n[2/7]  Drawing raw graph ...")
    draw_graph(adj, n=n, fname=f"cpu_graph_N{n}_raw.png")

    # Step 3 — run Floyd-Warshall with verbose per-k output
    print("\n[3/7]  Running Floyd-Warshall  (printing every k iteration) ...")
    dist, pred, elapsed, snaps = floyd_warshall_cpu(
        adj, verbose=True, snapshots=True
    )
    print(f"\n       Completed in {elapsed:.6f}s")

    # Step 4 — final matrices
    print("\n[4/7]  Final matrices ...")
    print_table(dist, "Final Distance Matrix  d[i][j]  (inf = unreachable)")
    print_table(pred, "Final Predecessor Matrix  pred[i][j]  (-- = no path)",
                pred=True)

    # Step 5 — heatmaps
    print("\n[5/7]  Heatmap analysis ...")
    draw_heatmap(adj, dist, pred, n, fname=f"cpu_heatmap_N{n}.png")

    # Step 6 — graph with SP tree overlay
    print("\n[6/7]  Shortest-path tree (from node 0) ...")
    draw_graph(adj, dist, pred, src=0, n=n, fname=f"cpu_graph_N{n}.png")

    # Step 7 — path table
    print("\n[7/7]  Reconstructed shortest paths ...")
    print_paths(dist, pred, max_rows=30)

    # Optional animation
    if do_animate:
        print("\n[+]   Saving animation ...")
        animate(adj, snaps, n, fname=f"cpu_animation_N{n}.gif")

    print(f"\n{sep}\n  Demo done  |  elapsed: {elapsed:.6f}s\n{sep}\n")


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print(f"\n{'='*54}")
    print(f"  Floyd-Warshall  |  CPU Only")
    print(f"  Python {sys.version.split()[0]}  |  NumPy {np.__version__}")
    print(f"{'='*54}")

    run_demo(n=10, edge_prob=0.5, seed=SEED, do_animate=True)

    print(f"\n{'='*68}\n  BENCHMARK\n{'='*68}")
    df = benchmark()
    print(f"\n{df.to_string(index=False)}")
    plot_benchmark(df)
    df.to_csv("cpu_results.csv", index=False)
    print("\n[SAVED] cpu_results.csv")
    print("[NEXT]  Run floyd_warshall_tpu.py in Google Colab,")
    print("        download tpu_results.csv, then run combine_results.py")


if __name__ == "__main__":
    main()
