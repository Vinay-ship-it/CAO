"""
Floyd-Warshall  |  COMBINE CPU + TPU RESULTS
=============================================
Requirements
    py -m pip install numpy pandas matplotlib tabulate

Run AFTER collecting both CSVs:
    Step 1:  py floyd_warshall_cpu.py              -> cpu_results.csv
    Step 2:  Run floyd_warshall_tpu.py in Colab    -> download tpu_results.csv
    Step 3:  py combine_results.py                 -> all comparison charts

Outputs
    combined_results.csv      merged table with speedup column
    compare_time.png          CPU vs TPU execution time (log scale)
    compare_speedup.png       TPU speedup bar chart per graph size
    compare_throughput.png    CPU vs TPU Mops/s grouped bar chart
    compare_all.png           4-panel summary figure with result table
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

CPU_CSV = "cpu_results.csv"
TPU_CSV = "tpu_results.csv"
BLUE    = "#2E86AB"   # CPU
AMBER   = "#E07B39"   # TPU
GREEN   = "#22C55E"   # speedup > 1
RED     = "#EF4444"   # speedup < 1


# ==============================================================================
#  1. LOAD & MERGE
# ==============================================================================

def load_and_merge():
    """
    Load cpu_results.csv and tpu_results.csv, merge on N, compute:
        Speedup      = CPU Time / TPU Time
        CPU Mops/s   = N^3 / CPU Time / 1e6
        TPU Mops/s   = N^3 / TPU Time / 1e6

    Returns a clean DataFrame ready for plotting.
    """
    missing = [f for f in [CPU_CSV, TPU_CSV] if not os.path.exists(f)]
    if missing:
        print(f"\n[ERROR] Missing file(s): {missing}")
        for f in missing:
            script = f.replace("_results.csv", "_warshall_cpu.py" if "cpu" in f
                                else "_warshall_tpu.py")
            print(f"        Generate with: py floyd_warshall_cpu.py"
                  if "cpu" in f else
                  f"        Generate with: run floyd_warshall_tpu.py in Colab")
        sys.exit(1)

    cpu = pd.read_csv(CPU_CSV)
    tpu = pd.read_csv(TPU_CSV)

    # Find N column and time column regardless of exact naming
    def _col(df, keywords):
        for k in keywords:
            for c in df.columns:
                if k.lower() in c.lower():
                    return c
        return None

    n_cpu  = _col(cpu, ["N"])
    n_tpu  = _col(tpu, ["N"])
    t_cpu  = _col(cpu, ["CPU Time", "cpu_time", "Time"])
    t_tpu  = _col(tpu, ["TPU Time", "tpu_time", "Time"])

    if not all([n_cpu, n_tpu, t_cpu, t_tpu]):
        print("[ERROR] Could not detect required columns.")
        print(f"  CPU columns: {list(cpu.columns)}")
        print(f"  TPU columns: {list(tpu.columns)}")
        sys.exit(1)

    df = pd.merge(
        cpu[[n_cpu, t_cpu]].rename(columns={n_cpu: "N", t_cpu: "CPU Time (s)"}),
        tpu[[n_tpu, t_tpu]].rename(columns={n_tpu: "N", t_tpu: "TPU Time (s)"}),
        on="N"
    ).sort_values("N").reset_index(drop=True)

    df["Speedup"]     = df["CPU Time (s)"] / df["TPU Time (s)"]
    df["CPU Mops/s"]  = df["N"] ** 3 / df["CPU Time (s)"] / 1e6
    df["TPU Mops/s"]  = df["N"] ** 3 / df["TPU Time (s)"] / 1e6

    return df


# ==============================================================================
#  2. PRINT COMPARISON TABLE
# ==============================================================================

def print_comparison(df):
    sep = "=" * 72
    print(f"\n{sep}")
    print("  FLOYD-WARSHALL  |  CPU vs TPU  |  RESULTS COMPARISON")
    print(f"{sep}\n")

    rows = []
    for _, r in df.iterrows():
        winner = "TPU" if r["Speedup"] > 1.0 else "CPU"
        rows.append([
            int(r["N"]),
            f"{r['CPU Time (s)']:.6f}",
            f"{r['TPU Time (s)']:.6f}",
            f"{r['Speedup']:.2f}x",
            f"{r['CPU Mops/s']:.1f}",
            f"{r['TPU Mops/s']:.1f}",
            winner,
        ])

    print(tabulate(rows,
                   headers=["N", "CPU (s)", "TPU (s)", "Speedup",
                             "CPU Mops/s", "TPU Mops/s", "Winner"],
                   tablefmt="simple"))

    best_n   = int(df.loc[df["Speedup"].idxmax(), "N"])
    best_spd = df["Speedup"].max()
    avg_spd  = df["Speedup"].mean()

    print(f"\n  Peak speedup : {best_spd:.2f}x  at N = {best_n}")
    print(f"  Avg speedup  : {avg_spd:.2f}x  across all graph sizes")
    if avg_spd > 1:
        print("  Verdict      : TPU is faster on average")
    else:
        print("  Verdict      : CPU is faster on average  "
              "(TPU dispatch overhead dominates at small N)")
    print(f"\n{sep}\n")


# ==============================================================================
#  3. CHART — EXECUTION TIME
# ==============================================================================

def plot_time(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#FAFAFA")

    xs    = df["N"].tolist()
    cpu_t = df["CPU Time (s)"].tolist()
    tpu_t = df["TPU Time (s)"].tolist()

    ax.plot(xs, cpu_t, "-o", color=BLUE,  lw=2.2, ms=8, label="CPU  (NumPy)")
    ax.plot(xs, tpu_t, "-^", color=AMBER, lw=2.2, ms=8, label="TPU  (JAX / XLA)")

    for x, t in zip(xs, cpu_t):
        ax.annotate(f"{t:.4f}s", xy=(x, t), xytext=(5, 6),
                    textcoords="offset points", fontsize=8, color=BLUE)
    for x, t in zip(xs, tpu_t):
        ax.annotate(f"{t:.4f}s", xy=(x, t), xytext=(5, -14),
                    textcoords="offset points", fontsize=8, color=AMBER)

    ax.set_yscale("log")
    ax.set_xlabel("Graph Size  N  (nodes)", fontsize=12)
    ax.set_ylabel("Execution Time (s, log scale)", fontsize=12)
    ax.set_title("Floyd-Warshall  |  CPU vs TPU  |  Execution Time",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("compare_time.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[SAVED] compare_time.png")


# ==============================================================================
#  4. CHART — SPEEDUP
# ==============================================================================

def plot_speedup(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#F8F8F6")

    xs  = df["N"].tolist()
    spd = df["Speedup"].tolist()
    bar_colors = [GREEN if s >= 1 else RED for s in spd]
    w = max(xs) // 10

    bars = ax.bar(xs, spd, color=bar_colors, alpha=0.78,
                  edgecolor="white", linewidth=0.6, width=w)
    ax.axhline(1.0, color="#888", ls="--", lw=1.4,
               label="Break-even  (1x = same speed)")

    for bar, s in zip(bars, spd):
        ax.text(bar.get_x() + bar.get_width() / 2,
                s + max(spd) * 0.01,
                f"{s:.2f}x",
                ha="center", va="bottom", fontsize=9, fontweight="600",
                color=GREEN if s >= 1 else RED)

    ax.set_xlabel("Graph Size  N  (nodes)", fontsize=12)
    ax.set_ylabel("TPU Speedup  (x  over CPU)", fontsize=12)
    ax.set_title("Floyd-Warshall  |  TPU Speedup over CPU",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("compare_speedup.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[SAVED] compare_speedup.png")


# ==============================================================================
#  5. CHART — THROUGHPUT
# ==============================================================================

def plot_throughput(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#F8F8F6")

    xs  = df["N"].tolist()
    cm  = df["CPU Mops/s"].tolist()
    tm  = df["TPU Mops/s"].tolist()
    w   = max(xs) // 12

    ax.bar([x - w/2 for x in xs], cm, width=w, color=BLUE,
           alpha=0.78, edgecolor="white", label="CPU  Mops/s")
    ax.bar([x + w/2 for x in xs], tm, width=w, color=AMBER,
           alpha=0.78, edgecolor="white", label="TPU  Mops/s")

    ax.set_xlabel("Graph Size  N  (nodes)", fontsize=12)
    ax.set_ylabel("Million operations / second", fontsize=12)
    ax.set_title("Floyd-Warshall  |  Throughput Comparison",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("compare_throughput.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[SAVED] compare_throughput.png")


# ==============================================================================
#  6. 4-PANEL SUMMARY FIGURE
# ==============================================================================

def plot_all(df):
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#F8F8F6")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.34)

    xs    = df["N"].tolist()
    cpu_t = df["CPU Time (s)"].tolist()
    tpu_t = df["TPU Time (s)"].tolist()
    spd   = df["Speedup"].tolist()
    cm    = df["CPU Mops/s"].tolist()
    tm    = df["TPU Mops/s"].tolist()

    # ── A  Execution time ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FAFAFA")
    ax1.plot(xs, cpu_t, "-o", color=BLUE,  lw=2.2, ms=7, label="CPU")
    ax1.plot(xs, tpu_t, "-^", color=AMBER, lw=2.2, ms=7, label="TPU")
    ax1.set_yscale("log")
    ax1.set_xlabel("N (nodes)", fontsize=10)
    ax1.set_ylabel("Time (s, log)", fontsize=10)
    ax1.set_title("A  Execution Time", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    # ── B  Speedup ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    w   = max(xs) // 10
    bar_c = [GREEN if s >= 1 else RED for s in spd]
    ax2.bar(xs, spd, color=bar_c, alpha=0.78, edgecolor="white", width=w)
    ax2.axhline(1.0, color="#888", ls="--", lw=1.2)
    for x, s in zip(xs, spd):
        ax2.text(x, s + max(spd)*0.01, f"{s:.2f}x",
                 ha="center", va="bottom", fontsize=9, fontweight="600",
                 color=GREEN if s >= 1 else RED)
    ax2.set_xlabel("N (nodes)", fontsize=10)
    ax2.set_ylabel("Speedup (x)", fontsize=10)
    ax2.set_title("B  TPU Speedup over CPU", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", ls="--", alpha=0.4)

    # ── C  Throughput ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    w2  = max(xs) // 12
    ax3.bar([x - w2/2 for x in xs], cm, width=w2, color=BLUE,
            alpha=0.78, edgecolor="white", label="CPU")
    ax3.bar([x + w2/2 for x in xs], tm, width=w2, color=AMBER,
            alpha=0.78, edgecolor="white", label="TPU")
    ax3.set_xlabel("N (nodes)", fontsize=10)
    ax3.set_ylabel("Mops/s", fontsize=10)
    ax3.set_title("C  Throughput (Mops/s)", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, axis="y", ls="--", alpha=0.4)

    # ── D  Result table ───────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    col_labels = ["N", "CPU (s)", "TPU (s)", "Speedup", "Winner"]
    cell_data  = [
        [int(r["N"]),
         f"{r['CPU Time (s)']:.4f}",
         f"{r['TPU Time (s)']:.4f}",
         f"{r['Speedup']:.2f}x",
         "TPU" if r["Speedup"] >= 1 else "CPU"]
        for _, r in df.iterrows()
    ]

    tbl = ax4.table(
        cellText  = cell_data,
        colLabels = col_labels,
        loc       = "center",
        cellLoc   = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.7)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1E293B")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Style winner column
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        if r["Speedup"] >= 1:
            tbl[i, 4].set_facecolor("#DCFCE7")
            tbl[i, 4].set_text_props(color="#166534", fontweight="bold")
        else:
            tbl[i, 4].set_facecolor("#DBEAFE")
            tbl[i, 4].set_text_props(color="#1E40AF", fontweight="bold")

    ax4.set_title("D  Summary Table", fontsize=11, fontweight="bold", pad=12)

    fig.suptitle("Floyd-Warshall  |  CPU vs TPU  |  Full Comparison",
                 fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig("compare_all.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[SAVED] compare_all.png")


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print(f"\n{'='*56}")
    print("  Floyd-Warshall  |  Combine CPU + TPU Results")
    print(f"  Python {sys.version.split()[0]}")
    print(f"{'='*56}")

    print(f"\n[1/6]  Loading {CPU_CSV}  +  {TPU_CSV} ...")
    df = load_and_merge()

    print("[2/6]  Comparison table ...")
    print_comparison(df)

    # Save merged CSV
    out = df.copy()
    out["Speedup"]    = out["Speedup"].round(3)
    out["CPU Mops/s"] = out["CPU Mops/s"].round(2)
    out["TPU Mops/s"] = out["TPU Mops/s"].round(2)
    out.to_csv("combined_results.csv", index=False)
    print("[SAVED] combined_results.csv")

    print("\n[3/6]  Execution time chart ...")
    plot_time(df)

    print("\n[4/6]  Speedup chart ...")
    plot_speedup(df)

    print("\n[5/6]  Throughput chart ...")
    plot_throughput(df)

    print("\n[6/6]  4-panel summary figure ...")
    plot_all(df)

    print(f"\n{'='*56}")
    print("  Done!  Output files:")
    for f in ["combined_results.csv", "compare_time.png",
              "compare_speedup.png", "compare_throughput.png", "compare_all.png"]:
        print(f"    {f}")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    main()
