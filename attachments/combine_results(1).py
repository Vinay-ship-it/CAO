import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLOURS = {"CPU": "#2E86AB", "TPU": "#3BB273"}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & MERGE
# ─────────────────────────────────────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    cpu_path = Path("cpu_results.csv")
    tpu_path = Path("tpu_results.csv")

    missing = [p for p in [cpu_path, tpu_path] if not p.exists()]
    if missing:
        print("[ERROR] Missing result files:")
        for p in missing:
            print(f"         {p}")
        print("\n  1. Run benchmark_cpu.py locally   → produces cpu_results.csv")
        print("  2. Run benchmark_tpu.py in Colab  → produces tpu_results.csv")
        print("  3. Download tpu_results.csv from Colab into this folder.")
        sys.exit(1)

    cpu_df = pd.read_csv(cpu_path)
    tpu_df = pd.read_csv(tpu_path)

    df = pd.merge(cpu_df, tpu_df, on="Graph Size", how="outer").sort_values("Graph Size")
    df = df.reset_index(drop=True)

    df["TPU Speedup"] = df["CPU Time (s)"] / df["TPU Time (s)"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("  Combined Results: CPU vs TPU")
    print("=" * 55)
    print(f"{'N':>6}  {'CPU (s)':>12}  {'TPU (s)':>12}  {'TPU Speedup':>12}")
    print("-" * 55)
    for _, row in df.iterrows():
        def fmt(v):  return f"{v:.4f}" if not np.isnan(v) else "  N/A  "
        def fmtx(v): return f"{v:.2f}x" if not np.isnan(v) else "  N/A  "
        print(f"{int(row['Graph Size']):>6}  "
              f"{fmt(row['CPU Time (s)']):>12}  "
              f"{fmt(row['TPU Time (s)']):>12}  "
              f"{fmtx(row['TPU Speedup']):>12}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PLOT 1 — Line chart: execution time vs graph size
# ─────────────────────────────────────────────────────────────────────────────

def plot_line_chart(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#F7F9FB")

    for arch, col, m in [("CPU", "CPU Time (s)", "o"), ("TPU", "TPU Time (s)", "^")]:
        s = df[col].replace(0, np.nan)
        if s.notna().any():
            ax.plot(df["Graph Size"], s, f"-{m}", color=COLOURS[arch],
                    linewidth=2.2, markersize=8, label=arch, zorder=3)
            last = s.dropna()
            ax.annotate(f"{last.iloc[-1]:.3f}s",
                        xy=(df["Graph Size"].iloc[last.index[-1]], last.iloc[-1]),
                        xytext=(7, 5), textcoords="offset points",
                        fontsize=8, color=COLOURS[arch])

    ax.set_yscale("log")
    ax.set_xlabel("Graph Size N (nodes)", fontsize=12)
    ax.set_ylabel("Execution Time (seconds, log scale)", fontsize=12)
    ax.set_title("Floyd–Warshall: CPU vs TPU\nExecution Time vs Graph Size",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("combined_line.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PLOT 2 — Bar chart at largest shared N
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_chart(df: pd.DataFrame) -> None:
    row = df.dropna(subset=["CPU Time (s)", "TPU Time (s)"]).iloc[-1]
    n   = int(row["Graph Size"])

    archs = ["CPU", "TPU"]
    times = [row["CPU Time (s)"], row["TPU Time (s)"]]
    cols  = [COLOURS[a] for a in archs]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("#F7F9FB")
    bars = ax.bar(archs, times, color=cols, width=0.4, edgecolor="white", linewidth=1.3)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.03,
                f"{t:.4f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Architecture", fontsize=12)
    ax.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax.set_title(f"CPU vs TPU at N = {n} nodes",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("combined_bar.png", dpi=150)
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOT 3 — Speedup chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_speedup_chart(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#F7F9FB")

    s = df["TPU Speedup"].replace(0, np.nan)
    ax.plot(df["Graph Size"], s, "-^", color=COLOURS["TPU"],
            linewidth=2.2, markersize=8, label="TPU Speedup vs CPU", zorder=3)
    ax.fill_between(df["Graph Size"], 1, s.fillna(1),
                    where=s.fillna(0) >= 1,
                    alpha=0.15, color=COLOURS["TPU"], label="TPU faster zone")

    ax.axhline(1.0, color="#888", linestyle="--", linewidth=1.3,
               label="1× — same as CPU", zorder=2)
    ax.set_xlabel("Graph Size N (nodes)", fontsize=12)
    ax.set_ylabel("Speedup Factor (CPU time ÷ TPU time)", fontsize=12)
    ax.set_title("TPU Speedup vs CPU\nValues > 1× = faster than CPU",
                 fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("combined_speedup.png", dpi=150)
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  Floyd–Warshall  |  Combined CPU vs TPU")
    print("=" * 55)

    df = load_results()
    print_table(df)

    print("\n[VIZ] Generating charts …")
    plot_line_chart(df)
    plot_bar_chart(df)
    plot_speedup_chart(df)

    df.to_csv("combined_results.csv", index=False)


if __name__ == "__main__":
    main()
