# plot_quantus_scatter.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    quantus_path = Path("outputs/metrics/quantus_summary.csv")
    perf_path    = Path("outputs/metrics/model_performance.csv")

    if not quantus_path.exists():
        raise FileNotFoundError(f"{quantus_path} nicht gefunden. Erst generate_quantus_summary.py ausführen.")
    if not perf_path.exists():
        raise FileNotFoundError(f"{perf_path} nicht gefunden. Erst generate_results_table.py ausführen.")

    df_q = pd.read_csv(quantus_path)
    df_p = pd.read_csv(perf_path)

    # Merge, damit wir Accuracy mitplotten können
    df = df_q.merge(df_p, on="model", how="left")

    # Konsistente Farben pro XAI-Methode
    palette = {
        "gradcam":   "#1f77b4",  # blau
        "gradcam++": "#ff7f0e",  # orange
        "ig":        "#2ca02c",  # grün
    }

    out_dir = Path("outputs/metrics/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.30, wspace=0.25)

    # ---- (a) Faithfulness vs Robustness ----
    ax = axes[0, 0]
    for method, sub in df.groupby("method"):
        ax.scatter(
            sub["faithfulness_corr"],
            sub["max_sens"],
            label=method,
            color=palette.get(method, "gray"),
            alpha=0.8,
        )
    ax.set_xlabel("FaithfulnessCorrelation")
    ax.set_ylabel("MaxSensitivity")
    ax.set_title("(a) Faithfulness vs. Robustness")

    # ---- (b) Faithfulness vs Randomisation (MPRT) ----
    ax = axes[0, 1]
    for method, sub in df.groupby("method"):
        ax.scatter(
            sub["faithfulness_corr"],
            sub["mprt"],
            label=method,
            color=palette.get(method, "gray"),
            alpha=0.8,
        )
    ax.set_xlabel("FaithfulnessCorrelation")
    ax.set_ylabel("MPRT")
    ax.set_title("(b) Faithfulness vs. Randomisation")

    # ---- (c) Robustness vs Randomisation ----
    ax = axes[1, 0]
    for method, sub in df.groupby("method"):
        ax.scatter(
            sub["max_sens"],
            sub["mprt"],
            label=method,
            color=palette.get(method, "gray"),
            alpha=0.8,
        )
    ax.set_xlabel("MaxSensitivity")
    ax.set_ylabel("MPRT")
    ax.set_title("(c) Robustness vs. Randomisation")

    # ---- (d) Accuracy vs Faithfulness (Trade-off) ----
    ax = axes[1, 1]
    for method, sub in df.groupby("method"):
        ax.scatter(
            sub["accuracy"],
            sub["faithfulness_corr"],
            label=method,
            color=palette.get(method, "gray"),
            alpha=0.8,
        )
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("FaithfulnessCorrelation")
    ax.set_title("(d) Accuracy vs. Faithfulness")

    # Gemeinsame Legende unten
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        title="XAI-Methode"
    )

    out_path = out_dir / "quantus_scatter_grid.png"
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("[OK] Saved:", out_path)


if __name__ == "__main__":
    main()
