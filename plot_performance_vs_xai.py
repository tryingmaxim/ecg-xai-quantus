# plot_performance_vs_xai.py
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

    df = df_q.merge(df_p, on="model", how="left")

    # Farben wie oben
    palette = {
        "gradcam":   "#1f77b4",
        "gradcam++": "#ff7f0e",
        "ig":        "#2ca02c",
    }

    out_dir = Path("outputs/metrics/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Bubble-Plot: x = Accuracy, y = Faithfulness, Größe = (1 - MaxSensitivity)
    # (je kleiner MaxSensitivity, desto robuster, also größere Blase)
    max_sens_norm = (df["max_sens"] - df["max_sens"].min()) / (df["max_sens"].max() - df["max_sens"].min() + 1e-9)
    sizes = (1.0 - max_sens_norm) * 600 + 100  # Bubble-Größen

    for method in df["method"].unique():
        sub = df[df["method"] == method]
        plt.scatter(
            sub["accuracy"],
            sub["faithfulness_corr"],
            s=sizes[sub.index],
            color=palette.get(method, "gray"),
            alpha=0.7,
            edgecolor="k",
            linewidth=0.5,
            label=method,
        )

    # Modelle als Text-Labels leicht versetzt
    for _, row in df.iterrows():
        plt.text(
            row["accuracy"] + 0.002,
            row["faithfulness_corr"] + 0.002,
            row["model"],
            fontsize=8,
            alpha=0.8,
        )

    plt.xlabel("Accuracy")
    plt.ylabel("FaithfulnessCorrelation")
    plt.title("Trade-off: Modellgüte vs. Erklärqualität\n(Blasengröße ≈ Robustheit der Erklärungen)")
    plt.grid(alpha=0.3)
    plt.legend(title="XAI-Methode")
    plt.tight_layout()

    out_path = out_dir / "performance_vs_xai.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("[OK] Saved:", out_path)


if __name__ == "__main__":
    main()
