# plot_sensitivity_analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    df = pd.read_csv("outputs/metrics/quantus_summary.csv")

    out_dir = Path("outputs/metrics/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="model", y="max_sens", hue="method", marker="o")
    plt.title("Sensitivity Trend (MaxSensitivity)")
    plt.tight_layout()

    out_path = out_dir / "sensitivity_trend.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("[OK] Saved:", out_path)

if __name__ == "__main__":
    main()
