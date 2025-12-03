# plot_heatmap_grid.py
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

MODELS = ["resnet18", "resnet50"]
METHODS = ["gradcam", "gradcam++", "ig"]
INDEX = "000.png"   # welches Bild anzeigen

def main():
    expl_root = Path("outputs/explanations")

    n_rows = len(MODELS)
    n_cols = len(METHODS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows)
    )

    if n_rows == 1:
        axes = [axes]

    for i, model in enumerate(MODELS):
        row_axes = axes[i]
        for j, method in enumerate(METHODS):
            ax = row_axes[j]
            img_path = expl_root / model / method / INDEX

            if not img_path.exists():
                ax.axis("off")
                ax.set_title(f"{model}\n{method}\n(N/A)")
                continue

            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{model} – {method}")

    plt.tight_layout()
    out_path = Path("outputs/metrics/plots/heatmap_grid.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("[OK] Saved heatmap grid:", out_path)

if __name__ == "__main__":
    main()
