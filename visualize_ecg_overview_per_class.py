# python visualize_ecg_one_per_class.py

from __future__ import annotations
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from src import configs
from plot_style import set_confmat_style


def _get_one_image(cls: str) -> Path | None:
    img_dir = configs.DATA_TRAIN / cls
    files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    return files[0] if files else None


def make_overview():
    set_confmat_style()

    classes = list(configs.CLASSES)
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4))

    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        f = _get_one_image(cls)
        if f is None:
            ax.text(0.5, 0.5, f"No image\nfor {cls}", ha="center", va="center")
            ax.axis("off")
            continue

        img = Image.open(f).convert("RGB")
        ax.imshow(img)
        ax.set_title(cls, fontsize=12)
        ax.axis("off")

    plt.tight_layout()

    out_dir = configs.THESIS_DIR / "ecg_overview_one_per_class"
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "ecg_one_example_per_class.png"
    pdf_path = out_dir / "ecg_one_example_per_class.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    plt.close()

    print("[OK] Saved:", png_path)
    print("[OK] Saved:", pdf_path)


def main():
    make_overview()


if __name__ == "__main__":
    main()
