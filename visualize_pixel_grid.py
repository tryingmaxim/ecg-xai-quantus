# starten mit: python visualize_pixel_grid.py
# visualisiert ein ECG-Bild als Pixel-Gitter mit Pixelwerten
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src import configs
from plot_style import set_confmat_style


def visualize_pixel_grid(img_path: Path, size: int = 28, cmap: str = "gray") -> None:
    set_confmat_style()

    if not img_path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {img_path}")

    img = Image.open(img_path).convert("L")

    tfms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )
    tensor = tfms(img)[0]
    pixels = tensor.numpy()

    img_28 = Image.fromarray(np.uint8(pixels * 255), mode="L")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_28, cmap=cmap, vmin=0, vmax=255)
    axes[0].set_title(f"ECG image ({size}×{size} model input)")
    axes[0].axis("off")

    axes[1].imshow(pixels, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(f"Pixel values ({size}×{size})")
    axes[1].axis("off")

    for i in range(size):
        for j in range(size):
            val = pixels[i, j]
            axes[1].text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="red",
            )

    fig.tight_layout()

    out_dir = configs.THESIS_DIR / "pixel_grids"
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"pixel_grid_{img_path.stem}_{size}x{size}"
    png_path = out_dir / f"{fname}.png"
    pdf_path = out_dir / f"{fname}.pdf"

    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight")

    plt.close(fig)

    print(f"[OK] Saved PNG: {png_path}")
    print(f"[OK] Saved PDF: {pdf_path}")


def main():
    sample_class = configs.CLASSES[0]
    cls_dir = configs.DATA_TEST / sample_class

    img_files = sorted(list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")))
    if not img_files:
        raise FileNotFoundError(f"Keine Testbilder in: {cls_dir}")

    visualize_pixel_grid(img_files[0], size=28)


if __name__ == "__main__":
    main()
