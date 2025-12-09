# starten mit: python visualize_super_grid_all.py
# visualisiert Super-Grids mit Originalbild und XAI-Heatmaps für alle Modelle
from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src import configs
from plot_style import set_confmat_style


def _get_data_root() -> Path:
    root = getattr(configs, "DATA_TEST", None)
    if root is None:
        root = Path("data") / "ecg_test"
    return Path(root)


def _get_original_image_by_index(index: str) -> Image.Image:
    idx = int(index.split(".")[0])
    data_root = _get_data_root()
    img_paths = sorted(list(data_root.rglob("*.png")) + list(data_root.rglob("*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"Keine Testbilder gefunden unter: {data_root}")
    if idx >= len(img_paths):
        raise IndexError(f"Index {idx} außerhalb der Bildliste (len={len(img_paths)})")
    return Image.open(img_paths[idx]).convert("RGB")


def load_explanation(model_name: str, method: str, index: str):
    p = configs.EXPL_DIR / model_name / method / index
    if not p.exists():
        print(f"[WARN] Heatmap fehlt: {p}")
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception as e:
        print(f"[WARN] Fehler beim Laden der Heatmap {p}: {e}")
        return None


def make_super_grid_for_model(model_name: str, index: str = "000.png"):
    set_confmat_style()

    print(f"\n[INFO] Erzeuge Super-Grid für: {model_name} (Index {index})")

    try:
        orig = _get_original_image_by_index(index)
    except Exception as e:
        print(f"[ERROR] Konnte Originalbild für Index {index} nicht laden: {e}")
        return

    gradcam = load_explanation(model_name, "gradcam", index)
    gradcampp = load_explanation(model_name, "gradcam++", index)
    ig = load_explanation(model_name, "ig", index)
    lime = load_explanation(model_name, "lime", index)

    methods = [
        ("Original ECG", orig),
        ("Grad-CAM", gradcam),
        ("Grad-CAM++", gradcampp),
        ("Integrated Gradients", ig),
        ("LIME", lime),
    ]

    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.5))

    if len(methods) == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, methods):
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                fontsize=10,
            )
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        f"{model_name} – original ECG and XAI heatmaps (index {index})",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_dir = configs.THESIS_DIR / "super_grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"super_grid_{model_name}_{index.replace('.png', '')}"
    png_path = out_dir / f"{base_name}.png"
    pdf_path = out_dir / f"{base_name}.pdf"

    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Gespeichert: {png_path}")
    print(f"[OK] Gespeichert: {pdf_path}")


def main():
    index = "000.png"

    expl_root = configs.EXPL_DIR
    if not expl_root.exists():
        raise FileNotFoundError(f"Erklärungs-Root existiert nicht: {expl_root}")

    models = sorted([p.name for p in expl_root.iterdir() if p.is_dir()])
    if not models:
        raise FileNotFoundError(f"Keine Modellordner in {expl_root}")

    for model_name in models:
        make_super_grid_for_model(model_name, index=index)


if __name__ == "__main__":
    main()
