# visualize_first_conv_filters.py
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch

from src import configs
from src.model_def import build_model
from plot_style import set_confmat_style


def _load_model_from_ckpt(model_name: str) -> torch.nn.Module | None:
    """
    Lädt ein Modell aus dem Checkpoint.

    Unterstützt sowohl:
        - {"state_dict": ...}
        - direktes state_dict
    und entfernt ggf. 'module.' Prefix (DataParallel).
    """
    ckpt_path = configs.CKPT_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        print(f"[WARN] Checkpoint fehlt: {ckpt_path}")
        return None

    blob = torch.load(ckpt_path, map_location="cpu")
    if isinstance(blob, dict):
        state = blob.get("state_dict", blob)
    else:
        state = blob

    model = build_model(model_name, num_classes=len(configs.CLASSES))

    clean_state = {
        (k.replace("module.", "") if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    # strict=False, falls der Checkpoint z.B. noch zusätzliche Keys enthält
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model


def _get_first_conv_layer(model: torch.nn.Module) -> torch.nn.Conv2d | None:
    """Findet die erste Conv2d-Schicht im Modell."""
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            return m
    return None


def visualize_first_conv(model_name: str, n_filters: int = 32) -> None:
    """
    Visualisiert die ersten Convolution-Filter eines Modells als Grid.

    - Filter werden über Farbkanäle gemittelt (C_in → 1)
    - auf [0, 1] normalisiert
    - als Graustufenbilder geplottet
    """
    set_confmat_style()

    model = _load_model_from_ckpt(model_name)
    if model is None:
        return

    first_conv = _get_first_conv_layer(model)
    if first_conv is None:
        print(f"[WARN] Keine Conv2d-Schicht in {model_name} gefunden.")
        return

    with torch.no_grad():
        w = first_conv.weight.detach().cpu().clone()  # (C_out, C_in, k, k)

    # über Eingangs-Farbkanäle mitteln → (C_out, 1, k, k)
    w = w.mean(dim=1, keepdim=True)

    # gegen Division durch 0 absichern
    w_min = float(w.min())
    w_max = float(w.max())
    if w_max > w_min:
        w = (w - w_min) / (w_max - w_min)
    else:
        # alle Gewichte gleich → konstante 0.5
        w = torch.zeros_like(w) + 0.5

    n_filters = min(n_filters, w.shape[0])
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5))
    axes = axes.flatten()

    for i in range(n_filters):
        ax = axes[i]
        ax.imshow(w[i, 0].numpy(), cmap="gray")
        ax.set_title(f"F{i}", fontsize=8)
        ax.axis("off")

    # restliche Achsen ausblenden
    for j in range(n_filters, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Erste Conv-Filter – {model_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = configs.THESIS_DIR / "filters"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"first_conv_{model_name}.png"
    pdf_path = out_dir / f"first_conv_{model_name}.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)

    plt.close(fig)
    print(f"[OK] Saved PNG: {png_path}")
    print(f"[OK] Saved PDF: {pdf_path}")


def main() -> None:
    """
    Standard: alle Modelle in configs.MODEL_NAMES.
    Optional: mit --model nur ein Modell ausgeben.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualisiert die ersten Conv-Filter der Modelle."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nur dieses Modell plotten (Default: alle in configs.MODEL_NAMES)",
    )
    parser.add_argument(
        "--n_filters",
        type=int,
        default=32,
        help="Anzahl der zu plottenden Filter (Default: 32)",
    )
    args = parser.parse_args()

    if args.model is not None:
        visualize_first_conv(args.model, n_filters=args.n_filters)
    else:
        for model_name in configs.MODEL_NAMES:
            visualize_first_conv(model_name, n_filters=args.n_filters)


if __name__ == "__main__":
    main()
