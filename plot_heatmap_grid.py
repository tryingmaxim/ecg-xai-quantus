# plot_heatmap_grid.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

from plot_style import set_confmat_style
from src import configs


# Default-Methoden & Index (kann via CLI überschrieben werden)
DEFAULT_METHODS = ["gradcam", "gradcam++", "ig"]
DEFAULT_INDEX = "000.png"


def _get_explanations_root() -> Path:
    """
    Liefert das Root-Verzeichnis für die Erklärungs-Heatmaps.

    Nutzt, falls vorhanden, configs.EXPLANATIONS_DIR, sonst fallback auf
    'outputs/explanations'.
    """
    root = getattr(configs, "EXPLANATIONS_DIR", None)
    if root is None:
        root = Path("outputs") / "explanations"
    return Path(root)


def _get_plots_out_dir() -> Path:
    """Ausgabeverzeichnis für allgemeine Plots."""
    # analog zu deinem Histogramm-Skript
    base = getattr(configs, "METRICS_DIR", Path("outputs/metrics"))
    out_dir = Path(base) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _get_thesis_xai_dir() -> Path:
    """Ausgabeverzeichnis für Thesis-XAI-Figuren."""
    thesis_base = getattr(configs, "THESIS_DIR", Path("outputs/thesis_figures"))
    xai_dir = Path(thesis_base) / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    return xai_dir


def _collect_models(expl_root: Path) -> List[str]:
    """Findet alle Modellordner unterhalb von expl_root."""
    if not expl_root.exists():
        print(f"[ERROR] Erklärungs-Root existiert nicht: {expl_root}")
        return []
    models = sorted([p.name for p in expl_root.iterdir() if p.is_dir()])
    if not models:
        print(f"[ERROR] Keine Modellordner in {expl_root}")
    return models


def _load_image(img_path: Path) -> Image.Image | None:
    """Lädt ein Bild robust; gibt None zurück, wenn etwas schiefgeht."""
    try:
        return Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        return None
    except UnidentifiedImageError:
        print(f"[WARN] Kann Bild nicht lesen (UnidentifiedImageError): {img_path}")
        return None
    except Exception as e:
        print(f"[WARN] Fehler beim Laden von {img_path}: {e}")
        return None


def _create_axes_grid(
    n_rows: int,
    n_cols: int,
    figsize: Tuple[float, float],
) -> Tuple[plt.Figure, List[List[plt.Axes]]]:
    """
    Erzeugt eine konsistent 2D-Liste von Axes, egal ob 1×N, N×1 oder NxN.
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # axes in 2D-Form bringen
    if n_rows == 1 and n_cols == 1:
        axes_2d = [[axes]]
    elif n_rows == 1:
        axes_2d = [list(axes)]
    elif n_cols == 1:
        axes_2d = [[ax] for ax in axes]
    else:
        axes_2d = [list(row) for row in axes]

    return fig, axes_2d


def plot_heatmap_grid(
    methods: Iterable[str] = DEFAULT_METHODS,
    index_filename: str = DEFAULT_INDEX,
) -> None:
    """
    Erzeugt ein Grid: Zeilen = Modelle, Spalten = Methoden.

    Parameters
    ----------
    methods : Iterable[str]
        XAI-Methoden, die als Spalten verwendet werden (Ordnernamen).
    index_filename : str
        Dateiname des Bildes innerhalb von model/method/, z.B. '000.png'.
    """
    set_confmat_style()

    expl_root = _get_explanations_root()
    models = _collect_models(expl_root)
    if not models:
        return

    methods = list(methods)
    if not methods:
        print("[ERROR] Keine Methoden angegeben.")
        return

    n_rows = len(models)
    n_cols = len(methods)

    # Figurgröße etwas dynamisch skalieren
    fig_width = 3.7 * n_cols
    fig_height = 2.6 * n_rows
    fig, axes = _create_axes_grid(n_rows, n_cols, figsize=(fig_width, fig_height))

    for i, model in enumerate(models):
        for j, method in enumerate(methods):
            ax = axes[i][j]
            img_path = expl_root / model / method / index_filename

            img = _load_image(img_path)
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

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Zeilen-Label = Modellname (nur erste Spalte)
            if j == 0:
                ax.set_ylabel(
                    model,
                    rotation=0,
                    labelpad=40,
                    fontsize=12,
                    va="center",
                )

            # Spalten-Label = Methodenname (nur erste Zeile)
            if i == 0:
                ax.set_title(method, fontsize=13, pad=10)

    fig.tight_layout()

    # Speichern
    plots_dir = _get_plots_out_dir()
    thesis_dir = _get_thesis_xai_dir()

    # Dateinamen lassen erkennen, welcher Index genutzt wurde
    png_name = f"heatmap_grid_{index_filename.replace('.png', '')}.png"
    pdf_name = f"heatmap_grid_{index_filename.replace('.png', '')}.pdf"

    png_path = plots_dir / png_name
    pdf_path = thesis_dir / pdf_name

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved heatmap grid PNG: {png_path}")
    print(f"[OK] Saved heatmap grid PDF: {pdf_path}")


def main() -> None:
    """
    Optional: CLI-Interface, damit du Methoden/Index aus der Konsole ändern kannst.

    Beispiele:
    ----------
    python -m plot_heatmap_grid
    python -m plot_heatmap_grid --index 005.png
    python -m plot_heatmap_grid --methods gradcam ig
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Erzeuge ein Modell×XAI-Methoden Heatmap-Grid."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="XAI-Methoden-Ordner (Default: %(default)s)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=DEFAULT_INDEX,
        help="Dateiname des Beispielbilds (Default: %(default)s)",
    )

    args = parser.parse_args()
    plot_heatmap_grid(methods=args.methods, index_filename=args.index)


if __name__ == "__main__":
    main()
