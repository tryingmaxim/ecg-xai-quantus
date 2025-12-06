# plot_style.py
import matplotlib as mpl

# Einheitliche Farben für alle Plots
PALETTE = {
    "gradcam":   "#1f77b4",  # blau
    "gradcam++": "#ff7f0e",  # orange
    "ig":        "#2ca02c",  # grün
}

def get_color(method: str) -> str:
    return PALETTE.get(method, "gray")


def set_confmat_style():
    """
    Globaler Matplotlib-Style – angelehnt an deine Confusion Matrix:
    weißer Hintergrund, klare Schriften, kein unnötiges Grid.
    """
    mpl.rcParams.update({
        "figure.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def apply_axes_style(ax):
    """
    Schlichte Achsen wie bei der Confusion-Matrix:
    obere und rechte Achse ausblenden.
    """
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
