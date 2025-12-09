# starten mit: python run_all_visualizations.py
# führt alle Visualisierungs-Skripte nacheinander aus
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

VIS_SCRIPTS = [
    "visualize_ecg_examples.py",
    "visualize_pixel_grid.py",
    "visualize_first_conv_filters.py",
    #"visualize_feature_maps.py",
    "visualize_umap_features.py",
    "visualize_super_grid_all.py",
]


def run_script(name: str):
    script_path = ROOT / name
    print(f"\n=== Starte {name} ===")
    subprocess.run([sys.executable, str(script_path)], check=True)
    print(f"=== Fertig: {name} ===")


if __name__ == "__main__":
    for script in VIS_SCRIPTS:
        run_script(script)

    print("\n[Check] Alle Visualisierungs-Skripte wurden ausgeführt.")
    print("   Die Abbildungen findest du unter 'outputs/thesis_figures/'.")
