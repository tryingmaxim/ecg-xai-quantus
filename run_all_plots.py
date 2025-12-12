# starten mit: python run_all_plots.py
# führt alle Plot-Skripte nacheinander aus
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PLOT_SCRIPTS = [
    "generate_quantus_summary.py",
    "generate_results_table.py",
    "analyze_dataset_distribution.py",
    "analyze_dataset_split.py",
    "generate_results_table.py",
    "generate_quantus_summary.py",
    "plot_training_history.py",
    "plot_confidence_histogram.py",
    "plot_quantus_results.py",
    "plot_heatmap_grid.py",
    "plot_performance_vs_xai.py",
    "plot_quantus_radar.py",
    "plot_quantus_scatter.py",
    "plot_sensitivity_analysis.py",
]


def run_script(name: str):
    script_path = ROOT / name
    print(f"\n=== Starte {name} ===")
    subprocess.run([sys.executable, str(script_path)], check=True)
    print(f"=== Fertig: {name} ===")


if __name__ == "__main__":
    for script in PLOT_SCRIPTS:
        run_script(script)

    print("\n[Check] Alle Plot-Skripte wurden ausgeführt.")
    print("   Wichtig: 'outputs/metrics/plots' und 'outputs/thesis_figures/' ansehen.")
