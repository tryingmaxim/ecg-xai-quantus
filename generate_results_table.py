# generate_results_table.py
import re
import pandas as pd
from pathlib import Path

METRICS_DIR = Path("outputs/metrics")
OUT_FILE = METRICS_DIR / "model_performance.csv"

def parse_classification_report(txt_path: Path):
    text = txt_path.read_text(encoding="utf-8")

    m = re.search(r"Accuracy:\s*([0-9.]+)", text)
    acc = float(m.group(1)) if m else None

    macro_f1 = None
    weighted_f1 = None

    lines = text.strip().splitlines()
    for line in lines:
        if line.strip().startswith("macro avg"):
            parts = line.split()
            macro_f1 = float(parts[-2])
        if line.strip().startswith("weighted avg"):
            parts = line.split()
            weighted_f1 = float(parts[-2])

    return acc, macro_f1, weighted_f1


def main():
    rows = []
    for model_dir in METRICS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        rpt = model_dir / "classification_report.txt"
        if not rpt.exists():
            continue

        acc, macro_f1, weighted_f1 = parse_classification_report(rpt)
        rows.append({
            "model": model_dir.name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })

    if not rows:
        print("[ERROR] Keine classification_report.txt gefunden.")
        return

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print("[OK] Modell-Performance gespeichert unter:", OUT_FILE)
    print(df)


if __name__ == "__main__":
    main()
