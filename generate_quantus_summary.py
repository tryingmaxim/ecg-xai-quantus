# generate_quantus_summary.py
import pandas as pd
from pathlib import Path

def main():
    in_dir = Path("outputs/metrics/quantus_raw")
    out_file = Path("outputs/metrics/quantus_summary.csv")

    csvs = sorted(in_dir.glob("*.csv"))
    if not csvs:
        print("[ERROR] Keine quantus_*.csv gefunden in quantus_raw/")
        return

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    print("[OK] Summary gespeichert unter:", out_file)
    print(df)

if __name__ == "__main__":
    main()
