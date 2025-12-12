from pathlib import Path
import csv

src = Path("data/ecg_test")
dst = Path("data/ecg_test_flat")
dst.mkdir(exist_ok=True)

classes = ["Abnormal", "HistoryMI", "MI", "Normal"]
exts = ("*.jpg", "*.jpeg", "*.png")

rows = []
i = 0

for cls in classes:
    files = []
    for pat in exts:
        files += sorted((src / cls).glob(pat))

    for img in files:
        fname = f"{i:03d}{img.suffix.lower()}"
        rows.append([fname, cls])
        i += 1

csv_path = dst / "labels.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "class"])
    w.writerows(rows)

print("labels.csv written to:", csv_path)
print("Number of entries:", len(rows))
