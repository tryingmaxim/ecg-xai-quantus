from pathlib import Path
import shutil

src = Path("data/ecg_test")
dst = Path("data/ecg_test_flat")
dst.mkdir(exist_ok=True)

exts = ("*.jpg", "*.jpeg", "*.png")

i = 0
for cls in ["Abnormal", "HistoryMI", "MI", "Normal"]:
    files = []
    for pat in exts:
        files += sorted((src / cls).glob(pat))

    for img in files:
        # Endung behalten (jpg bleibt jpg)
        shutil.copy(img, dst / f"{i:03d}{img.suffix.lower()}")
        i += 1

print("Copied images:", i)
