import random, shutil
from pathlib import Path

root = Path("data/ecg_images")
classes = ["Normal","MI","Abnormal","HistoryMI"]
splits = {"train":0.7, "val":0.15, "test":0.15}

for split in splits: 
    for c in classes:
        (root.parent / f"ecg_{split}" / c).mkdir(parents=True, exist_ok=True)

random.seed(42)
for c in classes:
    imgs = sorted((root / c).glob("*.*"))
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n*splits["train"])
    n_val   = int(n*splits["val"])
    parts = {
        "train": imgs[:n_train],
        "val":   imgs[n_train:n_train+n_val],
        "test":  imgs[n_train+n_val:],
    }
    for split, files in parts.items():
        for f in files:
            shutil.copy2(f, root.parent / f"ecg_{split}" / c / f.name)

print("Fertig: data/ecg_train, data/ecg_val, data/ecg_test")
