#Standard Tools für Timing, Zufall oder auch GPU-Training
import time, random, numpy as np  
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch import amp
from pathlib import Path
from torchvision import datasets, transforms
import pandas as pd
from . import configs
from .model_def import build_model
from .utils import make_loaders

#Stelle sicher, dass die benötigten Ordner existieren
configs.OUT_DIR.mkdir(exist_ok=True)
configs.CKPT_DIR.mkdir(parents=True, exist_ok=True)
configs.METRICS_DIR.mkdir(parents=True, exist_ok=True)

#Nimmt immer den gleichen Algorithmus für Zufallszahlen heißt die Traningsläufe sind reproduzierbar gleiches Setup gleiche Ergebnisse
def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Nur wichtig fürs Debugging, wählt ob GPU oder CPU genutzt wird
def get_device():
    use_gpu = getattr(configs, "USE_GPU", True)
    dev = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Device: {dev}")
    return dev

#Umwandeln in img_size und 3-Kanal-Graustufen für ImageFolder und liest die Ordnernamen aus
def infer_dataset_classes(train_dir: Path, img_size: int):
    if not train_dir.exists():
        return None
    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(str(train_dir), tfms)
    return list(ds.classes)


def train_one_model(model_name: str):
    #setzt den Seed also reproduzierbare Ergebnisse und hold das Gerät (CPU/GPU)
    set_seed(configs.SEED)
    device = get_device()
    #Versucht die Klassen aus dem ImageFolder-Ordner zu ermitteln
    train_dir = getattr(configs, "DATA_TRAIN", Path("data/ecg_train"))
    ds_classes = infer_dataset_classes(train_dir, configs.IMG_SIZE)

    if ds_classes is None:
        print(f"[WARN] Konnte Klassen aus '{train_dir}' nicht ermitteln. "
              f"Nutze configs.CLASSES wie konfiguriert.")
        classes_to_use = list(configs.CLASSES)
    else:
        cfg_classes = list(configs.CLASSES)
        if cfg_classes != ds_classes:
            print("[WARN] Klassenreihenfolge unterscheidet sich!")
            print(f"  configs.CLASSES = {cfg_classes}")
            print(f"  ImageFolder     = {ds_classes}")
            print("[INFO] Schalte automatisch auf ImageFolder-Order um.")
            classes_to_use = ds_classes
        else:
            classes_to_use = cfg_classes
    
    #lädt die Daten        
    train_loader, val_loader, _ = make_loaders()
    
    print(f"\n[INFO] Starte Training für Modell: {model_name}")
    #Erstellt Modell, Loss-Funktion und Optimierer
    model = build_model(model_name, len(classes_to_use)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=configs.LR, weight_decay=configs.WEIGHT_DECAY)

    use_amp = getattr(configs, "USE_AMP", True) and device.type == "cuda"
    scaler = amp.GradScaler(enabled=use_amp)

    #merkt sich das beste Ergebnis fürs Early Stopping
    best_acc = 0.0
    patience = getattr(configs, "EARLY_STOP_PATIENCE", 5)
    no_improve = 0
    hist_epochs = []
    hist_train_loss = []
    hist_val_loss = []
    hist_train_acc = []
    hist_val_acc = []

    scheduler = None
    if getattr(configs, "USE_COSINE_SCHEDULER", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.EPOCHS)

    print(f"[INFO] Start Training for {configs.EPOCHS} epochs | AMP={use_amp}")
    print(f"[INFO] Klassen (Train-Order): {classes_to_use}")

    #Haupt-Trainingsschleife
    for epoch in range(1, configs.EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for x, y in train_loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast('cuda', enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        
        #Validierungsschritt am Ende jeder Epoche 
        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad(), amp.autocast('cuda', enabled=use_amp):
            for x, y in val_loader:
                x = x.to(device, non_blocking=(device.type == "cuda"))
                y = y.to(device, non_blocking=(device.type == "cuda"))
                logits = model(x)
                loss = criterion(logits, y)
                vloss += loss.item() * x.size(0)
                vcorrect += (logits.argmax(1) == y).sum().item()
                vtotal += y.size(0)

        val_loss = vloss / max(vtotal, 1)
        val_acc = vcorrect / max(vtotal, 1)

        hist_epochs.append(epoch)
        hist_train_loss.append(train_loss)
        hist_val_loss.append(val_loss)
        hist_train_acc.append(train_acc)
        hist_val_acc.append(val_acc)
        dt = time.time() - t0
        print(f"[{model_name}] Epoch {epoch:02d}/{configs.EPOCHS} | "
              f"train {train_loss:.4f}/{train_acc:.3f} | "
              f"val {val_loss:.4f}/{val_acc:.3f} | {dt:.1f}s")

        if scheduler is not None:
            scheduler.step()

        #Speichert das beste Modell basierend auf Validierungsgenauigkeit
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            ckpt_path = configs.CKPT_DIR / f"{model_name}_best.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes_to_use,
                "img_size": configs.IMG_SIZE,
                "model_name": model_name,
            }, ckpt_path)
            print(f"[{model_name}] 🔥 Neues bestes Modell gespeichert: {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{model_name}] Early stopping.")
                break
    
    model_metric_dir = configs.METRICS_DIR / model_name
    model_metric_dir.mkdir(parents=True, exist_ok=True)

    hist_path = model_metric_dir / "history.csv"

    df_hist = pd.DataFrame({
        "epoch": hist_epochs,
        "train_loss": hist_train_loss,
        "val_loss": hist_val_loss,
        "train_acc": hist_train_acc,
        "val_acc": hist_val_acc,
    })
    df_hist.to_csv(hist_path, index=False)
    print(f"[{model_name}] Trainings-History gespeichert nach: {hist_path}")

    print(f"[{model_name}] Best val acc: {best_acc:.3f}")
#Führt das Training für alle in den configs.py angegebenen Modelle durch
if __name__ == "__main__":
    for model_name in configs.MODEL_NAMES:
        train_one_model(model_name)
