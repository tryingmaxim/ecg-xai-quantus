from pathlib import Path

DATA_TRAIN = Path("data/ecg_train")
DATA_VAL   = Path("data/ecg_val")
DATA_TEST  = Path("data/ecg_test")

CLASSES = ["Abnormal", "HistoryMI", "MI", "Normal"]

IMG_SIZE = 224 #Bildgröße für das Modell 
BATCH_SIZE = 64 #Batch-Größe für das Training
EPOCHS = 15 #Anzahl der Trainings-Epochen
LR = 1e-3 #Lernrate 0.001quan
WEIGHT_DECAY = 1e-4 #Gewichtszerfall für den Optimierer
NUM_WORKERS = 4 #Parallele Datenladeprozesse
SEED = 42 #Zufallstartwert 

OUT_DIR = Path("outputs")
CKPT_DIR = OUT_DIR / "checkpoints" #Modell
EXPL_DIR = OUT_DIR / "explanations" #Heatmap von OmniXAI
METRICS_DIR = OUT_DIR / "metrics" #CSV von Quantus Metriken
THESIS_DIR   = OUT_DIR / "thesis_figures"
THESIS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "densenet121",
    "densenet169",
    "vgg16_bn",
    "efficientnet_b0",
    "efficientnet_b1",
    "mobilenet_v2",
]
 #CNN Modellname
EARLY_STOP_PATIENCE = 5  #Falls keine Verbseserung, stoppe nach 5 Epochen

USE_GPU = True
USE_AMP = True
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2