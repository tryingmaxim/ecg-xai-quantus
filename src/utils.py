from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from . import configs
import random, numpy as np, torch

def _seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _worker_init_fn(wid):
    # pro-Worker deterministische Seeds
    s = configs.SEED + wid
    random.seed(s); np.random.seed(s)

def make_transforms():
    # Erzwinge 3 Kanäle (auch bei ursprünglich grauen PNGs)
    train_tfms = transforms.Compose([
        transforms.Resize((configs.IMG_SIZE, configs.IMG_SIZE)),
        transforms.RandomApply([transforms.RandomRotation(3, fill=0)], p=0.5),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((configs.IMG_SIZE, configs.IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return train_tfms, eval_tfms

def make_loaders():
    _seed_everything(configs.SEED)

    train_tfms, eval_tfms = make_transforms()
    train_ds = datasets.ImageFolder(configs.DATA_TRAIN, transform=train_tfms)
    val_ds   = datasets.ImageFolder(configs.DATA_VAL,   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(configs.DATA_TEST,  transform=eval_tfms)

    g = torch.Generator()
    g.manual_seed(configs.SEED)

    dl_args = dict(
        num_workers=configs.NUM_WORKERS,
        pin_memory=getattr(configs, "PIN_MEMORY", True),
        worker_init_fn=_worker_init_fn,
        generator=g,
    )
    # prefetch/persistent nur wenn es Worker gibt
    if dl_args["num_workers"] > 0:
        dl_args.update(
            persistent_workers=getattr(configs, "PERSISTENT_WORKERS", True),
            prefetch_factor=getattr(configs, "PREFETCH_FACTOR", 2),
        )

    train_loader = DataLoader(train_ds, batch_size=configs.BATCH_SIZE, shuffle=True,  **dl_args)
    val_loader   = DataLoader(val_ds,   batch_size=configs.BATCH_SIZE, shuffle=False, **dl_args)
    test_loader  = DataLoader(test_ds,  batch_size=configs.BATCH_SIZE, shuffle=False, **dl_args)
    return train_loader, val_loader, test_loader
