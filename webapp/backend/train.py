"""
Trains the dog-emotion classifier (angry vs happy) with transfer learning
on top of a MobileNetV2 backbone pretrained on ImageNet.

Data layout expected (relative to repo root):
    Train/angry/*.jpg
    Train/happy/*.jpg

Usage:
    python3 backend/train.py
"""
import copy
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "Train"
OUT_PATH = Path(__file__).resolve().parent / "model.pt"
CLASSES = ["angry", "happy"]  # alphabetical, matches ImageFolder ordering
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_model():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False
    for p in m.features[-4:].parameters():
        p.requires_grad = True
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.last_channel, len(CLASSES)),
    )
    return m


def main():
    full_ds = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    assert full_ds.classes == CLASSES, f"unexpected class order: {full_ds.classes}"

    n_val = int(len(full_ds) * 0.15)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    # val split should use non-augmented transform
    val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=val_tf)

    # class weights to correct for angry/happy imbalance
    targets = [full_ds.samples[i][1] for i in train_ds.indices]
    counts = [targets.count(c) for c in range(len(CLASSES))]
    weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    best_acc = 0.0
    best_state = None

    print(f"device={device} train={len(train_ds)} val={len(val_ds)} counts={dict(zip(CLASSES, counts))}")

    for epoch in range(EPOCHS):
        model.train()
        running_loss, running_correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)
        train_loss = running_loss / seen
        train_acc = running_correct / seen

        model.eval()
        val_correct, val_seen = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_seen += x.size(0)
        val_acc = val_correct / val_seen

        print(f"epoch {epoch+1}/{EPOCHS} train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    print(f"best val_acc={best_acc:.3f}")
    torch.save({"state_dict": best_state, "classes": CLASSES, "img_size": IMG_SIZE}, OUT_PATH)
    print(f"saved model to {OUT_PATH}")


if __name__ == "__main__":
    main()
