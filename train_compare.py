import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


TARGET_CLASSES = ["COVID19", "PNEUMONIA", "TURBERCULOSIS"]
DISPLAY_NAME_MAP = {"TURBERCULOSIS": "TUBERCULOSIS"}


@dataclass
class TrainResult:
    model_name: str
    best_val_acc: float
    test_acc: float
    checkpoint_path: str


class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root: str, include_classes: List[str], transform=None):
        super().__init__(root=root, transform=transform)
        include_set = set(include_classes)

        samples = []
        for path, class_idx in self.samples:
            class_name = self.classes[class_idx]
            if class_name in include_set:
                samples.append((path, class_name))

        self.classes = sorted(include_classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = [(p, self.class_to_idx[c]) for p, c in samples]
        self.targets = [label for _, label in self.samples]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_transforms(img_size: int) -> Dict[str, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return {"train": train_tfms, "val": eval_tfms, "test": eval_tfms}


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name == "cnn":
        return SimpleCNN(num_classes)
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> Tuple[float, float]:
    model.train(train)
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate(model, loader, device) -> Tuple[float, Dict]:
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / max(total, 1)
    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    return acc, report


def train_one_model(
    model_name: str,
    dataloaders: Dict[str, DataLoader],
    args,
    class_names: List[str],
    device: torch.device,
) -> TrainResult:
    model = build_model(model_name, num_classes=len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{model_name}_best.pt"

    patience = args.patience
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, dataloaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, dataloaders["val"], criterion, optimizer, device, train=False)
        scheduler.step()

        print(
            f"[{model_name}] Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "img_size": args.img_size,
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{model_name}] Early stopping triggered.")
                break

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_acc, report = evaluate(model, dataloaders["test"], device)

    report_path = output_dir / f"{model_name}_classification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return TrainResult(
        model_name=model_name,
        best_val_acc=best_val_acc,
        test_acc=test_acc,
        checkpoint_path=str(ckpt_path),
    )


def make_dataloaders(data_root: str, img_size: int, batch_size: int, num_workers: int):
    tfms = get_transforms(img_size)
    datasets_by_split = {
        split: FilteredImageFolder(
            root=str(Path(data_root) / split),
            include_classes=TARGET_CLASSES,
            transform=tfms[split],
        )
        for split in ["train", "val", "test"]
    }
    class_names = datasets_by_split["train"].classes
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        for split, ds in datasets_by_split.items()
    }
    return loaders, class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Compare chest X-ray classifiers for 3 disease classes.")
    parser.add_argument("--data-root", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--models", nargs="+", default=["cnn", "resnet50", "densenet121", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, class_names = make_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Training classes:", [DISPLAY_NAME_MAP.get(c, c) for c in class_names])
    results = []
    for name in args.models:
        print(f"\n=== Training {name} ===")
        result = train_one_model(name, loaders, args, class_names, device)
        results.append(asdict(result))

    df = pd.DataFrame(results).sort_values("test_acc", ascending=False)
    out_csv = Path(args.output_dir) / "model_comparison.csv"
    out_json = Path(args.output_dir) / "model_comparison.json"
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient="records", indent=2)

    print("\nModel comparison:")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
