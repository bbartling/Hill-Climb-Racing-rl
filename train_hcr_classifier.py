# train_hcr_classifier.py
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_hcr_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.replace("", pd.NA, inplace=True)

    # Ensure required numerics
    for col in ["angle", "height_px", "ground_slope", "gas_pressed", "brake_pressed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Choose features dynamically (prefer 3 if slope exists)
    feat_cols = ["angle", "height_px"]
    if "ground_slope" in df.columns:
        feat_cols.append("ground_slope")

    df.dropna(subset=feat_cols, inplace=True)

    X = df[feat_cols].values.astype(np.float32)
    y = np.select(
        [(df.get("gas_pressed", 0) == 1), (df.get("brake_pressed", 0) == 1)],
        [1, 2],
        default=0,
    ).astype(np.int64)
    return X, y, feat_cols


class HCRDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(
            self.y[idx], dtype=torch.long
        )


class HCRModel(nn.Module):
    def __init__(self, input_size=2, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train(
    csv_path: Path,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 10,
    seed: int = 42,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    out_dir = csv_path.parent / "ml_training_loss_plots"
    ensure_dir(out_dir)
    best_model_path = csv_path.parent / "hcr_model_best.pth"
    scaler_path = csv_path.parent / "hcr_scaler.pkl"
    report_path = out_dir / "classification_report.txt"
    cm_path = out_dir / "confusion_matrix.png"
    curves_path = out_dir / "training_curves.png"
    summary_path = out_dir / "summary.json"

    X, y, feat_cols = load_hcr_csv(csv_path)
    input_size = X.shape[1]
    print(f"Feature columns: {feat_cols} (input_size={input_size})")

    if len(X) < 10:
        raise SystemExit("Not enough samples to train. Collect more gameplay.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to: {scaler_path}")

    train_ds = HCRDataset(X_train, y_train)
    val_ds = HCRDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = HCRModel(input_size=input_size).to(device)

    classes, counts = np.unique(y_train, return_counts=True)
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-12)
    weights = weights / weights.sum() * len(classes)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print("Class counts (train):", dict(zip(classes, counts)))
    print("Loss weights:", weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    # Baseline with same feature set (scaled)
    logit = LogisticRegression(max_iter=500, class_weight="balanced").fit(
        X_train, y_train
    )
    baseline_acc = accuracy_score(y_val, logit.predict(X_val))
    print(f"Logistic Regression baseline val acc: {baseline_acc:.3f}")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / max(1, len(train_loader))

        model.eval()
        val_loss_sum = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item()
                preds = logits.argmax(dim=1)
                total += yb.size(0)
                correct += (preds == yb).sum().item()

        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        val_acc = 100.0 * correct / max(1, total)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}"
                )
                break

    print(f"Best model saved to: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.numpy())

    class_names = ["Coast", "Gas", "Brake"]
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print("\nClassification Report:\n", report)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved classification report to: {report_path}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig_cm, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(class_names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path, dpi=200)
    plt.close(fig_cm)
    print(f"Saved confusion matrix to: {cm_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(history["val_accuracy"], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(curves_path, dpi=200)
    plt.close(fig)
    print(f"Saved training curves to: {curves_path}")

    summary = {
        "device": device,
        "samples_total": int(len(X)),
        "samples_train": int(len(X_train)),
        "samples_val": int(len(X_val)),
        "best_val_loss": float(best_val_loss),
        "baseline_logreg_val_acc": float(baseline_acc),
        "features": feat_cols,
        "paths": {
            "best_model": str(best_model_path),
            "scaler": str(scaler_path),
            "classification_report": str(report_path),
            "confusion_matrix": str(cm_path),
            "training_curves": str(curves_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a 3-class action classifier from HCR CSV (angle, height, optional ground_slope)."
    )
    p.add_argument("csv_file", type=Path, help="Path to manual_play_data.csv")
    p.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (epochs)"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        csv_path=args.csv_file,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
