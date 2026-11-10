"""
Deep learning baselines for UCI HAR (Human Activity Recognition Using Smartphones).

Trains 1D-CNN / LSTM / CNN-LSTM models in PyTorch on the RAW inertial signals
(9 channels x 128 timesteps per window) and evaluates them with subject-wise
GroupKFold + macro-F1, so results are directly comparable with the
scikit-learn models trained on the 561 hand-crafted features.

Usage (Colab or local):
    pip install torch scikit-learn numpy pandas
    python har_pytorch.py --model cnn                # downloads the dataset if needed
    python har_pytorch.py --model lstm --epochs 40
    python har_pytorch.py --model all                # cnn, lstm, cnn_lstm
    python har_pytorch.py --synthetic --epochs 2     # smoke test without the dataset

Outputs (in --out-dir, default ./results):
    <model>_folds.csv      per-fold accuracy / macro-F1 / best epoch
    <model>_summary.json   mean +- std, pooled confusion matrix, config
    stdout                 one Markdown row ready to paste into MODEL_LOG.md
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
ACTIVITIES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
N_CLASSES = len(ACTIVITIES)


@dataclass
class Config:
    model: str = "cnn"
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6          # early stopping on inner-validation macro-F1
    n_splits: int = 5          # outer GroupKFold folds (grouped by subject)
    val_frac: float = 0.2      # fraction of TRAINING subjects held out for early stopping
    seed: int = 42


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def download_dataset(data_dir: Path) -> Path:
    """Download and extract UCI HAR; return the 'UCI HAR Dataset' folder."""
    root = data_dir / "UCI HAR Dataset"
    if root.exists():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading UCI HAR to {data_dir} ...")
    with urllib.request.urlopen(DATASET_URL) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    # The UCI archive wraps the real dataset in a nested zip.
    inner_name = next((n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip")), None)
    if inner_name:
        zipfile.ZipFile(io.BytesIO(outer.read(inner_name))).extractall(data_dir)
    else:
        outer.extractall(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Expected {root} after extraction.")
    return root


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (N, 9, 128), y (N,) in 0..5, subjects (N,) for 'train' or 'test'."""
    sig_dir = root / split / "Inertial Signals"
    X = np.stack(
        [np.loadtxt(sig_dir / f"{s}_{split}.txt", dtype=np.float32) for s in SIGNALS],
        axis=1,
    )
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64)
    return X, y, subjects


def load_har(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge the official train/test splits; folds are re-made by subject below."""
    root = download_dataset(data_dir)
    parts = [load_split(root, s) for s in ("train", "test")]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    groups = np.concatenate([p[2] for p in parts])
    return X, y, groups


def make_synthetic(n_subjects: int = 10, per_subject: int = 60, seed: int = 0):
    """Fake data with the real shapes, for a quick end-to-end smoke test."""
    rng = np.random.default_rng(seed)
    n = n_subjects * per_subject
    y = rng.integers(0, N_CLASSES, n)
    t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
    X = rng.normal(0, 1, (n, len(SIGNALS), 128)).astype(np.float32)
    X += np.sin((y[:, None, None] + 1) * t).astype(np.float32)  # (N, 1, 128) broadcast to 9 channels
    groups = np.repeat(np.arange(1, n_subjects + 1), per_subject)
    return X, y.astype(np.int64), groups


class ChannelStandardizer:
    """Per-channel z-score. Fit on training windows only (prevents leakage)."""

    def fit(self, X: np.ndarray) -> "ChannelStandardizer":
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)


def to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------- #
# Models  (input: batch x channels(9) x time(128))
# --------------------------------------------------------------------------- #
class CNN1D(nn.Module):
    def __init__(self, in_ch: int = 9, n_classes: int = N_CLASSES, width: int = 64, dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out, k):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
            )

        self.features = nn.Sequential(
            block(in_ch, width, 7),
            block(width, width, 5),
            nn.MaxPool1d(2),
            block(width, width * 2, 5),
            block(width * 2, width * 2, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 2, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class LSTMNet(nn.Module):
    def __init__(self, in_ch: int = 9, n_classes: int = N_CLASSES, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, n_classes))

    def forward(self, x):
        out, _ = self.lstm(x.transpose(1, 2))   # (B, T, C)
        return self.head(out.mean(dim=1))       # mean-pool over time


class CNNLSTM(nn.Module):
    """Conv layers extract local motion patterns; the LSTM models their order."""

    def __init__(self, in_ch: int = 9, n_classes: int = N_CLASSES, width: int = 64,
                 hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(width, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        h = self.conv(x).transpose(1, 2)        # (B, T/2, width)
        _, (h_n, _) = self.lstm(h)
        return self.head(h_n[-1])


MODELS = {"cnn": CNN1D, "lstm": LSTMNet, "cnn_lstm": CNNLSTM}


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = [model(xb.to(device)).argmax(1).cpu().numpy() for xb, _ in loader]
    return np.concatenate(preds)


def train_one_model(cfg: Config, X_tr, y_tr, X_val, y_val, device) -> tuple[nn.Module, int, float]:
    """Train with early stopping on validation macro-F1; return best model, epoch, F1."""
    model = MODELS[cfg.model]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = to_loader(X_tr, y_tr, cfg.batch_size, shuffle=True)
    val_loader = to_loader(X_val, y_val, cfg.batch_size, shuffle=False)

    best_f1, best_epoch, best_state, bad_epochs = -1.0, 0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)

        val_f1 = f1_score(y_val, predict(model, val_loader, device), average="macro")
        print(f"    epoch {epoch:3d}  loss {running / len(X_tr):.4f}  val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"    early stop (best epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_f1


def cross_validate(cfg: Config, X, y, groups, device) -> dict:
    """Outer GroupKFold by subject; inner subject-wise split for early stopping."""
    outer = GroupKFold(n_splits=cfg.n_splits)
    fold_rows, y_true_all, y_pred_all = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y, groups), start=1):
        set_seed(cfg.seed + fold)
        # Inner split: hold out a few TRAINING subjects for early stopping.
        inner = GroupShuffleSplit(n_splits=1, test_size=cfg.val_frac, random_state=cfg.seed + fold)
        fit_rel, val_rel = next(inner.split(X[tr_idx], y[tr_idx], groups[tr_idx]))
        fit_idx, val_idx = tr_idx[fit_rel], tr_idx[val_rel]

        # Sanity check: no subject appears on both sides.
        assert not set(groups[fit_idx]) & set(groups[te_idx])
        assert not set(groups[val_idx]) & set(groups[te_idx])

        scaler = ChannelStandardizer().fit(X[fit_idx])
        X_fit, X_val, X_te = (scaler.transform(X[i]) for i in (fit_idx, val_idx, te_idx))

        print(f"Fold {fold}/{cfg.n_splits}: test subjects {sorted(set(groups[te_idx].tolist()))}")
        t0 = time.time()
        model, best_epoch, val_f1 = train_one_model(cfg, X_fit, y[fit_idx], X_val, y[val_idx], device)
        y_pred = predict(model, to_loader(X_te, y[te_idx], cfg.batch_size, False), device)

        row = {
            "fold": fold,
            "test_subjects": " ".join(map(str, sorted(set(groups[te_idx].tolist())))),
            "best_epoch": best_epoch,
            "val_macro_f1": round(val_f1, 4),
            "test_accuracy": round(accuracy_score(y[te_idx], y_pred), 4),
            "test_macro_f1": round(f1_score(y[te_idx], y_pred, average="macro"), 4),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  -> test macro-F1 {row['test_macro_f1']:.4f}  acc {row['test_accuracy']:.4f}")
        fold_rows.append(row)
        y_true_all.append(y[te_idx])
        y_pred_all.append(y_pred)

    y_true_all, y_pred_all = np.concatenate(y_true_all), np.concatenate(y_pred_all)
    f1s = np.array([r["test_macro_f1"] for r in fold_rows])
    return {
        "folds": fold_rows,
        "macro_f1_mean": float(f1s.mean()),
        "macro_f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
        "per_class_f1": dict(zip(ACTIVITIES, f1_score(y_true_all, y_pred_all, average=None).round(4).tolist())),
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all, labels=range(N_CLASSES)).tolist(),
    }


def save_results(cfg: Config, res: dict, out_dir: Path, n_params: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res["folds"]).to_csv(out_dir / f"{cfg.model}_folds.csv", index=False)
    summary = {k: v for k, v in res.items() if k != "folds"}
    summary.update(config=asdict(cfg), n_params=n_params, labels=ACTIVITIES)
    (out_dir / f"{cfg.model}_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nPer-class F1:", res["per_class_f1"])
    print("\nMODEL_LOG.md row:")
    print("| Model | Input | CV | Macro-F1 (mean ± std) | Params | Notes |")
    print("|---|---|---|---|---|---|")
    print(f"| PyTorch {cfg.model} | raw 9x128 | GroupKFold({cfg.n_splits}) by subject | "
          f"{res['macro_f1_mean']:.4f} ± {res['macro_f1_std']:.4f} | {n_params:,} | "
          f"lr={cfg.lr}, bs={cfg.batch_size}, early stop on inner subject split |")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="cnn", choices=[*MODELS, "all"])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--n-splits", type=int, default=Config.n_splits)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--synthetic", action="store_true", help="use fake data (smoke test)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y, groups = make_synthetic() if args.synthetic else load_har(args.data_dir)
    print(f"Data: X {X.shape}, {len(np.unique(groups))} subjects, class counts {np.bincount(y).tolist()}")

    model_names = list(MODELS) if args.model == "all" else [args.model]
    for name in model_names:
        cfg = Config(model=name, epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr, n_splits=args.n_splits, seed=args.seed)
        print(f"\n===== {name} =====")
        set_seed(cfg.seed)
        n_params = sum(p.numel() for p in MODELS[name]().parameters())
        res = cross_validate(cfg, X, y, groups, device)
        save_results(cfg, res, args.out_dir, n_params)


if __name__ == "__main__":
    main()
