import torch
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple
from torch import nn

# --- Classifier training and evaluation functions ---

def train_classifier_one_epoch(model, loader, optimizer, criterion, device="cpu", log_every=0):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    for it, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device)            # (B, 1, 16, 16)
        yb = yb.to(device)            # (B,)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)            # (B, C)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            running_loss += loss.item() * xb.size(0)
            running_correct += (logits.argmax(1) == yb).sum().item()
            running_count += xb.size(0)

        if log_every and it % log_every == 0:
            avg_loss = running_loss / running_count
            avg_acc = running_correct / running_count

    epoch_loss = running_loss / max(running_count, 1)
    epoch_acc = running_correct / max(running_count, 1)
    return {"loss": epoch_loss, "acc": epoch_acc}

@torch.no_grad()
def eval_classifier(model, loader, criterion, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_count += xb.size(0)

    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1)
    }

def fit_classifier(model, train_loader, val_loader, optimizer, criterion, device="cpu", epochs=10):
    history = {"train": {"loss": [], "acc": []}, "val": {"loss": [], "acc": []}}

    pbar = tqdm(range(1, epochs + 1), desc="Training (epochs)", leave=True)
    for epoch in pbar:
        train_metrics = train_classifier_one_epoch(model, train_loader, optimizer, criterion, device, log_every=0)
        val_metrics = eval_classifier(model, val_loader, criterion, device)

        history["train"]["loss"].append(train_metrics["loss"])
        history["train"]["acc"].append(train_metrics["acc"])
        history["val"]["loss"].append(val_metrics["loss"])
        history["val"]["acc"].append(val_metrics["acc"])

        pbar.set_postfix({
            "train_loss": f"{train_metrics['loss']:.4f}",
            "train_acc":  f"{train_metrics['acc']:.4f}",
            "val_loss":   f"{val_metrics['loss']:.4f}",
            "val_acc":    f"{val_metrics['acc']:.4f}"
        })

    return history, model

# --- Masked Autoencoder training and evaluation functions ---

def train_mae_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
    log_every: int = 0,
) -> Dict[str, float]:
    """
    One training epoch for TinyMAE.

    Uses the model's reconstruction_loss over MASKED patches as objective.
    """
    model.train()
    running_loss = 0.0
    running_count = 0

    for it, (xb, *rest) in enumerate(loader, start=1):
        # Accept loaders that yield (images,) or (images, labels)
        xb = xb.to(device, non_blocking=True)                    # (B, 1, 16, 16)

        optimizer.zero_grad(set_to_none=True)
        recon, mask = model(xb)                                  # (B, N, P), (B, N)
        loss = model.reconstruction_loss(xb, recon, mask)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            bsz = xb.size(0)
            running_loss += loss.item() * bsz
            running_count += bsz

        if log_every and it % log_every == 0:
            avg_loss = running_loss / max(1, running_count)
            print(f"[train] it={it:04d}  loss={avg_loss:.4f}")

    avg_loss = running_loss / max(1, running_count)
    return {"loss": float(avg_loss)}


@torch.no_grad()
def _knn_eval_from_embeddings(
    feats: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    metric: str = "cosine",
    leave_one_out: bool = True,
) -> float:
    """
    K-NN classification accuracy given a full set of embeddings and labels.

    Args:
        feats: (N, D) L2-normalized features.
        labels: (N,) int labels.
        k: number of neighbors.
        metric: "cosine" or "euclidean".
        leave_one_out: if True, excludes the sample itself from neighbors (useful when
                       feats come from the same split).

    Returns:
        Top-1 accuracy in [0, 100].
    """
    N, D = feats.shape
    if metric == "cosine":
        # feats already normalized -> cosine similarity via dot product
        sim = feats @ feats.t()                                  # (N, N)
        if leave_one_out:
            sim.fill_diagonal_(-1.0)
        # topk on similarity
        vals, idx = torch.topk(sim, k=k, dim=1, largest=True)
    elif metric == "euclidean":
        # pairwise distances
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b ; feats normalized -> ||a||^2 = 1
        # so dist^2 = 2 - 2cos ; sorting by smallest distance == sorting by largest cos
        sim = feats @ feats.t()
        if leave_one_out:
            sim.fill_diagonal_(-1.0)
        vals, idx = torch.topk(sim, k=k, dim=1, largest=True)
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    # majority vote among neighbor labels (optionally weighted by similarity)
    neighbor_labels = labels[idx]                                # (N, k)
    # simple unweighted voting (works well with normalized embeddings)
    # for weighted voting, use vals as weights.
    preds = torch.mode(neighbor_labels, dim=1).values            # (N,)
    acc = (preds == labels).float().mean().item() * 100.0
    return acc


@torch.no_grad()
def eval_mae(
    model: nn.Module,
    loader,
    device: str | torch.device = "cpu",
    k: int = 5,
    metric: str = "cosine",
    leave_one_out: bool = True,
) -> Dict[str, float]:
    """
    Evaluate TinyMAE with two metrics:
      - Reconstruction loss (masked-patch MSE) computed under the model's masking.
      - K-NN accuracy from encoder embeddings computed on the SAME split.

    The K-NN evaluation performs leave-one-out classification on the split's embeddings.
    """
    model.eval()
    all_feats = []
    all_labels = []
    running_loss = 0.0
    running_count = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            xb, yb = batch
        else:
            xb, yb = batch, None

        xb = xb.to(device, non_blocking=True)
        recon, mask = model(xb)
        loss = model.reconstruction_loss(xb, recon, mask)

        running_loss += loss.item() * xb.size(0)
        running_count += xb.size(0)

        # extract encoder features (mean-pooled visible tokens)
        feats = model.mae_encode_features(xb, device=device)     # (B, D)
        all_feats.append(feats.cpu())
        if yb is not None:
            all_labels.append(yb.detach().cpu())

    avg_loss = running_loss / max(1, running_count)
    if len(all_labels) == 0:
        # if no labels are provided, only report loss
        return {"loss": float(avg_loss), "knn_acc": float("nan")}

    feats = torch.cat(all_feats, dim=0)                           # (N, D)
    labels = torch.cat(all_labels, dim=0).long()                  # (N,)

    knn_acc = _knn_eval_from_embeddings(
        feats, labels, k=k, metric=metric, leave_one_out=leave_one_out
    )
    return {"loss": float(avg_loss), "knn_acc": float(knn_acc)}


def fit_mae(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
    epochs: int = 10,
    k: int = 5,
    metric: str = "cosine",
) -> Tuple[Dict[str, Dict[str, list]], nn.Module]:
    """
    Train TinyMAE and track reconstruction loss and K-NN accuracy (from encoder features).

    History format:
        history = {
            "train": {"loss": [], "knn_acc": []},
            "val":   {"loss": [], "knn_acc": []},
        }
    """
    history = {
        "train": {"loss": [], "knn_acc": []},
        "val": {"loss": [], "knn_acc": []},
    }

    pbar = tqdm(range(1, epochs + 1), desc="Training MAE (epochs)", leave=True)
    for epoch in pbar:
        # ---- train one epoch (reconstruction objective) ----
        train_metrics = train_mae_one_epoch(model, train_loader, optimizer, device=device, log_every=0)

        # ---- evaluation: recon loss + K-NN on TRAIN (leave-one-out) and VAL ----
        # For logging symmetry with classifier fit, we compute K-NN on the training split
        # (leave-one-out) and report it under "train". This gives a sense of learned features.
        train_eval = eval_mae(
            model, train_loader, device=device, k=k, metric=metric, leave_one_out=True
        )
        val_eval = eval_mae(
            model, val_loader, device=device, k=k, metric=metric, leave_one_out=False
        )

        history["train"]["loss"].append(train_metrics["loss"])
        history["train"]["knn_acc"].append(train_eval["knn_acc"])
        history["val"]["loss"].append(val_eval["loss"])
        history["val"]["knn_acc"].append(val_eval["knn_acc"])

        pbar.set_postfix({
            "train_loss": f"{train_metrics['loss']:.4f}",
            "train_kNN":  f"{train_eval['knn_acc']:.2f}%",
            "val_loss":   f"{val_eval['loss']:.4f}",
            "val_kNN":    f"{val_eval['knn_acc']:.2f}%"
        })

    return history, model