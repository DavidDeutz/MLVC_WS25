import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import math
from matplotlib.lines import Line2D

@torch.no_grad()
def plot_attn_per_head_for_query(
    model,
    image_16x16,   # torch.Tensor shape (16, 16), values in [0,1] or [0,255]
    device="cpu",
    block_idx=0,
    query_rc=None  # tuple (row, col) or None for center
):
    model.eval()
    with torch.no_grad():
        x = image_16x16
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.0:
            x = x / 255.0

        # Forward pass to populate last_attn
        _ = model(x.unsqueeze(0).unsqueeze(0).to(device))  # shape (1,1,16,16) into your model

        # Fetch attention from the chosen block
        attn = model.blocks[block_idx].attn.last_attn  # (B, H, S, S)
        attn = attn[0]  # (H, S, S)

    H, S, _ = attn.shape

    # Handle CLS token offset
    use_cls = bool(getattr(model, "use_cls_token", False))
    cls_offset = 1 if use_cls else 0
    grid_tokens = S - cls_offset
    side = int(math.isqrt(grid_tokens))
    assert side * side == grid_tokens, "Token count does not form a square grid"

    # Choose query token
    if query_rc is None:
        query_rc = (side // 2, side // 2)  # center by default
    qr, qc = query_rc
    assert 0 <= qr < side and 0 <= qc < side
    query_idx = cls_offset + qr * side + qc

    # Gather per head attention to keys for that single query
    # Shape per head: (S,)
    per_head_maps = []
    for h in range(H):
        a = attn[h, query_idx]  # (S,)
        a_img = a[cls_offset:].reshape(side, side).detach().cpu().numpy()
        per_head_maps.append(a_img)

    # Plot
    cols = H + 1
    fig, axs = plt.subplots(1, cols, figsize=(3 * cols, 3), dpi=200)
    axs[0].imshow(image_16x16.detach().cpu().numpy(), cmap="viridis")
    axs[0].set_title("Input")
    axs[0].axis("off")
    for h in range(H):
        axs[h + 1].imshow(per_head_maps[h], cmap="viridis")
        axs[h + 1].set_title(f"Head {h}  q=({qr},{qc})")
        axs[h + 1].axis("off")
    plt.tight_layout()
    plt.show()

def plot_results_attention(
    all_histories,
    title=None,
    smooth_window=0,        # e.g., 3 or 5 for simple moving average; 0 disables smoothing
    annotate_last=True,     # write the last value next to each curve
    mark_best=True,         # mark best validation epoch (min val loss and max val acc)
    markers_every=0,        # e.g., 5 to place a small marker every 5 epochs; 0 disables
    linewidth=1.8,          # line width for curves
    alpha_train=0.95,       # slight transparency for train curves
    alpha_val=0.95,         # slight transparency for val curves
):
    """
    Plots train/val loss (left) and accuracy (right) for multiple histories.

    Expected structure per history:
        hist = {
            "train": {"loss": list[float], "acc": list[float]},
            "val":   {"loss": list[float], "acc": list[float]}
        }
    """

    def _to_np(x):
        a = np.asarray(x, dtype=float)
        return a

    def _sma(x, w):
        if w <= 1:
            return x
        if len(x) < w:
            return x
        kernel = np.ones(w, dtype=float) / float(w)
        y = np.convolve(x, kernel, mode="valid")
        # pad to original length by repeating edges
        pad_left = w - 1
        left = np.full(pad_left // 2, y[0])
        right = np.full(pad_left - pad_left // 2, y[-1])
        return np.concatenate([left, y, right])

    # Collect model names in a stable order
    names = list(all_histories.keys())

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), dpi=150, constrained_layout=True)

    # Left: Loss
    axL = axs[0]
    axL.set_title("Loss")
    axL.set_xlabel("Epochs")
    axL.set_ylabel("Loss")
    axL.grid(which="both", alpha=0.35)

    # Right: Accuracy
    axR = axs[1]
    axR.set_title("Accuracy")
    axR.set_xlabel("Epochs")
    axR.set_ylabel("Accuracy (%)")
    axR.set_ylim(0, 101)
    axR.grid(which="both", alpha=0.35)

    # Prepare proxy legend entries
    model_handles = []
    style_handles = [
        Line2D([0], [0], linestyle="solid", color="black", lw=linewidth, label="Train"),
        Line2D([0], [0], linestyle="dashed", color="black", lw=linewidth, label="Val"),
    ]

    # Color cycle comes from matplotlib defaults
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"])

    # Track y-lims to autoscale with margins
    loss_vals = []
    acc_vals = []

    for i, name in enumerate(names):
        hist = all_histories[name]
        color = color_cycle[i % len(color_cycle)]
        model_handles.append(Line2D([0], [0], color=color, lw=linewidth, label=name))

        # Extract arrays and check wether acc or knn_acc
        tr_loss = _to_np(hist["train"]["loss"])
        va_loss = _to_np(hist["val"]["loss"])
        if "acc" in hist["train"]:
            tr_acc = _to_np(hist["train"]["acc"]) * 100.0
            va_acc = _to_np(hist["val"]["acc"]) * 100.0
        elif "knn_acc" in hist["train"]:
            tr_acc = _to_np(hist["train"]["knn_acc"])
            va_acc = _to_np(hist["val"]["knn_acc"])

        # Optional smoothing
        if smooth_window and smooth_window > 1:
            tr_loss = _sma(tr_loss, smooth_window)
            va_loss = _sma(va_loss, smooth_window)
            tr_acc  = _sma(tr_acc,  smooth_window)
            va_acc  = _sma(va_acc,  smooth_window)

        # Epoch axis (align to shortest to avoid indexing issues)
        n = min(len(tr_loss), len(va_loss), len(tr_acc), len(va_acc))
        epochs = np.arange(1, n + 1)
        tr_loss, va_loss = tr_loss[:n], va_loss[:n]
        tr_acc, va_acc   = tr_acc[:n], va_acc[:n]

        # Plot Loss
        marker_kw = {}
        if markers_every and markers_every > 0:
            marker_kw = {"markevery": markers_every, "marker": "o", "markersize": 3}

        axL.plot(epochs, tr_loss, linestyle="solid", color=color, lw=linewidth, alpha=alpha_train, **marker_kw)
        axL.plot(epochs, va_loss, linestyle="dashed", color=color, lw=linewidth, alpha=alpha_val, **marker_kw)

        # Plot Accuracy
        axR.plot(epochs, tr_acc, linestyle="solid", color=color, lw=linewidth, alpha=alpha_train, **marker_kw)
        axR.plot(epochs, va_acc, linestyle="dashed", color=color, lw=linewidth, alpha=alpha_val, **marker_kw)

        loss_vals.extend(tr_loss.tolist())
        loss_vals.extend(va_loss.tolist())
        acc_vals.extend(tr_acc.tolist())
        acc_vals.extend(va_acc.tolist())

        # Mark best validation epoch
        if mark_best and n > 0:
            # Best val loss (min) and best val acc (max)
            best_loss_ep = int(np.argmin(va_loss))
            best_acc_ep  = int(np.argmax(va_acc))

            axL.scatter(epochs[best_loss_ep], va_loss[best_loss_ep], s=28, color=color, edgecolor="white", zorder=3)
            axR.scatter(epochs[best_acc_ep],  va_acc[best_acc_ep],  s=28, color=color, edgecolor="white", zorder=3)

        # Annotate last values
        if annotate_last and n > 0:
            # small horizontal offset for readability
            x_last = epochs[-1]
            axL.text(x_last + 0.2, va_loss[-1], f"{name} Val {va_loss[-1]:.3f}", color=color, fontsize=8, va="center")
            axR.text(x_last + 0.2, va_acc[-1],  f"{name} Val {va_acc[-1]:.1f}%", color=color, fontsize=8, va="center")

    # Autoscale y with margin
    if loss_vals:
        y = np.array(loss_vals, dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            ymin, ymax = float(np.min(y)), float(np.max(y))
            if ymin == ymax:
                ymin -= 0.1
                ymax += 0.1
            margin = 0.05 * (ymax - ymin)
            axL.set_ylim(ymin - margin, ymax + margin)

    # Legends: one for models (colors), one for styles (train vs val)
    leg1 = axL.legend(handles=model_handles, loc="upper right", fontsize=8, title="Models")
    leg2 = axR.legend(handles=style_handles, loc="lower right", fontsize=8, title="Curves")

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)

    plt.show()

@torch.no_grad()
def plot_example_mae(model, image, device="cpu", suptitle=None):
    """
    Plots original, masked input, full reconstruction, and masked-only reconstruction
    for a single example. Also returns these images and the mask.

    Args:
        model: TinyMAE instance (already on `device`).
        image: Tensor with shape (16,16), (1,16,16), or (1,1,16,16).
               Values can be in [0,1] or [0,255].
        device: Torch device for inference.
        suptitle: Optional figure title.

    Returns:
        out (dict of torch.Tensor on CPU, shape (1,1,16,16) unless noted):
            {
              "orig":               (1,1,16,16) in [0,1],
              "masked_input":       (1,1,16,16) in [0,1],
              "recon_full":         (1,1,16,16) in [0,1],
              "recon_masked_only":  (1,1,16,16) in [0,1],
              "mask":               (1, N_patches) boolean
            }
    """
    import torch
    import matplotlib.pyplot as plt
    # avoid circular import
    from solutions import patchify, unpatchify

    model.eval()

    # ---- prepare input shape (B=1, 1, 16, 16) and range ----
    if isinstance(image, torch.Tensor):
        x = image.clone()
    else:
        x = torch.tensor(image, dtype=torch.float32)

    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)         # (1,1,16,16)
    elif x.dim() == 3:
        if x.shape[0] == 1:                     # (1,16,16) -> (1,1,16,16)
            x = x.unsqueeze(0)
        else:
            raise ValueError("3D input must be (1,16,16).")
    elif x.dim() == 4:
        if x.shape[0] != 1 or x.shape[1] != 1:
            raise ValueError("4D input must be (1,1,16,16).")
    else:
        raise ValueError("Input must have 2, 3, or 4 dims.")

    x = x.to(device=device, dtype=torch.float32)
    if x.max() > 1.0:
        x = x / 255.0
    x = x.clamp(0.0, 1.0)

    # ---- forward pass to get reconstruction and mask ----
    # recon: (1, N, P), mask: (1, N) with True meaning "masked"
    recon, mask = model(x)

    # ---- build masked image by zeroing masked patches ----
    patches = patchify(x, model.patch_size)     # (1, N, P)
    patches_masked = patches.clone()
    patches_masked[0, mask[0]] = 0.0            # zero masked patches
    x_masked = unpatchify(patches_masked, model.patch_size)  # (1,1,16,16)

    # ---- full reconstruction from model predictions ----
    # If your model.reconstruct_image(unpatches) expects (1,N,P), keep using it:
    x_recon_full = model.reconstruct_image(recon)            # (1,1,16,16)

    # ---- masked-only reconstruction:
    # use predictions at masked positions; keep original patches at unmasked
    patches_mo = patches.clone()                              # start from originals
    patches_mo[0, mask[0]] = recon[0, mask[0]]               # overwrite masked with preds
    x_recon_masked_only = unpatchify(patches_mo, model.patch_size)  # (1,1,16,16)

    # ---- to CPU numpy for plotting ----
    orig_np = x[0, 0].detach().cpu().numpy()
    masked_np = x_masked[0, 0].detach().cpu().numpy()
    recon_full_np = x_recon_full[0, 0].detach().cpu().numpy()
    recon_mo_np = x_recon_masked_only[0, 0].detach().cpu().numpy()

    # ---- plot ----
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), dpi=150)
    if suptitle is not None:
        fig.suptitle(suptitle)

    axs[0].imshow(orig_np, cmap="gray", vmin=0.0, vmax=1.0)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(masked_np, cmap="gray", vmin=0.0, vmax=1.0)
    axs[1].set_title("Masked input")
    axs[1].axis("off")

    axs[2].imshow(recon_full_np, cmap="gray", vmin=0.0, vmax=1.0)
    axs[2].set_title("Full reconstruction")
    axs[2].axis("off")

    axs[3].imshow(recon_mo_np, cmap="gray", vmin=0.0, vmax=1.0)
    axs[3].set_title("Masked-only reconstruction")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "orig": x.detach().cpu(),
        "masked_input": x_masked.detach().cpu(),
        "recon_full": x_recon_full.detach().cpu(),
        "recon_masked_only": x_recon_masked_only.detach().cpu(),
        "mask": mask.detach().cpu().to(torch.bool),
    }
