import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def plot_results_perceptron(weights, miss_rate, labels, pred):
    """Plot the results of the perceptron"""

    cm = confusion_matrix(labels, pred)
    acc = accuracy_score(labels, pred)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3), dpi=200)
    ax[0].imshow(weights.reshape((16, 16)))
    ax[0].axis('off')
    ax[0].set_title("Learned weights")

    ax[1].plot(miss_rate)
    ax[1].set_title("Misclassifications")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Misclassification rate")

    df_cm = pd.DataFrame(cm, index = [i for i in ["Circle", "Square"]],
                  columns = [i for i in ["Circle", "Square"]])
    sns.heatmap(df_cm, annot=True, fmt='g', ax=ax[2])
    ax[2].set_title("Confusion matrix, accuracy: {:.2f}".format(acc))
    plt.tight_layout()
    plt.show()

def _to_numpy(W):
    if isinstance(W, np.ndarray):
        return W
    try:
        return W.cpu().detach().numpy()
    except AttributeError:
        return np.asarray(W)

def plot_results_mlp_cnn(model):
    """
    Visualize loss, accuracy and all FC layer weights for a dynamic MLP.
    For each FC layer with weight matrix W of shape (in_dim, out_dim),
    a heatmap of W is shown. If the first FC layer has a square in_dim,
    a small montage of its first few neuron kernels is shown.
    """
    # ----- metrics row -----
    epochs_nr = np.arange(1, len(model.loss_train_plot) + 1)
    # collect FC weights in order
    fc_weights = []
    for layer in getattr(model, "layers", []):
        if isinstance(layer, dict) and layer.get("type") == "FC":
            W = _to_numpy(layer["W"])
            fc_weights.append(W)

    # collect Conv weights in order if model has conv_layers
    conv_weights = []
    for layer in getattr(model, "conv_layers", []):
        if isinstance(layer, dict) and layer.get("type") == "conv":
            W = _to_numpy(layer["W"])        # shape (C_out, C_in, k, k)
            conv_weights.append(W)

    # decide layout rows: 1 row for metrics, optionally 1 for montage, plus 1 per FC heatmap
    rows = 1  # metrics
    rows += len(conv_weights)  # one row per conv montage
    show_montage = False
    montage_cfg = None
    if len(fc_weights) > 0:
        in_dim0, out_dim0 = fc_weights[0].shape
        s = int(math.sqrt(in_dim0))
        if s * s == in_dim0:
            show_montage = True
            rows += 1
            n_show = min(out_dim0, 10)
            n_cols = min(5, n_show)
            n_rows = int(math.ceil(n_show / n_cols))
            montage_cfg = (s, n_show, n_rows, n_cols)

    # rows += len(fc_weights)  # one row per FC heatmap

    fig = plt.figure(layout="constrained", figsize=(15, 15), dpi=150)
    subfigs = fig.subfigures(rows, 1, wspace=0.07)

    # Row 0: loss and accuracy
    axs0 = subfigs[0].subplots(1, 2)

    axs0[0].set_xlabel("Epochs")
    axs0[0].set_ylabel("Loss")
    axs0[0].plot(epochs_nr, model.loss_train_plot)
    axs0[0].plot(epochs_nr, model.loss_test_plot, linestyle="dashed")
    axs0[0].grid(which="both")
    axs0[0].legend(["Train", "Test"], loc="upper right")
    axs0[0].set_title("Loss")

    axs0[1].set_xlabel("Epochs")
    axs0[1].set_ylabel("Accuracy")
    axs0[1].plot(epochs_nr, np.array(model.acc_train_plot) * 100.0)
    axs0[1].plot(epochs_nr, np.array(model.acc_test_plot) * 100.0, linestyle="dashed")
    axs0[1].grid(which="both")
    axs0[1].set_ylim([0, 101])
    axs0[1].legend(["Train", "Test"], loc="lower right")
    axs0[1].set_title("Accuracy")

    next_row = 1

    for ci, W in enumerate(conv_weights, start=1):
        # W: (C_out, C_in, k, k)
        C_out, C_in, k, _ = W.shape
        n_show = min(C_out, 16)
        n_cols = min(8, n_show)
        n_rows = int(math.ceil(n_show / n_cols))
        axs_c = subfigs[next_row].subplots(n_rows, n_cols)
        axs_c = np.atleast_1d(axs_c).reshape(n_rows, n_cols)
        subfigs[next_row].suptitle(f"Conv {ci} kernels, k={k}, Cout x Cin = {C_out} x {C_in}")

        for j in range(n_rows * n_cols):
            ax = axs_c[j // n_cols, j % n_cols]
            if j < n_show:
                # average across input channels for a single 2D view
                kernel2d = W[j].mean(axis=0) if C_in > 1 else W[j, 0]
                ax.imshow(kernel2d)
            ax.axis("off")
        next_row += 1

    # Optional montage of first FC neuron kernels if input is square
    if show_montage and montage_cfg is not None:
        s, n_show, n_rows, n_cols = montage_cfg
        axs_m = subfigs[next_row].subplots(n_rows, n_cols)
        axs_m = np.atleast_1d(axs_m).reshape(n_rows, n_cols)
        subfigs[next_row].suptitle("First FC neuron kernels")
        W0 = fc_weights[0]  # shape (in_dim, out_dim)
        for j in range(n_rows * n_cols):
            ax = axs_m[j // n_cols, j % n_cols]
            if j < n_show:
                kernel = W0[:, j].reshape(s, s)
                ax.imshow(kernel)
            ax.axis("off")
        next_row += 1

    plt.show()