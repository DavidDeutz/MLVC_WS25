import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pendulum_dataset(df, length=1.0, show_phase=True):
    """
    Plot time series and trajectory from a pendulum dataset created by simulate_pendulum_dataset.

    Args:
        df: DataFrame with columns t, theta, omega, x, y, x_obs, y_obs.
        length: Pendulum length in meters, used for a reference circle in the trajectory plot.
        show_phase: If True, also plot the phase portrait theta vs omega.
    """
    required = {"t", "theta", "omega", "x", "y", "x_obs", "y_obs"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    t = df["t"].to_numpy()
    theta = df["theta"].to_numpy()
    omega = df["omega"].to_numpy()
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    x_obs = df["x_obs"].to_numpy()
    y_obs = df["y_obs"].to_numpy()

    ncols = 3 if show_phase else 2
    fig, axs = plt.subplots(1, ncols, figsize=(14, 4), dpi=150)

    # Angle over time
    ax0 = axs[0]
    ax0.plot(t, theta, label="theta")
    ax0.set_xlabel("time [s]")
    ax0.set_ylabel("angle [rad]")
    ax0.set_title("Angle over time")
    ax0.grid(True, which="both")

    # Trajectory
    ax1 = axs[1]
    ax1.plot(x, y, label="true path")
    ax1.scatter(x_obs[::max(1, len(x_obs)//100)], y_obs[::max(1, len(y_obs)//100)], s=10, alpha=0.6, label="observations")
    # Reference circle for length
    circle = plt.Circle((0.0, 0.0), radius=length, fill=False, linestyle="dashed")
    ax1.add_patch(circle)
    ax1.set_aspect("equal", "box")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Trajectory in the plane")
    ax1.grid(True, which="both")
    ax1.legend(loc="best")

    if show_phase:
        ax2 = axs[2]
        ax2.plot(theta, omega)
        ax2.set_xlabel("theta [rad]")
        ax2.set_ylabel("omega [rad/s]")
        ax2.set_title("Phase portrait")
        ax2.grid(True, which="both")

    plt.tight_layout()
    plt.show()

def plot_regression_over_time(
    df,
    idx_train,
    X_val,
    y_val_pred,
    y_val_std=None,
    time_col="t",
    true_theta_col="theta",
    obs_theta_col="theta_obs",
    obs_x_col="x_obs",
    obs_y_col="y_obs",
    title="Regression on pendulum data",
):
    """
    Plots:
      - true theta(t) over the full time axis
      - noisy observed training samples (only the training part)
      - model prediction on a dense time grid X_val
      - optional ±2 std band if y_val_std is provided

    Args:
        df: DataFrame with simulated data.
        idx_train: 1D array of training indices (first part).
        X_val: array of shape (M, 1) containing time points for dense prediction grid.
        y_val_pred: array of shape (M,) with model predictions at X_val.
        y_val_std: optional array of shape (M,) with predictive std at X_val.
        time_col: column name for time in df.
        true_theta_col: column name for ground-truth theta in df.
        obs_theta_col: column name for observed theta; if missing, it will be computed from (x_obs, y_obs).
        obs_x_col, obs_y_col: names of noisy Cartesian observations used to back-compute theta if needed.
        title: figure title.
    """
    # Full ground-truth curve
    t_full = df[time_col].to_numpy()
    theta_true = df[true_theta_col].to_numpy()

    # Training observed-theta (prefer precomputed column; else derive from x_obs,y_obs)
    if obs_theta_col in df.columns:
        theta_obs = df.loc[idx_train, obs_theta_col].to_numpy()
    else:
        x_obs = df.loc[idx_train, obs_x_col].to_numpy()
        y_obs = df.loc[idx_train, obs_y_col].to_numpy()
        theta_obs = np.arctan2(x_obs, -y_obs)

    t_train = df.loc[idx_train, time_col].to_numpy()

    # Sort the dense grid for a clean prediction line
    X_val = np.asarray(X_val).reshape(-1, 1)
    order = np.argsort(X_val[:, 0])
    t_grid = X_val[:, 0][order]
    y_pred_sorted = np.asarray(y_val_pred)[order]
    y_std_sorted = None if y_val_std is None else np.asarray(y_val_std)[order]

    plt.figure(figsize=(10, 5), dpi=150)

    # True function over the whole time range
    plt.plot(t_full, theta_true, label="true θ(t)")

    # Noisy training samples
    plt.scatter(t_train, theta_obs, s=20, alpha=0.8, label="train points (obs)")

    # Prediction on dense grid
    plt.plot(t_grid, y_pred_sorted, linestyle="--", label="prediction")

    # Optional uncertainty band
    if y_std_sorted is not None:
        plt.fill_between(
            t_grid,
            y_pred_sorted - y_std_sorted,
            y_pred_sorted + y_std_sorted,
            alpha=0.25,
            label="±2 std",
        )

    plt.xlabel("time [s]")
    plt.ylabel("angle [rad]")
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()