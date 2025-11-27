import numpy as np
import pandas as pd

def simulate_pendulum_dataset(
    num_samples=500,
    t_max=10.0,
    length=1.0,
    gravity=9.81,
    damping=0.02,
    theta0=np.deg2rad(20.0),
    omega0=0.0,
    noise_std=0.0,
    seed=None,
):
    """
    Simulate a physical pendulum and return a pandas DataFrame suitable for regression.

    State equation for a damped simple pendulum:
        d theta / dt = omega
        d omega / dt = - (g / L) * sin(theta) - c * omega

    Args:
        num_samples: Number of time samples to generate over [0, t_max].
        t_max: Total simulated time in seconds.
        length: Pendulum length in meters.
        gravity: Gravitational acceleration in m/s^2.
        damping: Linear damping coefficient c in 1/s.
        theta0: Initial angular displacement in radians.
        omega0: Initial angular velocity in radians/s.
        noise_std: Standard deviation of additive Gaussian noise for measured x and y.
        seed: Optional random seed for reproducible noise.

    Returns:
        df: DataFrame with columns
            t           time stamp
            theta       angular displacement in radians
            omega       angular velocity in radians/s
            x           bob x position in meters
            y           bob y position in meters
            x_obs       noisy x measurement
            y_obs       noisy y measurement
    """
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2 to define a time grid.")

    rng = np.random.default_rng(seed)
    dt = t_max / (num_samples - 1)

    def f(state):
        th, om = state
        dth = om
        dom = - (gravity / length) * np.sin(th) - damping * om
        return np.array([dth, dom], dtype=np.float64)

    def rk4_step(state, h):
        k1 = f(state)
        k2 = f(state + 0.5 * h * k1)
        k3 = f(state + 0.5 * h * k2)
        k4 = f(state + h * k3)
        return state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    t = np.linspace(0.0, t_max, num_samples, dtype=np.float64)
    theta = np.empty_like(t)
    omega = np.empty_like(t)

    state = np.array([theta0, omega0], dtype=np.float64)
    theta[0], omega[0] = state

    for i in range(1, num_samples):
        state = rk4_step(state, dt)
        theta[i], omega[i] = state

    # Cartesian position of bob relative to pivot
    x = length * np.sin(theta)
    y = -length * np.cos(theta)

    if noise_std > 0.0:
        x_obs = x + rng.normal(0.0, noise_std, size=x.shape)
        y_obs = y + rng.normal(0.0, noise_std, size=y.shape)
        theta_obs = np.arctan2(x_obs, -y_obs)
    else:
        x_obs = x.copy()
        y_obs = y.copy()
        theta_obs = theta.copy()

    df = pd.DataFrame(
        {
            "t": t,
            "theta": theta,
            "theta_obs": theta_obs,
            "omega": omega,
            "x": x,
            "y": y,
            "x_obs": x_obs,
            "y_obs": y_obs,
        }
    )

    return df