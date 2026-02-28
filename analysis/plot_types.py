"""Plotting routines for the two-type household extension.

Produces:
(a) Savings rate heatmap by type (side-by-side panels)
(b) Time series panel (6-panel figure)
(c) Kuznets-style plot (wealth share vs aggregate output)
"""

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_timeseries_by_type(trajectory, save_path=None, tmin=None):
    """Six-panel time series figure showing per-type dynamics.

    Panels:
    1. Mean savings rate by type
    2. Mean capital by type
    3. Mean consumption by type
    4. Power-seekers' wealth share
    5. Interest rate r
    6. Mean power metric W by type
    """
    idx = trajectory.index.values
    if tmin is not None:
        mask = idx >= tmin
        trajectory = trajectory.iloc[mask]
        idx = trajectory.index.values

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    fig.suptitle("Two-type household dynamics", fontsize=14)

    # 1. Mean savings rate by type
    ax = axes[0, 0]
    ax.plot(idx, trajectory["s_mean_h"], label="hedonists", alpha=0.8)
    ax.plot(idx, trajectory["s_mean_p"], label="power-seekers", alpha=0.8)
    ax.set_ylabel("Mean savings rate")
    ax.set_xlabel("Time")
    ax.set_title("Mean savings rate by type")
    ax.legend()

    # 2. Mean capital by type
    ax = axes[0, 1]
    ax.plot(idx, trajectory["K_mean_h"], label="hedonists", alpha=0.8)
    ax.plot(idx, trajectory["K_mean_p"], label="power-seekers", alpha=0.8)
    ax.set_ylabel("Mean capital")
    ax.set_xlabel("Time")
    ax.set_title("Mean capital by type")
    ax.legend()

    # 3. Mean consumption by type
    ax = axes[0, 2]
    ax.plot(idx, trajectory["C_mean_h"], label="hedonists", alpha=0.8)
    ax.plot(idx, trajectory["C_mean_p"], label="power-seekers", alpha=0.8)
    ax.set_ylabel("Mean consumption")
    ax.set_xlabel("Time")
    ax.set_title("Mean consumption by type")
    ax.legend()

    # 4. Wealth share of power-seekers
    ax = axes[1, 0]
    ax.plot(idx, trajectory["wealth_share_p"], color="purple", alpha=0.8)
    ax.set_ylabel("Wealth share")
    ax.set_xlabel("Time")
    ax.set_title("Power-seekers' wealth share")

    # 5. Interest rate r
    ax = axes[1, 1]
    ax.plot(idx, trajectory["r"], color="darkgreen", alpha=0.8)
    ax.set_ylabel("Interest rate r")
    ax.set_xlabel("Time")
    ax.set_title("Interest rate")

    # 6. Mean power metric W by type
    ax = axes[1, 2]
    ax.plot(idx, trajectory["W_mean_h"], label="hedonists", alpha=0.8)
    ax.plot(idx, trajectory["W_mean_p"], label="power-seekers", alpha=0.8)
    ax.set_ylabel("Mean W = K·C^d")
    ax.set_xlabel("Time")
    ax.set_title("Mean power metric by type")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved time series plot to {save_path}")
    plt.close(fig)
    return fig


def plot_savings_heatmap_by_type(micro_trajectory, save_path=None, bins=50):
    """Savings rate heatmap with two side-by-side panels: hedonists vs power-seekers."""
    fig, (ax_h, ax_p) = plt.subplots(ncols=2, figsize=(14, 6), sharey=True)
    fig.suptitle("Savings rate distribution by type", fontsize=14)

    times = micro_trajectory.index.values
    s_bins = np.linspace(0, 1, bins + 1)

    for ax, type_label, title in [
        (ax_h, "hedonist", "Hedonists"),
        (ax_p, "power_seeker", "Power-seekers"),
    ]:
        heatmap = np.zeros((len(times), bins))
        for i, t in enumerate(times):
            row = micro_trajectory.iloc[i]
            s = row["indiv_savings_rate"]
            ht = row["household_type"]
            mask = ht == type_label
            if mask.any():
                counts, _ = np.histogram(s[mask], bins=s_bins)
                heatmap[i, :] = counts

        if heatmap.max() > 0:
            ax.imshow(
                heatmap.T,
                aspect="auto",
                origin="lower",
                extent=[times[0], times[-1], 0, 1],
                cmap="hot",
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Savings rate")
        ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved savings heatmap to {save_path}")
    plt.close(fig)
    return fig


def plot_kuznets(trajectory, save_path=None):
    """Scatter of wealth share of power-seekers vs aggregate output Y."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        trajectory["Y"],
        trajectory["wealth_share_p"],
        alpha=0.3,
        s=5,
        c=trajectory.index.values,
        cmap="viridis",
    )
    ax.set_xlabel("Aggregate output Y")
    ax.set_ylabel("Wealth share of power-seekers")
    ax.set_title("Kuznets-style: wealth share vs output")
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label("Time")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved Kuznets plot to {save_path}")
    plt.close(fig)
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot two-type household results")
    parser.add_argument("--traj", type=str, required=True, help="Path to macro trajectory pickle")
    parser.add_argument("--micro", type=str, default=None, help="Path to micro trajectory pickle")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for plots")
    parser.add_argument("--tmin", type=float, default=None, help="Min time to plot")
    args = parser.parse_args()

    traj = pd.read_pickle(args.traj)
    plot_timeseries_by_type(traj, save_path=f"{args.outdir}/timeseries_by_type.png", tmin=args.tmin)
    plot_kuznets(traj, save_path=f"{args.outdir}/kuznets.png")

    if args.micro:
        micro = pd.read_pickle(args.micro)
        plot_savings_heatmap_by_type(micro, save_path=f"{args.outdir}/savings_heatmap_by_type.png")
