#!/usr/bin/env python3
"""Quick demo of the two-type household extension (hedonists vs power-seekers).

Runs three short simulations, prints a comparison table, and saves diagnostic
plots to the ``demo_output/`` directory.  Designed to finish in under a minute
on a laptop so you can quickly see the key dynamics.

Usage
-----
    python3 run_quick_demo.py                # uses defaults
    python3 run_quick_demo.py --nagents 200  # larger but slower
    python3 run_quick_demo.py --tmax 50      # longer time horizon

The demo compares:
  1. **Baseline** – all hedonists (λ_p = 0)
  2. **Power-seekers (power_mult)** – 10% power-seekers using W = K·C^d
  3. **Power-seekers (capital)** – 10% power-seekers copying by capital
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from model.model_runner import runner
from analysis.plot_types import plot_timeseries_by_type, plot_kuznets


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_args(**overrides):
    """Return an argparse.Namespace with sensible fast-demo defaults."""
    defaults = dict(
        top="FC", nagents=100, k=20, p=0.01,
        tau=50.0, d=0.1, alpha=None, phi=None,
        K=1.02e06, delta_s=None, w_future=None,
        pfixed=None, rfixed=None, sfixed=None, pexplore=None,
        tmax=20, saveloc="demo_output/", micro=False, dontsave=True,
        logiter=100000, seed=42, movie=None,
        lambda_p=0.0, power_target="power_mult", d_power=0.5,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def summary_row(label, traj):
    """Extract late-time summary statistics from a macro trajectory."""
    late = traj.iloc[int(len(traj) * 0.8):]   # last 20 %
    return {
        "scenario": label,
        "s_mean_h": f"{late['s_mean_h'].mean():.4f}",
        "s_mean_p": f"{late['s_mean_p'].mean():.4f}",
        "K_mean_h": f"{late['K_mean_h'].mean():.0f}",
        "K_mean_p": f"{late['K_mean_p'].mean():.0f}",
        "C_mean_h": f"{late['C_mean_h'].mean():.0f}",
        "C_mean_p": f"{late['C_mean_p'].mean():.0f}",
        "wealth_share_p": f"{late['wealth_share_p'].mean():.4f}",
        "r_mean": f"{late['r'].mean():.4f}",
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_demo(nagents=100, tmax=20, tau=50.0, seed=42, outdir="demo_output"):
    os.makedirs(outdir, exist_ok=True)

    scenarios = [
        ("baseline (all hedonists)", dict(lambda_p=0.0)),
        ("10% power-seekers (power_mult)", dict(lambda_p=0.1, power_target="power_mult")),
        ("10% power-seekers (capital)", dict(lambda_p=0.1, power_target="capital")),
    ]

    results = []
    trajectories = {}

    for label, params in scenarios:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        args = make_args(nagents=nagents, tmax=tmax, tau=tau, seed=seed, **params)
        final, traj = runner(args)[:2]
        trajectories[label] = traj
        results.append(summary_row(label, traj))

    # ---- console summary --------------------------------------------------
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("  Summary (late-time averages)")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print()

    # ---- plots ------------------------------------------------------------
    for label, traj in trajectories.items():
        tag = label.split("(")[-1].rstrip(")").replace(" ", "_")
        plot_timeseries_by_type(
            traj,
            save_path=os.path.join(outdir, f"timeseries_{tag}.png"),
        )
        plot_kuznets(
            traj,
            save_path=os.path.join(outdir, f"kuznets_{tag}.png"),
        )

    # ---- combined comparison plot -----------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    fig.suptitle("Quick demo – scenario comparison", fontsize=14)

    colors = ["tab:blue", "tab:orange", "tab:red"]
    for (label, traj), color in zip(trajectories.items(), colors):
        idx = traj.index.values
        axes[0].plot(idx, traj["wealth_share_p"], label=label, alpha=0.8, color=color)
        axes[1].plot(idx, traj["s_mean_h"], "--", alpha=0.6, color=color,
                     label=f"hedonists – {label}")
        axes[1].plot(idx, traj["s_mean_p"], "-", alpha=0.8, color=color,
                     label=f"power-seekers – {label}")
        axes[2].plot(idx, traj["r"], label=label, alpha=0.8, color=color)

    axes[0].set_title("Power-seekers' wealth share")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Wealth share")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Mean savings rate by type")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Savings rate")
    axes[1].legend(fontsize=7)

    axes[2].set_title("Interest rate")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("r")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    cmp_path = os.path.join(outdir, "comparison.png")
    fig.savefig(cmp_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {cmp_path}")

    print(f"\nAll plots saved to {outdir}/")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick demo comparing hedonist vs power-seeker households",
    )
    parser.add_argument("--nagents", type=int, default=100,
                        help="Number of agents (default: 100)")
    parser.add_argument("--tmax", type=int, default=20,
                        help="Simulation length in tau units (default: 20)")
    parser.add_argument("--tau", type=float, default=50.0,
                        help="Mean interaction time (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--outdir", type=str, default="demo_output",
                        help="Output directory for plots (default: demo_output)")
    args = parser.parse_args()

    run_demo(
        nagents=args.nagents,
        tmax=args.tmax,
        tau=args.tau,
        seed=args.seed,
        outdir=args.outdir,
    )
