"""Batch experiment runner for the two-type household extension.

Experiments:
A: Varying lambda_p (fraction of power-seekers)
B: Varying power_target (imitation target)
C: Baseline comparison (lambda_p=0, regression test)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from model.model_runner import runner

# fraction of trajectory to skip when computing summary statistics
LATE_FRACTION = 0.9


def make_args(**overrides):
    """Create a Namespace with default arguments, applying overrides."""
    defaults = dict(
        top="FC", nagents=500, k=20, p=0.01,
        tau=50.0, d=0.1, alpha=None, phi=None,
        K=1.02e06, delta_s=None, w_future=None,
        pfixed=None, rfixed=None, sfixed=None, pexplore=None,
        tmax=500, saveloc="experiments/", micro=False, dontsave=True,
        logiter=100000, seed=0, movie=None,
        lambda_p=0.0, power_target="power_mult", d_power=0.5,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def run_experiment_A(outdir="experiments/A", n_seeds=5, tau=50.0, tmax=500, nagents=500):
    """Experiment A: Varying lambda_p.

    Sweep lambda_p over [0.0, 0.01, 0.05, 0.1, 0.2, 0.5].
    """
    os.makedirs(outdir, exist_ok=True)
    lambda_ps = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    results = []

    for lp in lambda_ps:
        for s in range(1, n_seeds + 1):
            print(f"\n=== Experiment A: lambda_p={lp}, seed={s} ===")
            args = make_args(
                tau=tau, tmax=tmax, nagents=nagents, seed=s,
                lambda_p=lp, power_target="power_mult", d_power=0.5,
            )
            final, traj = runner(args)[:2]
            late = traj.iloc[int(len(traj) * LATE_FRACTION):]
            results.append({
                "lambda_p": lp,
                "seed": s,
                "wealth_share_p": late["wealth_share_p"].mean(),
                "Y_mean": late["Y"].mean(),
                "K_mean": late["capital"].mean(),
                "s_mean_h": late["s_mean_h"].mean(),
                "s_mean_p": late["s_mean_p"].mean(),
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(outdir, "experiment_A.csv"), index=False)

    # plot wealth share and output vs lambda_p
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    fig.suptitle("Experiment A: Varying λ_p")
    for lp in lambda_ps:
        sub = df[df["lambda_p"] == lp]
        ax1.scatter([lp] * len(sub), sub["wealth_share_p"], c="blue", alpha=0.5, s=30)
        ax2.scatter([lp] * len(sub), sub["Y_mean"], c="red", alpha=0.5, s=30)
    means = df.groupby("lambda_p").mean()
    ax1.plot(means.index, means["wealth_share_p"], "b-o")
    ax1.set_xlabel("λ_p")
    ax1.set_ylabel("Wealth share of power-seekers")
    ax1.set_title("Wealth share vs λ_p")
    ax2.plot(means.index, means["Y_mean"], "r-o")
    ax2.set_xlabel("λ_p")
    ax2.set_ylabel("Aggregate output Y")
    ax2.set_title("Output vs λ_p")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "experiment_A.png"), dpi=150)
    plt.close(fig)
    print(f"\nExperiment A results saved to {outdir}")
    return df


def run_experiment_B(outdir="experiments/B", n_seeds=5, tau=50.0, tmax=500, nagents=500):
    """Experiment B: Varying power_target.

    Fix lambda_p=0.1, sweep over 'capital', 'power_mult', 'power_log'.
    """
    os.makedirs(outdir, exist_ok=True)
    targets = ["capital", "power_mult", "power_log"]
    results = []

    for pt in targets:
        for s in range(1, n_seeds + 1):
            print(f"\n=== Experiment B: power_target={pt}, seed={s} ===")
            args = make_args(
                tau=tau, tmax=tmax, nagents=nagents, seed=s,
                lambda_p=0.1, power_target=pt, d_power=0.5,
            )
            final, traj = runner(args)[:2]
            late = traj.iloc[int(len(traj) * LATE_FRACTION):]
            results.append({
                "power_target": pt,
                "seed": s,
                "wealth_share_p": late["wealth_share_p"].mean(),
                "s_mean_p": late["s_mean_p"].mean(),
                "K_mean_p": late["K_mean_p"].mean(),
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(outdir, "experiment_B.csv"), index=False)
    print(f"\nExperiment B results saved to {outdir}")
    return df


def run_experiment_C(outdir="experiments/C", tau=50.0, tmax=500, nagents=500, seed=1):
    """Experiment C: Baseline comparison.

    Run lambda_p=0.0 (pure hedonist) as regression test.
    """
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== Experiment C: baseline (lambda_p=0.0), seed={seed} ===")
    args = make_args(
        tau=tau, tmax=tmax, nagents=nagents, seed=seed,
        lambda_p=0.0,
    )
    final, traj = runner(args)[:2]
    traj.to_pickle(os.path.join(outdir, "baseline_traj.pkl"))
    print(f"\nExperiment C baseline saved to {outdir}")
    return traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run two-type household experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["A", "B", "C", "all"],
        help="Which experiment to run (default: all)",
    )
    parser.add_argument("--outdir", type=str, default="experiments/", help="Output directory")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds per config")
    parser.add_argument("--tau", type=float, default=50.0, help="Mean interaction time")
    parser.add_argument("--tmax", type=int, default=500, help="Simulation length in tau units")
    parser.add_argument("--nagents", type=int, default=500, help="Number of agents")
    args = parser.parse_args()

    if args.experiment in ("A", "all"):
        run_experiment_A(
            outdir=os.path.join(args.outdir, "A"),
            n_seeds=args.n_seeds, tau=args.tau, tmax=args.tmax, nagents=args.nagents,
        )
    if args.experiment in ("B", "all"):
        run_experiment_B(
            outdir=os.path.join(args.outdir, "B"),
            n_seeds=args.n_seeds, tau=args.tau, tmax=args.tmax, nagents=args.nagents,
        )
    if args.experiment in ("C", "all"):
        run_experiment_C(
            outdir=os.path.join(args.outdir, "C"),
            tau=args.tau, tmax=args.tmax, nagents=args.nagents,
        )
