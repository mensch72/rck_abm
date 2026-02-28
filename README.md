# Ramsey-Cass-Koopman Agent-Based Model.

As described in our paper [_Emergent inequality and endogenous dynamics in a simple behavioral macroeconomic model_](https://arxiv.org/abs/1907.02155) by Yuki M. Asano, Jakob J. Kolb, Jobst Heitzig, J. Doyne Farmer.


## Installation
Install environment:
`conda env create -f environment.yml`

or if you're on a Mac:
`conda env create -f environment_mac.yml`

Alternatively, install dependencies directly via pip:
```
pip install numpy scipy networkx pandas matplotlib tqdm
```

Activate environment:
`conda activate rck_abm`


## Quick demo

The fastest way to see the model in action — including the two-type household
extension — is:

```bash
python3 run_quick_demo.py
```

This runs three short simulations (~8 seconds total), prints a comparison table
to the console, and saves diagnostic plots to `demo_output/`:

| File | Description |
|------|-------------|
| `comparison.png` | Side-by-side comparison of all three scenarios |
| `timeseries_*.png` | Six-panel per-type dynamics for each scenario |
| `kuznets_*.png` | Wealth share vs aggregate output scatter |

You can tune the demo size:
```bash
python3 run_quick_demo.py --nagents 200 --tmax 50   # larger / longer
python3 run_quick_demo.py --help                     # all options
```


## Running the model

Simply run as `python3 main.py --nagents 500 --tau 500`

### Two-type household extension

The model supports two household types — **hedonists** (copy the neighbour with
highest consumption, the original rule) and **power-seekers** (copy the neighbour
that maximises a capital/power metric). Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda-p` | `0.0` | Fraction of power-seeker households |
| `--power-target` | `power_mult` | Imitation target: `capital`, `power_mult`, or `power_log` |
| `--d-power` | `0.5` | Exponent on consumption in the power metric W = K · C^d |

Examples:
```bash
# Pure hedonist baseline (original model)
python3 main.py --nagents 500 --tau 50 --tmax 500 --d 0.1

# 10% power-seekers using the multiplicative power metric
python3 main.py --nagents 500 --tau 50 --tmax 500 --d 0.1 \
    --lambda-p 0.1 --power-target power_mult --d-power 0.5

# 10% power-seekers copying by capital
python3 main.py --nagents 500 --tau 50 --tmax 500 --d 0.1 \
    --lambda-p 0.1 --power-target capital
```

### Full usage

```
python3 main.py --help
```

## Evaluating results

Show basic trajectory:
```
python3 analysis/show_trajectory.py --path test_output/_TFC_N500_d10_tau500.0_tmax20_al66_pf0_rf0--traj.pkl
```

Show periodograms and trajectory:
```
python3 analysis/periodogram.py test_output/_TFC_N500_d10_tau500.0_tmax20_al66_pf0_rf0--traj.pkl
```

Plot per-type diagnostics from a saved trajectory:
```
python3 analysis/plot_types.py --traj path/to/traj.pkl --outdir plots/
```

## Batch experiments

`run_experiments.py` sweeps over the two-type extension parameters:

```bash
# Run all three experiments (A: vary λ_p, B: vary power_target, C: baseline)
python3 run_experiments.py --experiment all --nagents 500 --tmax 500

# Run only experiment A with fewer seeds for a quick check
python3 run_experiments.py --experiment A --n-seeds 2 --nagents 100 --tmax 20
```

## Running experiments on a cluster
Further install MPI4py: `pip install mpi4py`
