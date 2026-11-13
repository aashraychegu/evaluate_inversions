from row_plotting_helpers import plot_field, plot_log_field, Plot
from pathlib import Path
import xarray as xr
import rioxarray as rxr
from argparse import ArgumentParser
import tomllib
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--folder", required = True)
parser.add_argument("--config-name", default = "diff.toml")
parser.add_argument("--output-dir", default = "/oak/stanford/groups/cyaolai/AashrayChegu/DIFFICE_DIFFS")
args = parser.parse_args()

folder = Path(args.folder)
config_path = folder / args.config_name
output_dir = Path(args.output_dir) / folder.name
output_dir.mkdir(exist_ok = True)

config = tomllib.loads(config_path.read_text())
comparisons = config["comparisons"]


plots = {
    # linear scale
    "mu":           Plot(function=plot_field,        field="mu",      cbmin = None, cbmax = None, label="Linear Viscosity",                latex=r"$\mu$"       ),
    "e_eff":        Plot(function=plot_field,        field="e_eff",   cbmin = None, cbmax = None, label="Linear Effective Strain Rate",    latex=r"$e_{eff}$"   ),
    "tau_eff":      Plot(function=plot_field,        field="tau_eff", cbmin = None, cbmax = None, label="Linear Effective Stress",         latex=r"$\tau_{eff}$"),
    # log scale 
    "log_mu":       Plot(function=plot_log_field,    field="mu",      cbmin = None, cbmax = None, label="Log Viscosity",            latex=r"$\mu$"        ),
    "log_e_eff":    Plot(function=plot_log_field,    field="e_eff",   cbmin = None, cbmax = None, label="Log Effective Strain Rate",latex=r"$e_{eff}$"    ),
    "log_tau_eff":  Plot(function=plot_log_field,    field="tau_eff", cbmin = None, cbmax = None, label="Log Effective Stress",     latex=r"$\tau_{eff}$" ),
}

for label, zarrs in tqdm(comparisons.items(), smoothing=0, position=1,):

    first_path = folder / zarrs["minuend"] / "evaluations.zarr"
    second_path = folder / zarrs["subtrahend"] / "evaluations.zarr"

    first_zarr = xr.open_zarr(first_path, consolidated=False)
    second_zarr = xr.open_zarr(second_path, consolidated=False)

    first_aligned, second_aligned = xr.align(first_zarr, second_zarr, join="inner")

    n_rows, n_cols = len(plots), 2

    fig, ax = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        squeeze=False,           # always 2D, even for 1 row / 1 column
        layout="constrained",    # keeps the per-row colorbars tidy
    )

    for row,(name, plot) in tqdm(enumerate(plots.items())):
        plot({zarrs["minuend"]: first_zarr, zarrs["subtrahend"]: second_zarr}, ax[row])

    parent_dir = output_dir / label
    save_path =  parent_dir / "Side By Side Fields.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
