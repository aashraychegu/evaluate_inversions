from pathlib import Path
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import tomllib
import shutil
from tqdm import tqdm
from functools import partial
from plotting_helpers import Plot, plot_field, plot_field, compare_fields, compare_speed, plot_side_by_side, plot_stress_to_strain_rate, histogram

plot_vars = [
    Plot(
        function=compare_fields, 
        field=("h", "regrid_h"), 
        label="Thickness Difference"
    ),
    Plot(
        function=compare_fields, 
        field=("u", "regrid_u"), 
        label="X Velocity Difference"
    ),
    Plot(
        function=compare_fields, 
        field=("v", "regrid_v"), 
        label="Y Velocity Difference"
    ),
    Plot(
        function=compare_speed,
        field=("u", "v", "regrid_u", "regrid_v"),
        label="Speed Difference",
    ),
    Plot(
        function=partial(plot_side_by_side, log_first = True, log_second = True, diff = True),
        field=("e_eff","mu"),
        label="Log Effective Strain Rate and Diff Log Viscosity",
    ),
    Plot(
        function=partial(plot_side_by_side, log_first = True, log_second = True, diff = True),
        field=("e_eff", "log(delta mu)"),
        label="Log Effective Strain Rate and Log Diff Viscosity",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="mu",
        label="Viscosity",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="mu",
        label="Diff Log Viscosity",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="log(delta mu)",
        label="Log Diff Viscosity",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="e_eff",
        label="Effective Strain Rate",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="e_eff",
        label="Log Effective Strain Rate",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="e_xx",
        label="Strain Rate XX Component",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="e_xy",
        label="Strain Rate XY Component",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="e_yy",
        label="Strain YY Component",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="tau_eff",
        label="Effective Stress",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="tau_xx",
        label="Stress XX Component",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="tau_xy",
        label="Stress XY Component",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="tau_yy",
        label="Stress YY Component",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="tau_1",
        label="Stress Primary Component",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="tau_2",
        label="Stress Secondary Component",
    ),
    Plot(
        function=partial(plot_field,diff = True),
        field="R_xx",
        label="Extensional Stress (R_xx)",
    ),
]

parser = ArgumentParser()
parser.add_argument("--folder")
args = parser.parse_args()

folder_path = Path(args.folder)

for idx, subfolder_path in (pbar1 := tqdm(list(enumerate(folder_path.glob("*/"))),smoothing = 0, position = 1)):
    pbar1.set_description(f"{idx}: {subfolder_path.name}")
    zarr_path = subfolder_path / "diff.zarr"
    zarr = xr.open_zarr(zarr_path,consolidated=False)
    image_path = subfolder_path / "images"
    image_path.mkdir(exist_ok = True)

    for plot_spec in (pbar2:= tqdm(plot_vars,smoothing = 0, position = 0)):
        pbar2.set_description(f"Plotting: {plot_spec.label} ")
        save_path, spec_str = plot_spec(zarr, subfolder_path.name, image_path)
        # tqdm.write(f"Saved {spec_str} to {save_path}")
