from pathlib import Path
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import tomllib
import shutil
from tqdm import tqdm
from functools import partial
from plotting_helpers import Plot, plot_field, plot_log_field, plot_field_notext, compare_fields, compare_speed, plot_side_by_side, plot_stress_to_strain_rate, histogram

plot_vars = [
    Plot(
        function=compare_fields,
        field=("h", "regrid_h"),
        label="Thickness Difference",
        latex=("$h$", r"$h_{regrid}$"),
    ),
    Plot(
        function=compare_fields,
        field=("u", "regrid_u"),
        label="X Velocity Difference",
        latex=("$u$", r"$u_{regrid}$"),
    ),
    Plot(
        function=compare_fields,
        field=("v", "regrid_v"),
        label="Y Velocity Difference",
        latex=("$v$", r"$v_{regrid}$"),
    ),
    Plot(
        function=compare_speed,
        field=("u", "v", "regrid_u", "regrid_v"),
        label="Speed Difference",
        latex=("$u$", "$v$", r"$u_{regrid}$", r"$v_{regrid}$"),
    ),
    Plot(
        function=partial(plot_side_by_side, log_first=True, log_second=True, diff=True),
        field=("e_eff", "mu"),
        label="Log Effective Strain Rate and Log of the Viscosity Difference",
        latex=(r"$\dot{\epsilon}_{eff}$", r"$\Delta\log\mu$"),
    ),
    Plot(
        function=partial(plot_side_by_side, log_first=True, log_second=True, diff=True),
        field=("e_eff", "delta log(mu)"),
        label="Log Effective Strain Rate and Difference of the Log Viscosities",
        latex=(r"$\dot{\epsilon}_{eff}$", r"$\log(\Delta\mu)$"),
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="speed",
        label="Speed Difference",  
        latex=r"$\Delta\sqrt{u^2+v^2}$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="h",
        label="Thickness Difference",  
        latex=r"$\Delta\mu$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="mu",
        label="Viscosity Difference",  
        latex=r"$\Delta\mu$",
    ),
    Plot(
        function=partial(plot_log_field, diff=True),
        field="mu",
        label="Log of the Difference in Viscosity",
        latex=r"$\log(\Delta\mu)$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="delta log(mu)",
        label="Difference of Log Viscosity",
        latex=r"$\Delta\log\mu$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="e_eff",
        label="Effective Strain Rate Difference",  
        latex=r"$\Delta\dot{\epsilon}_{eff}$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="e_eff",
        label="Log Effective Strain Rate Difference",  
        latex=r"$\Delta\dot{\epsilon}_{eff}$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="e_xx",
        label="Strain Rate XX Component Difference",  
        latex=r"$\Delta\dot{\epsilon}_{xx}$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="e_xy",
        label="Strain Rate XY Component Difference",  
        latex=r"$\Delta\dot{\epsilon}_{xy}$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="e_yy",
        label="Strain Rate YY Component Difference",  
        latex=r"$\Delta\dot{\epsilon}_{yy}$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="tau_eff",
        label="Effective Stress Difference",  
        latex=r"$\Delta\tau_{eff}$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="tau_xx",
        label="Stress XX Component Difference", 
        latex=r"$\Delta\tau_{xx}$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="tau_xy",
        label="Stress XY Component Difference",  
        latex=r"$\Delta\tau_{xy}$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="tau_yy",
        label="Stress YY Component Difference",  
        latex=r"$\Delta\tau_{yy}$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="tau_1",
        label="Stress Primary Component Difference",  
        latex=r"$\Delta\tau_{1}$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="tau_2",
        label="Stress Secondary Component Difference", 
        latex=r"$\Delta\tau_{2}$",
    ),
    Plot(
        function=partial(plot_field, diff=True),
        field="R_xx",
        label="Extensional Stress Difference (R_xx)",  
        latex=r"$\Delta R_{xx}$",
    ),
]

parser = ArgumentParser()
parser.add_argument("--folder", required=True)
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
