from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from load_trained_network import load_trained_network
from shapely import wkt
from shapely import contains_xy
import tomllib
import shutil
import json
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pyproj
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import xarray as xr
# pyrefly: ignore [missing-import]
import rioxarray as rxr
from time import perf_counter
from functools import partial
from plotting_helpers import Plot, plot_field, plot_log_field, compare_fields, compare_speed, plot_side_by_side, plot_stress_to_strain_rate, histogram

parser = ArgumentParser()
parser.add_argument("--folder", required = True)
folder = Path(parser.parse_args().folder)

subfolders = sorted(list(folder.glob("*/")))

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
        function=partial(plot_side_by_side, log_first=True, log_second=True),
        field=("e_eff", "mu"),
        label="Log Effective Strain Rate and Log Viscosity",
        latex=(r"$\dot{\epsilon}_{eff}$", r"$\mu$"),
    ),
    Plot(
        function=plot_field,
        field="mu",
        label="Viscosity",
        latex=r"$\mu$",
    ),
    Plot(
        function=plot_log_field,
        field="mu",
        label="Log Viscosity",
        latex=r"$\mu$",
    ),
    Plot(
        function=plot_field,
        field="e_eff",
        label="Effective Strain Rate",
        latex=r"$\dot{\epsilon}_{eff}$",
    ),
    Plot(
        function=plot_log_field,
        field="e_eff",
        label="Log Effective Strain Rate",
        latex=r"$\dot{\epsilon}_{eff}$",
    ),
    Plot(
        function=plot_field,
        field="e_xx",
        label="Strain Rate XX Component",
        latex=r"$\dot{\epsilon}_{xx}$",
    ),
    Plot(
        function=plot_field,
        field="e_xy",
        label="Strain Rate XY Component",
        latex=r"$\dot{\epsilon}_{xy}$",
    ),
    Plot(
        function=plot_field,
        field="e_yy",
        label="Strain Rate YY Component",  
        latex=r"$\dot{\epsilon}_{yy}$",
    ),
    Plot(
        function=plot_field,
        field="tau_eff",
        label="Effective Stress",
        latex=r"$\tau_{eff}$",
    ),
    Plot(
        function=plot_field,
        field="tau_xx",
        label="Stress XX Component",
        latex=r"$\tau_{xx}$",
    ),
    Plot(
        function=plot_field,
        field="tau_xy",
        label="Stress XY Component",
        latex=r"$\tau_{xy}$",
    ),
    Plot(
        function=plot_field,
        field="tau_yy",
        label="Stress YY Component",
        latex=r"$\tau_{yy}$",
    ),
    Plot(
        function=plot_field,
        field="tau_1",
        label="Stress Primary Component",
        latex=r"$\tau_{1}$",
    ),
    Plot(
        function=plot_field,
        field="tau_2",
        label="Stress Secondary Component",
        latex=r"$\tau_{2}$",
    ),
    Plot(
        function=plot_field,
        field="R_xx",
        label="Extensional Stress (R_xx)",
        latex=r"$R_{xx}$",
    ),
    Plot(
        function=histogram,
        field="e_eff",
        label="Histogram of Effective Strain Rate", 
        latex=r"$\dot{\epsilon}_{eff}$",
    ),
    Plot(
        function=plot_stress_to_strain_rate,
        field=("tau_eff", "e_eff"),
        label="Effective Stress to Effective Strain",
        latex=(r"$\tau_{eff}$", r"$\dot{\epsilon}_{eff}$"),
    ),
]

pad = 0.02

for subfolder in (pbar := tqdm(subfolders, smoothing = 0, position = 0, dynamic_ncols= True)):
    pbar.set_description(f"Plotting: {subfolder.name}")
    
    ds = xr.open_zarr(subfolder / f"evaluations.zarr", consolidated = False)

    img_dir = subfolder / "eval_images"
    img_dir.mkdir(exist_ok = True)
    for file in img_dir.iterdir():
        file.unlink()

    for plot_spec in (pbar2:= tqdm(plot_vars)):
        pbar2.set_description(f"Plotting: {plot_spec.label} ")
        save_path, spec_str = plot_spec(ds, subfolder.name, img_dir)
        # tqdm.write(f"Saved {spec_str} to {save_path}")