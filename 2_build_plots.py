
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

parser = ArgumentParser()
parser.add_argument("--folder", required = True)
folder = Path(parser.parse_args().folder)

subfolders = sorted(list(folder.glob("*/")))

def to_grid(flat_values, mask):
    grid = np.full(mask.shape, np.nan)
    grid[mask] = np.asarray(flat_values).ravel()
    return grid


buffersize = .1

plotvars = ["h","vx","vy","speed","e_eff", "e_xx", "e_xy", "e_yy", "tau_eff", "tau_xx", "tau_xy", "tau_yy", "tau_1", "tau_2", "R_xx"]

shelves_path = Path(".shelves.toml")
shelf_to_bbox = tomllib.loads(shelves_path.read_text())


pad = 0.02

# pyrefly: ignore [invalid-syntax]
for subfolder in (pbar := tqdm(subfolders, smoothing = 0, position = 0, dynamic_ncols= True)):
    t0 = perf_counter()
    pbar.set_description(f"Processing {subfolder.name:<75}")

    name = subfolder.name
    shelf = name.split("__")[0]
    res = shelf_to_bbox[shelf].get("res",500)

    config = json.loads((subfolder / "config.json").read_text())
    source_file = Path(config["data"]["source"])
    parent_dir = source_file.parent
    yaml_path = parent_dir / f"{name}.yaml"
    polygon_path = parent_dir / f"{name}_polygon.wkt"

    if not source_file.exists():
        tqdm.write(f"skipping {subfolder.name}")
        continue
    
    network = load_trained_network(yaml_path,subfolder)

    split = name.split("__")
    shelf = split[0]
    velocity_name = split[1]
    thickness_name = split[2]

    polygon = wkt.loads(polygon_path.read_text())

    max_x, min_x, max_y, min_y = shelf_to_bbox[shelf]["bbox"]
    extent_x, extent_y = max_x - min_x, max_y - min_y
    buffer_x, buffer_y = extent_x * buffersize, extent_y * buffersize
    maxx, minx, maxy, miny = max_x + buffer_x, min_x - buffer_x, max_y + buffer_y, min_y - buffer_y

    x = np.arange(minx, maxx, res)
    y = np.arange(miny, maxy, res)
    X, Y = np.meshgrid(x, y)
    inside_mask = contains_xy(polygon, X, Y)

    stacked_xy = np.column_stack([X[inside_mask], Y[inside_mask]])

    t1 = perf_counter()
    strain_rate = network.strain_rate(stacked_xy)
    fields = network.field(stacked_xy)
    t2 = perf_counter()
    mu    = to_grid(fields["mu"], inside_mask)
    u     = to_grid(fields["u"] , inside_mask) 
    v     = to_grid(fields["v"] , inside_mask)
    h     = to_grid(fields["h"] , inside_mask)

    du_dx = to_grid(strain_rate["u_x"]  ,   inside_mask)
    du_dy = to_grid(strain_rate["u_y"]  ,   inside_mask)
    dv_dx = to_grid(strain_rate["v_x"]  ,   inside_mask)
    dv_dy = to_grid(strain_rate["v_y"]  ,   inside_mask)
    e_eff = to_grid(strain_rate["e_eff"],   inside_mask)
    e_xx  = to_grid(strain_rate["exx"]  ,   inside_mask)
    e_xy  = to_grid(strain_rate["exy"]  ,   inside_mask)
    e_yy  = to_grid(strain_rate["eyy"]  ,   inside_mask)

    tau_xx = 2 * mu * e_xx
    tau_xy = 2 * mu * e_xy
    tau_yy = 2 * mu * e_yy
    tau_eff = 2 * mu * e_eff

    eigen_stress_prefactor = (tau_xx + tau_yy) / 2
    inside_radical_section = ((tau_xx - tau_yy) ** 2) / 4 + tau_xy**2
    eigen_stress_radical = np.sqrt(inside_radical_section)

    tau_1 = eigen_stress_prefactor + eigen_stress_radical
    tau_2 = eigen_stress_prefactor - eigen_stress_radical
    R_xx = 2 * tau_1 + tau_2
    t3 = perf_counter()
    ds = xr.Dataset(
        data_vars=dict(
            mu      = ( ("y", "x"), mu       ),
            u       = ( ("y", "x"), u        ),
            v       = ( ("y", "x"), v        ),
            h       = ( ("y", "x"), h        ),
            du_dx   = ( ("y", "x"), du_dx    ),
            du_dy   = ( ("y", "x"), du_dy    ),
            dv_dx   = ( ("y", "x"), dv_dx    ),
            dv_dy   = ( ("y", "x"), dv_dy    ),
            e_eff   = ( ("y", "x"), e_eff    ),
            e_xx    = ( ("y", "x"), e_xx     ),
            e_xy    = ( ("y", "x"), e_xy     ),
            e_yy    = ( ("y", "x"), e_yy     ),
            tau_eff = ( ("y", "x"), tau_eff  ),
            tau_xx  = ( ("y", "x"), tau_xx   ),
            tau_xy  = ( ("y", "x"), tau_xy   ),
            tau_yy  = ( ("y", "x"), tau_yy   ),
            tau_1   = ( ("y", "x"), tau_1    ),
            tau_2   = ( ("y", "x"), tau_2    ),
            R_xx    = ( ("y", "x"), R_xx     ),
        ),  
        coords=dict(
            x=("x", x),
            y=("y", y),
        ),
    )
    ds = ds.rio.write_crs("EPSG:3031")
    valid = None
    for var in ds.data_vars:
        v = ds[var].notnull()
        valid = v if valid is None else (valid | v)

    # Reduce to per-axis presence of any valid data
    valid_x = valid.any("y")   # over x
    valid_y = valid.any("x")   # over y

    xv = ds["x"].values[valid_x.values]
    yv = ds["y"].values[valid_y.values]

    if xv.size and yv.size:
        xmin, xmax = xv.min(), xv.max()
        ymin, ymax = yv.min(), yv.max()

        # slice by coordinate value (handles either ascending/descending order)
        ds = ds.sel(
            x=slice(min(xmin, xmax), max(xmin, xmax)),
            y=slice(min(ymin, ymax), max(ymin, ymax)),
        )
        # re-write CRS since sel can drop the spatial_ref association in some setups
        ds = ds.rio.write_crs("EPSG:3031")

    zarr_path = subfolder / f"stress.zarr"
    if zarr_path.exists(): shutil.rmtree(zarr_path)
    ds.to_zarr(zarr_path, mode="w", consolidated=False)

    imgs_save_dir = subfolder / "plots" / "stresses"
    if imgs_save_dir.exists(): shutil.rmtree(imgs_save_dir)
    imgs_save_dir.mkdir(exist_ok=True,parents = True)

    left, bottom, right, top = ds.rio.bounds()
    t4 = perf_counter()

    for field in tqdm(plotvars, smoothing = 0, position = 1, dynamic_ncols= True):
        itime0 = perf_counter()
        da = ds[field]
        valid = da.notnull()
        xv = da["x"].values[valid.any("y")]
        yv = da["y"].values[valid.any("x")]
        xmin, xmax = xv.min(), xv.max()
        ymin, ymax = yv.min(), yv.max()

        p = da.plot.imshow(aspect = 1, size = 10, add_colorbar = True,)

        dx = (xmax-xmin) * pad
        dy = (ymax-ymin) * pad
        lo, hi = ymin - dy, ymax + dy

        p.axes.set_title(f"{name} : {field}", fontsize=20)
        p.axes.set_xlim(float(xmin) - dx, float(xmax) + dx)
        p.axes.set_ylim(float(lo),float(hi))

        itime1 = perf_counter()
        plt.savefig(imgs_save_dir / f"{field}.png", dpi=150)
        itime2 = perf_counter()
        # tqdm.write(f"Time for data prep: {itime1-itime0} | Time for Saving: {itime2-itime1}")
        plt.close()
        
    t5 = perf_counter()
    # tqdm.write(f"Total: {t5-t0}\n\t Setup: {t1-t0}\n\t Forward Pass: {t2-t1}\n\t Field Calculations: {t3-t2}\n\t Zarring: {t4-t3}\n\t Plotting {t5-t4}")
    tqdm.write(f"Zarr saved to:  {zarr_path}")
    tqdm.write(f"Plots saved to: {imgs_save_dir}")