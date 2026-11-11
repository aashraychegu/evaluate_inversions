from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from load_trained_network import load_trained_network
from shapely import wkt, Polygon, GeometryCollection, MultiPolygon
from shapely import contains_xy
import tomllib
import shutil
import json
import numpy as np
import pyproj
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rxr
from scipy.interpolate import griddata

parser = ArgumentParser()
parser.add_argument("--folder", required = True)
folder = Path(parser.parse_args().folder)

subfolders = sorted(list(folder.glob("*/")))

def to_grid(flat_values, mask):
    grid = np.full(mask.shape, np.nan)
    grid[mask] = np.asarray(flat_values).ravel()
    return grid

# Julian year
sec_in_year = 3.1557600E7

def regrid_scalar(src_x, src_y, src_values, X, Y, valid_mask):

    src_x = np.asarray(src_x).ravel()
    src_y = np.asarray(src_y).ravel()
    src_values = np.asarray(src_values).ravel()

    # Drop NaN source points so they don't corrupt the interpolation
    finite = np.isfinite(src_x) & np.isfinite(src_y) & np.isfinite(src_values)
    src_points = np.column_stack([src_x[finite], src_y[finite]])
    src_vals = src_values[finite]

    grid = griddata(
        src_points,
        src_vals,
        (X, Y),
        method="linear",   
        fill_value=np.nan,
    )

    # Restrict to inside the polygon
    grid = np.where(valid_mask, grid, np.nan)
    return grid

def outer_shell(geom):
    if isinstance(geom, GeometryCollection):
        parts = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        geom = max(parts, key=lambda g: g.area)
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    return Polygon(geom.exterior)

def largest_polygon(geom):
    if isinstance(geom, GeometryCollection):
        parts = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not parts:
            raise ValueError("GeometryCollection contains no polygonal geometry")
        geom = max(parts, key=lambda g: g.area)
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom 

buffersize = .1

data_vars = {
        ("regrid_h",): None,
        ("regrid_u",): None, 
        ("regrid_v",): None, 
        ("mask",): None, 
        ("h",): None,
        ("u",): None,
        ("v",): None,
        ("e_eff",): None,
        ("e_xx",): None, 
        ("e_xy",): None, 
        ("e_yy",): None, 
        ("tau_eff",): None, 
        ("tau_xx",): None, 
        ("tau_xy",): None, 
        ("tau_yy",): None, 
        ("tau_1",): None, 
        ("tau_2",): None, 
        ("R_xx",): None,
}

shelves_path = Path(".shelves.toml")
shelf_to_bbox = tomllib.loads(shelves_path.read_text())


pad = 0.02

# pyrefly: ignore [invalid-syntax]
for subfolder in (pbar := tqdm(subfolders, smoothing = 0, position = 0, dynamic_ncols= True)):
    pbar.set_description(f"Processing {subfolder.name:<75}")

    name = subfolder.name
    shelf = name.split("__")[0]
    res = shelf_to_bbox[shelf].get("res",500)
    
    fields_and_residuals_path = subfolder / "field_and_residual_data.npz"
    raw_source = np.load(fields_and_residuals_path)
    
    orig_uv_x = raw_source["x"] 
    orig_uv_y = raw_source["y"]
    orig_u = raw_source["u_true"] / sec_in_year
    orig_v = raw_source["v_true"] / sec_in_year
    orig_h_x = raw_source["x_h"]
    orig_h_y = raw_source["y_h"]
    orig_h = raw_source["h_true"]

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
    
    tqdm.write(f"{polygon.geom_type} {polygon_path} \n{"-"*100}")
    outer_polygon = largest_polygon(polygon)

    max_x, min_x, max_y, min_y = shelf_to_bbox[shelf]["bbox"]
    extent_x, extent_y = max_x - min_x, max_y - min_y
    buffer_x, buffer_y = extent_x * buffersize, extent_y * buffersize
    maxx, minx, maxy, miny = max_x + buffer_x, min_x - buffer_x, max_y + buffer_y, min_y - buffer_y

    x = np.arange(minx, maxx, res)
    y = np.arange(miny, maxy, res)
    X, Y = np.meshgrid(x, y)
    valid_data_mask = contains_xy(polygon, X, Y)
    inside_mask = contains_xy(outer_polygon, X, Y)

    # --- Regrid original fields onto the new grid (no extrapolation) ---
    regrid_h = regrid_scalar(orig_h_x,  orig_h_y,  orig_h, X, Y, valid_data_mask)
    regrid_u = regrid_scalar(orig_uv_x, orig_uv_y, orig_u, X, Y, valid_data_mask)
    regrid_v = regrid_scalar(orig_uv_x, orig_uv_y, orig_v, X, Y, valid_data_mask)

    stacked_xy = np.column_stack([X[inside_mask], Y[inside_mask]])

    strain_rate = network.strain_rate(stacked_xy)
    fields = network.field(stacked_xy)

    mu    = to_grid(fields["mu"] * sec_in_year,              inside_mask)
    u     = to_grid(fields["u"] / sec_in_year, inside_mask) 
    v     = to_grid(fields["v"] / sec_in_year, inside_mask)
    h     = to_grid(fields["h"],               inside_mask)
 
    du_dx = to_grid(strain_rate["u_x"]      /  sec_in_year   , inside_mask)
    du_dy = to_grid(strain_rate["u_y"]      /  sec_in_year   , inside_mask)
    dv_dx = to_grid(strain_rate["v_x"]      /  sec_in_year   , inside_mask)
    dv_dy = to_grid(strain_rate["v_y"]      / sec_in_year   , inside_mask)
    e_eff = to_grid(strain_rate["e_eff"]    / sec_in_year , inside_mask)
    e_xx  = to_grid(strain_rate["exx"]      / sec_in_year   , inside_mask)
    e_xy  = to_grid(strain_rate["exy"]      / sec_in_year   , inside_mask)
    e_yy  = to_grid(strain_rate["eyy"]      / sec_in_year   , inside_mask)

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

    ds = xr.Dataset(
        data_vars=dict(
            mask      = ( ("y", "x"), valid_data_mask   ),
            regrid_h  = ( ("y", "x"), regrid_h          ),
            regrid_u  = ( ("y", "x"), regrid_u          ),
            regrid_v  = ( ("y", "x"), regrid_v          ),
            mu        = ( ("y", "x"), mu                ),
            u         = ( ("y", "x"), u                 ),
            v         = ( ("y", "x"), v                 ),
            h         = ( ("y", "x"), h                 ),
            du_dx     = ( ("y", "x"), du_dx             ),
            du_dy     = ( ("y", "x"), du_dy             ),
            dv_dx     = ( ("y", "x"), dv_dx             ),
            dv_dy     = ( ("y", "x"), dv_dy             ),
            e_eff     = ( ("y", "x"), e_eff             ),
            e_xx      = ( ("y", "x"), e_xx              ),
            e_xy      = ( ("y", "x"), e_xy              ),
            e_yy      = ( ("y", "x"), e_yy              ),
            tau_eff   = ( ("y", "x"), tau_eff           ),
            tau_xx    = ( ("y", "x"), tau_xx            ),
            tau_xy    = ( ("y", "x"), tau_xy            ),
            tau_yy    = ( ("y", "x"), tau_yy            ),
            tau_1     = ( ("y", "x"), tau_1             ),
            tau_2     = ( ("y", "x"), tau_2             ),
            R_xx      = ( ("y", "x"), R_xx              ),
        ),  
        coords=dict(
            x=("x", x),
            y=("y", y),
        ),
    )

    ds = ds.rio.write_crs("EPSG:3031")
    
    zarr_path = subfolder / f"evaluations.zarr"
    if zarr_path.exists(): shutil.rmtree(zarr_path)
    ds.to_zarr(zarr_path, mode="w", consolidated=False)

    tqdm.write(f"Zarr saved to:  {zarr_path}")