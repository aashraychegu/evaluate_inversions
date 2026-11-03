from typing import Dict, List, Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
from pathlib import Path
import xarray as xr
import tomllib
from argparse import ArgumentParser
from cmocean import cm
from tqdm import tqdm

def _get_2d(ds: xr.Dataset, field: str) -> xr.DataArray:
    """Pull a field out of a dataset and make sure it is a 2D (y, x) array."""
    da = ds[field].squeeze()
    if set(da.dims) != {"x", "y"}:
        raise ValueError(
            f"Field '{field}' has dims {da.dims}; expected only ('y', 'x'). "
            "Select/reduce extra dimensions before plotting."
        )
    return da.transpose("y", "x").compute()


def _valid_extent(da: xr.DataArray):
    """Bounding box (xmin, xmax, ymin, ymax) of the non-NaN region."""
    valid = da.notnull()
    xv = da["x"].values[valid.any("y").values]
    yv = da["y"].values[valid.any("x").values]
    if xv.size == 0 or yv.size == 0:
        return None
    return xv.min(), xv.max(), yv.min(), yv.max()


def _plot_row(
    ds: Dict[str, xr.Dataset],
    field: str,
    ax: Sequence[Axes],
    save_name: str,
    latex: Optional[str],
    norm_factory,
    cmap,
    title_suffix: str = "",
    pad: float = 0.02,
    anomaly_func = None,
):
    """Shared logic: plot `field` from every dataset across one row of axes."""
    if latex is None:
        latex = field

    names = list(ds.keys())
    das = [_get_2d(ds[name], field) for name in names]

    if anomaly_func is not None:
        all_values = np.concatenate([da.values for da in das])
        median = np.median(all_values)
        for da in das:
            da = anomaly_func(da,median)        

    # Common color limits across the row
    vmin = float(np.nanmin([np.nanmin(da.values) for da in das]))
    vmax = float(np.nanmax([np.nanmax(da.values) for da in das]))
    norm = norm_factory(vmin, vmax)

    # Common spatial extent across the row (union of valid regions)
    extents = [e for e in (_valid_extent(da) for da in das) if e is not None]
    if extents:
        xmin = min(e[0] for e in extents)
        xmax = max(e[1] for e in extents)
        ymin = min(e[2] for e in extents)
        ymax = max(e[3] for e in extents)
        dx = (xmax - xmin) * pad
        dy = (ymax - ymin) * pad

    im = None
    for col, (name, da) in enumerate(zip(names, das)):
        a = ax[col]
        im = da.plot.imshow(
            ax=a,
            add_colorbar=False,
            norm=norm,
            cmap=cmap,
            robust = True,
        )
        a.set_aspect(1)
        a.set_title(rf"{name} : {latex}{title_suffix}", fontsize=10)
        if extents:
            a.set_xlim(float(xmin) - dx, float(xmax) + dx)
            a.set_ylim(float(ymin) - dy, float(ymax) + dy)

        # Only label outer axes to reduce clutter
        if col != 0:
            a.set_ylabel("")

    # Row label on the left-most axis
    ax[0].set_ylabel(f"{save_name}\n{ax[0].get_ylabel()}")

    # One colorbar for the whole row
    if im is not None:
        fig = ax[0].get_figure()
        cbar = fig.colorbar(im, ax=list(ax), shrink=0.9, pad=0.01) 
        cbar.set_label(rf"{latex}")

    return im

def subtract(subtrahend, minuend):
    return subtrahend - minuend

def divide(numerator, denominator):
    return numerator / denominator

def plot_field(
    ds: Dict[str, xr.Dataset],
    field: str,
    ax: Sequence[Axes],
    save_name: str,
    latex: Optional[str] = None,
    pad: float = 0.02,
    diff: bool = False,
    anomaly = False
):
    cmap = cm.curl if diff else cm.thermal
    
    anomaly_func = None
    if anomaly:
        anomaly_func = subtract

    def norm_factory(vmin, vmax):
        if diff:  # symmetric about zero for difference plots
            lim = max(abs(vmin), abs(vmax))
            return Normalize(vmin=-lim, vmax=lim)
        return Normalize(vmin=vmin, vmax=vmax)

    return _plot_row(ds, field, ax, save_name, latex, norm_factory, cmap, pad=pad, anomaly_func = anomaly_func)

def plot_log_field(
    ds: Dict[str, xr.Dataset],
    field: str,
    ax: Sequence[Axes],
    save_name: str,
    latex: Optional[str] = None,
    pad: float = 0.02,
    diff: bool = False,
    linthresh: float = 1.0,
    anomaly = False
):
    cmap = cm.delta if diff else cm.dense

    def norm_factory(vmin, vmax):
        if diff:
            lim = max(abs(vmin), abs(vmax))
            return SymLogNorm(linthresh=linthresh, vmin=-lim, vmax=lim)
        return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
    
    anomaly_func = None
    if anomaly:
        anomaly_func = divide
    
    return _plot_row(
        ds, field, ax, save_name, latex, norm_factory, cmap,
        title_suffix=" (log scale)", pad=pad, anomaly_func = anomaly_func,
    )


class Plot:
    def __init__(self, function, field, label, anomaly = False, latex=None):
        self.field = field
        self.function = function
        self.label = label
        self.latex = latex if latex is not None else f"${field}$"
        self.anomaly = anomaly

    def __call__(self, ds, ax):
        return self.function(
            ds=ds, field=self.field, ax=ax,
            save_name=self.label, latex=self.latex,
        )


all_plot_vars = {
    "mu"        : Plot(function=plot_field,     field="mu", label="Viscosity"),
    "log_mu"    : Plot(function=plot_log_field, field="mu", label="Log Viscosity"),
    "e_eff"     : Plot(function=plot_field,     field="e_eff", label="Effective Strain Rate"),
    "log_e_eff" : Plot(function=plot_log_field, field="e_eff", label="Log Effective Strain Rate"),
    "e_xx"      : Plot(function=plot_field,     field="e_xx", label="Strain Rate XX Component"),
    "e_xy"      : Plot(function=plot_field,     field="e_xy", label="Strain Rate XY Component"),
    "e_yy"      : Plot(function=plot_field,     field="e_yy", label="Strain YY Component"),
    "tau_eff"   : Plot(function=plot_field,     field="tau_eff", label="Effective Stress"),
    "tau_xx"    : Plot(function=plot_field,     field="tau_xx", label="Stress XX Component"),
    "tau_xy"    : Plot(function=plot_field,     field="tau_xy", label="Stress XY Component"),
    "tau_yy"    : Plot(function=plot_field,     field="tau_yy", label="Stress YY Component"),
    "tau_1"     : Plot(function=plot_field,     field="tau_1", label="Stress Primary Component"),
    "tau_2"     : Plot(function=plot_field,     field="tau_2", label="Stress Secondary Component"),
    "R_xx"      : Plot(function=plot_field,     field="R_xx", label="Extensional Stress (R_xx)"),
    "anom_mu"   : Plot(function=plot_field,     field="mu", label="Viscosity"),
    "anom_log_mu"   : Plot(function=plot_log_field, anomaly = True, field="mu",      label="Anomaly of Log Viscosity"),
    "anom_e_eff"    : Plot(function=plot_field,     anomaly = True, field="e_eff",   label="Anomaly of Effective Strain Rate"),
    "anom_log_e_eff": Plot(function=plot_log_field, anomaly = True, field="e_eff",   label="Anomaly of Log Effective Strain Rate"),
    "anom_e_xx"     : Plot(function=plot_field,     anomaly = True, field="e_xx",    label="Anomaly of Strain Rate XX Component"),
    "anom_e_xy"     : Plot(function=plot_field,     anomaly = True, field="e_xy",    label="Anomaly of Strain Rate XY Component"),
    "anom_e_yy"     : Plot(function=plot_field,     anomaly = True, field="e_yy",    label="Anomaly of Strain YY Component"),
    "anom_tau_eff"  : Plot(function=plot_field,     anomaly = True, field="tau_eff", label="Anomaly of Effective Stress"),
    "anom_tau_xx"   : Plot(function=plot_field,     anomaly = True, field="tau_xx",  label="Anomaly of Stress XX Component"),
    "anom_tau_xy"   : Plot(function=plot_field,     anomaly = True, field="tau_xy",  label="Anomaly of Stress XY Component"),
    "anom_tau_yy"   : Plot(function=plot_field,     anomaly = True, field="tau_yy",  label="Anomaly of Stress YY Component"),
    "anom_tau_1"    : Plot(function=plot_field,     anomaly = True, field="tau_1",   label="Anomaly of Stress Primary Component"),
    "anom_tau_2"    : Plot(function=plot_field,     anomaly = True, field="tau_2",   label="Anomaly of Stress Secondary Component"),
    "anom_R_xx"     : Plot(function=plot_field,     anomaly = True, field="R_xx",    label="Anomaly of Extensional Stress (R_xx)"),
}


parser = ArgumentParser()
parser.add_argument("--folder", required=True)
parser.add_argument("--spec-file", default="fields_over_time.toml")
args = parser.parse_args()

folder = Path(args.folder)
spec_file = tomllib.loads((folder / args.spec_file).read_text())

inversions: Dict = spec_file["inversions"]
fields: List = spec_file["fields"]

plot_vars = [all_plot_vars[field] for field in fields]
name_to_zarr: Dict[str, xr.Dataset] = {
    key: xr.open_zarr(folder / path / "stress.zarr", consolidated=False)
    for key, path in inversions.items()
}

n_rows, n_cols = len(plot_vars), len(name_to_zarr)
fig, ax = plt.subplots(
    n_rows, n_cols,
    figsize=(4.5 * n_cols, 4 * n_rows),
    squeeze=False,           # always 2D, even for 1 row / 1 column
    layout="constrained",    # keeps the per-row colorbars tidy
)

for row, plot in tqdm(enumerate(plot_vars),desc = "Plotting Rows"):
    plot(ds=name_to_zarr, ax=ax[row])

save_path = folder / "Fields Over Time.png"
fig.savefig(save_path, dpi=200)
plt.close(fig)