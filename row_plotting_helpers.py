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
    anomaly_func=None,
    cbmin: Optional[float] = None,
    cbmax: Optional[float] = None,
):
    """Shared logic: plot `field` from every dataset across one row of axes.

    cbmin / cbmax override the lower / upper colorbar limits. A limit left as
    None is computed automatically from the data (shared across the row).
    """
    if latex is None:
        latex = field

    names = list(ds.keys())
    das = [_get_2d(ds[name], field) for name in names]

    if anomaly_func is not None:
        all_values = np.concatenate([da.values.ravel() for da in das])
        median = np.nanmedian(all_values)
        das = [anomaly_func(da, median) for da in das]

    # Common color limits across the row (automatic)
    vmin = float(np.nanmin([np.nanmin(da.values) for da in das]))
    vmax = float(np.nanmax([np.nanmax(da.values) for da in das]))
    norm = norm_factory(vmin, vmax)

    # Explicit colorbar limits take precedence over the automatic ones
    if cbmin is not None:
        norm.vmin = float(cbmin)
    if cbmax is not None:
        norm.vmax = float(cbmax)

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
            robust=True,
        )
        a.set_aspect(1)
        a.set_title(rf"{name} : {latex}{title_suffix}", fontsize=5)
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
    diff: bool = True,
    anomaly: bool = False,
    cbmin: Optional[float] = None,
    cbmax: Optional[float] = None,
):
    cmap = cm.curl if diff else cm.thermal

    anomaly_func = subtract if anomaly else None

    def norm_factory(vmin, vmax):
        if diff:  # symmetric about zero for difference plots
            lim = max(abs(vmin), abs(vmax))
            return Normalize(vmin=-lim, vmax=lim)
        return Normalize(vmin=vmin, vmax=vmax)

    return _plot_row(
        ds, field, ax, save_name, latex, norm_factory, cmap,
        pad=pad, anomaly_func=anomaly_func, cbmin=cbmin, cbmax=cbmax,
    )


def plot_log_field(
    ds: Dict[str, xr.Dataset],
    field: str,
    ax: Sequence[Axes],
    save_name: str,
    latex: Optional[str] = None,
    pad: float = 0.02,
    diff: bool = True,
    linthresh: float = 1.0,
    anomaly: bool = False,
    cbmin: Optional[float] = None,
    cbmax: Optional[float] = None,
):
    cmap = cm.delta if diff else cm.dense

    def norm_factory(vmin, vmax):
        if diff:
            lim = max(abs(vmin), abs(vmax))
            return SymLogNorm(linthresh=linthresh, vmin=-lim, vmax=lim)
        return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)

    anomaly_func = divide if anomaly else None

    return _plot_row(
        ds, field, ax, save_name, latex, norm_factory, cmap,
        title_suffix=" (log scale)", pad=pad, anomaly_func=anomaly_func,
        cbmin=cbmin, cbmax=cbmax,
    )


class Plot:
    def __init__(self, function, field, label, anomaly=False, latex=None, cbmin=None, cbmax=None):
        self.field = field
        self.function = function
        self.label = label
        self.latex = latex if latex is not None else f"$${field}$$"
        self.anomaly = anomaly
        self.cbmin = cbmin
        self.cbmax = cbmax

    def __call__(self, ds, ax):
        return self.function(
            ds=ds, field=self.field, ax=ax,
            save_name=self.label, latex=self.latex,
            anomaly=self.anomaly,
            cbmin=self.cbmin, cbmax=self.cbmax,
        )