import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import tomllib
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Dict, List

from row_plotting_helpers import Plot, plot_field, plot_log_field



all_plot_vars = {
    "speed"         : Plot(function = plot_field, field = "speed", latex = r"$\sqrt(u^2 + v^2)$", label = "Speed", cbmin = None, cbmax = None),
    "speed_err"         : Plot(function = plot_field, field = "speed_err", latex = r"$\sqrt(u^2 + v^2) - \sqrt(u_{pred}^2 + v_{pred}^2)$", label = "Speed Error", cbmin = None, cbmax = None),
    "height"         : Plot(function = plot_field, field = "h", latex = r"$h$", label = "Height", cbmin = None, cbmax = None),
    "height_err"         : Plot(function = plot_field, field = "h_err", latex = "r$h - h_{pred}$", label = "Height Error", cbmin = None, cbmax = None),

    "mu"            : Plot(function=plot_field,     field="mu",         latex=r"$\mu$",                       label="Viscosity",                   cbmin=None, cbmax=None),
    "log_mu"        : Plot(function=plot_log_field, field="mu",         latex=r"$log(\mu)$",                  label="Log Viscosity",               cbmin=None, cbmax=None),
    "e_eff"         : Plot(function=plot_field,     field="e_eff",      latex=r"$\dot \epsilon_{eff}$",       label="Effective Strain Rate",       cbmin=None, cbmax=None),
    "log_e_eff"     : Plot(function=plot_log_field, field="e_eff",      latex=r"$log(\dot \epsilon_{eff})$",  label="Log Effective Strain Rate",   cbmin=None, cbmax=None),
    "e_xx"          : Plot(function=plot_field,     field="e_xx",       latex=r"$\dot \epsilon_{xx}$",        label="Strain Rate XX Component",    cbmin=None, cbmax=None),
    "e_xy"          : Plot(function=plot_field,     field="e_xy",       latex=r"$\dot \epsilon_{xy}$",        label="Strain Rate XY Component",    cbmin=None, cbmax=None),
    "e_yy"          : Plot(function=plot_field,     field="e_yy",       latex=r"$\dot \epsilon_{yy}$",        label="Strain YY Component",         cbmin=None, cbmax=None),
    "tau_eff"       : Plot(function=plot_field,     field="tau_eff",    latex=r"$\tau_{eff}$",                label="Effective Stress",            cbmin=None, cbmax=None),
    "tau_xx"        : Plot(function=plot_field,     field="tau_xx",     latex=r"$\tau_{xx}$",                 label="Stress XX Component",         cbmin=None, cbmax=None),
    "tau_xy"        : Plot(function=plot_field,     field="tau_xy",     latex=r"$\tau_{xy}$",                 label="Stress XY Component",         cbmin=None, cbmax=None),
    "tau_yy"        : Plot(function=plot_field,     field="tau_yy",     latex=r"$\tau_{yy}$",                 label="Stress YY Component",         cbmin=None, cbmax=None),
    "tau_1"         : Plot(function=plot_field,     field="tau_1",      latex=r"$\tau_{1}$",                  label="Stress Primary Component",    cbmin=None, cbmax=None),
    "tau_2"         : Plot(function=plot_field,     field="tau_2",      latex=r"$\tau_{2}$",                  label="Stress Secondary Component",  cbmin=None, cbmax=None),
    "R_xx"          : Plot(function=plot_field,     field="R_xx",       latex=r"$R_{xx}$",                    label="Extensional Stress (R_xx)",   cbmin=None, cbmax=None),

    "anom_mu"       : Plot(function=plot_field,     anomaly=True, field="mu",      latex=r"$\text{Anomaly of} \mu$",                       label="Anomaly of Viscosity",                     cbmin=None, cbmax=None),
    "anom_log_mu"   : Plot(function=plot_log_field, anomaly=True, field="mu",      latex=r"$\text{Anomaly of} log(\mu)$",                  label="Anomaly of Log Viscosity",                 cbmin=None, cbmax=None),
    "anom_e_eff"    : Plot(function=plot_field,     anomaly=True, field="e_eff",   latex=r"$\text{Anomaly of} \dot \epsilon_{eff}$",       label="Anomaly of Effective Strain Rate",         cbmin=None, cbmax=None),
    "anom_log_e_eff": Plot(function=plot_log_field, anomaly=True, field="e_eff",   latex=r"$\text{Anomaly of} log(\dot \epsilon_{eff})$",  label="Anomaly of Log Effective Strain Rate",     cbmin=None, cbmax=None),
    "anom_e_xx"     : Plot(function=plot_field,     anomaly=True, field="e_xx",    latex=r"$\text{Anomaly of} \dot \epsilon_{xx}$",        label="Anomaly of Strain Rate XX Component",      cbmin=None, cbmax=None),
    "anom_e_xy"     : Plot(function=plot_field,     anomaly=True, field="e_xy",    latex=r"$\text{Anomaly of} \dot \epsilon_{xy}$",        label="Anomaly of Strain Rate XY Component",      cbmin=None, cbmax=None),
    "anom_e_yy"     : Plot(function=plot_field,     anomaly=True, field="e_yy",    latex=r"$\text{Anomaly of} \dot \epsilon_{yy}$",        label="Anomaly of Strain YY Component",           cbmin=None, cbmax=None),
    "anom_tau_eff"  : Plot(function=plot_field,     anomaly=True, field="tau_eff", latex=r"$\text{Anomaly of} \tau_{eff}$",                label="Anomaly of Effective Stress",              cbmin=None, cbmax=None),
    "anom_tau_xx"   : Plot(function=plot_field,     anomaly=True, field="tau_xx",  latex=r"$\text{Anomaly of} \tau_{xx}$",                 label="Anomaly of Stress XX Component",           cbmin=None, cbmax=None),
    "anom_tau_xy"   : Plot(function=plot_field,     anomaly=True, field="tau_xy",  latex=r"$\text{Anomaly of} \tau_{xy}$",                 label="Anomaly of Stress XY Component",           cbmin=None, cbmax=None),
    "anom_tau_yy"   : Plot(function=plot_field,     anomaly=True, field="tau_yy",  latex=r"$\text{Anomaly of} \tau_{yy}$",                 label="Anomaly of Stress YY Component",           cbmin=None, cbmax=None),
    "anom_tau_1"    : Plot(function=plot_field,     anomaly=True, field="tau_1",   latex=r"$\text{Anomaly of} \tau_{1}$",                  label="Anomaly of Stress Primary Component",      cbmin=None, cbmax=None),
    "anom_tau_2"    : Plot(function=plot_field,     anomaly=True, field="tau_2",   latex=r"$\text{Anomaly of} \tau_{2}$",                  label="Anomaly of Stress Secondary Component",    cbmin=None, cbmax=None),
    "anom_R_xx"     : Plot(function=plot_field,     anomaly=True, field="R_xx",    latex=r"$\text{Anomaly of} R_{xx}$",                    label="Anomaly of Extensional Stress (R_xx)",     cbmin=None, cbmax=None),
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
    key: xr.open_zarr(folder / path / "evaluations.zarr", consolidated=False)
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