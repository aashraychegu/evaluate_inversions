from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import MaxNLocator
import cmocean
from cmocean import cm


def plot_field(ds, field: tuple[str], name, imgs_save_dir, save_name, latex=None, pad=0.02, diff=False,
               cbmin=None, cbmax=None) -> Path:
    if latex is None:
        latex = field
    save_path = imgs_save_dir / save_name
    data_field = field
    da = ds[data_field]
    valid = da.notnull()
    xv = da["x"].values[valid.any("y")]
    yv = da["y"].values[valid.any("x")]
    xmin, xmax = xv.min(), xv.max()
    ymin, ymax = yv.min(), yv.max()

    internal_cmap = cm.thermal
    if diff:
        internal_cmap = cm.curl

    p = da.plot.imshow(aspect=1, size=10, add_colorbar=True, cmap=internal_cmap,
                       vmin=cbmin, vmax=cbmax)
    if p.colorbar is not None:
        p.colorbar.set_label(rf"{latex}")

    dx = (xmax - xmin) * pad
    dy = (ymax - ymin) * pad

    p.axes.set_title(rf"{name} : {latex}", fontsize=10)
    p.axes.set_xlim(float(xmin) - dx, float(xmax) + dx)
    p.axes.set_ylim(float(ymin) - dy, float(ymax) + dy)

    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_log_field(ds, field: tuple[str], name, imgs_save_dir, save_name, latex=None, pad=0.02, diff=False,
                   cbmin=None, cbmax=None) -> Path:
    if latex is None:
        latex = field
    save_path = imgs_save_dir / save_name
    data_field = field
    da = ds[data_field]
    valid = da.notnull()
    xv = da["x"].values[valid.any("y")]
    yv = da["y"].values[valid.any("x")]
    xmin, xmax = xv.min(), xv.max()
    ymin, ymax = yv.min(), yv.max()

    internal_cmap = cm.dense
    if diff:
        internal_cmap = cm.delta

    norm = SymLogNorm(linthresh=1, vmin=cbmin, vmax=cbmax)
    p = da.plot.imshow(aspect=1, size=10, add_colorbar=True, norm=norm, cmap=internal_cmap)
    if p.colorbar is not None:
        p.colorbar.set_label(rf"{latex}")

    dx = (xmax - xmin) * pad
    dy = (ymax - ymin) * pad

    p.axes.set_title(rf"{name} : {latex} (log scale)", fontsize=10)
    p.axes.set_xlim(float(xmin) - dx, float(xmax) + dx)
    p.axes.set_ylim(float(ymin) - dy, float(ymax) + dy)

    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_field_notext(
    ds,
    field: tuple[str],
    name,
    imgs_save_dir,
    save_name,
    latex=None,
    pad=0.02,
    diff=False,
    grid=False,
    cmap=None,
    grid_kwargs: dict | None = None,
    n_gridlines: int = 10,
    cbmin=None,
    cbmax=None,
) -> Path:
    save_path = imgs_save_dir / save_name
    da = ds[field]

    valid = da.notnull()
    xv = da["x"].values[valid.any("y")]
    yv = da["y"].values[valid.any("x")]
    xmin, xmax = xv.min(), xv.max()
    ymin, ymax = yv.min(), yv.max()
    dx = (xmax - xmin) * pad
    dy = (ymax - ymin) * pad

    if cmap is None:
        cmap = cm.curl if diff else cm.thermal
    else:
        if cmap == 0:
            cmap = cm.balance
        if cmap == 1:
            cmap = cm.curl

    fig, ax = plt.subplots(figsize=(10, 10), layout="compressed")
    p = da.plot.imshow(
        ax=ax,
        cmap=cmap,
        cbar_kwargs={"label": "", "ticks": []},
        robust=True,          # only used for whichever limit is left as None
        vmin=cbmin,
        vmax=cbmax,
        extend="neither",
    )

    vmin, vmax = p.get_clim()
    p.colorbar.ax.set_title(f"{vmax:.3g}", fontsize=20)
    p.colorbar.ax.set_xlabel(f"{vmin:.3g}", fontsize=20)

    ax.set_xlim(float(xmin) - dx, float(xmax) + dx)
    ax.set_ylim(float(ymin) - dy, float(ymax) + dy)

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(which="both", left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_visible(False)

    if grid:
        ax.xaxis.set_major_locator(MaxNLocator(n_gridlines))
        ax.yaxis.set_major_locator(MaxNLocator(n_gridlines))
        gk = dict(color="k", linestyle="--", linewidth=0.5, alpha=0.35)
        if grid_kwargs:
            gk.update(grid_kwargs)
        ax.grid(True, **gk)
        ax.set_axisbelow(False)  # draw the grid on top of the image

    fig.savefig(save_path, dpi=512, bbox_inches="tight", pad_inches=0.125)
    plt.close(fig)
    return save_path


def compare_fields(ds, field: tuple[str, str], name, imgs_save_dir, save_name, latex=None, pad=0.02, diff=False,
                   cbmin=None, cbmax=None) -> Path:
    if latex is None:
        latex = field
    save_path = imgs_save_dir / save_name
    predicted_field, actual_field = field
    predicted_latex, actual_latex = latex

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(rf"{name} : {save_name}", fontsize=12)

    # Get data for both fields
    da_pred = ds[predicted_field]
    da_actual = ds[actual_field]

    # Find valid data points
    valid_pred = da_pred.notnull()
    valid_actual = da_actual.notnull()
    valid = valid_pred & valid_actual

    xv = da_pred["x"].values[valid.any("y")]
    yv = da_pred["y"].values[valid.any("x")]
    xmin, xmax = xv.min(), xv.max()
    ymin, ymax = yv.min(), yv.max()

    # Calculate padding
    dx = (xmax - xmin) * pad
    dy = (ymax - ymin) * pad

    # Shared color limits for the top two plots (overridable via cbmin/cbmax)
    shared_min = float(cbmin) if cbmin is not None else float(min(da_pred.min(), da_actual.min()))
    shared_max = float(cbmax) if cbmax is not None else float(max(da_pred.max(), da_actual.max()))

    # 1,1 - Predicted field (no individual colorbar)
    p1 = da_pred.plot.imshow(
        ax=axes[0, 0], add_colorbar=False, vmin=shared_min, vmax=shared_max, cmap=cm.thermal
    )
    axes[0, 0].set_title(rf"Predicted: {predicted_latex}")
    axes[0, 0].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[0, 0].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # 1,2 - Actual field (no individual colorbar)
    p2 = da_actual.plot.imshow(
        ax=axes[0, 1], add_colorbar=False, vmin=shared_min, vmax=shared_max, cmap=cm.haline
    )
    axes[0, 1].set_title(rf"Actual: {actual_latex}")
    axes[0, 1].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[0, 1].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # Single shared colorbar for the top row
    fig.colorbar(p1, ax=axes[0, :], shrink=0.8, label=rf"{predicted_latex}")

    # 2,1 - Difference (predicted - actual)
    diff_da = da_pred - da_actual
    diff_max = float(np.abs(diff_da).max())
    p3 = diff_da.plot.imshow(ax=axes[1, 0], add_colorbar=True, cmap=cm.balance, vmin=-diff_max, vmax=diff_max)
    axes[1, 0].set_title(rf"Difference: {predicted_latex} - {actual_latex}")
    axes[1, 0].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[1, 0].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # 2,2 - Relative difference ((predicted - actual)/actual)
    rel_diff = (da_pred - da_actual) / da_actual
    rel_diff_max = float(np.abs(rel_diff).max())
    p4 = rel_diff.plot.imshow(ax=axes[1, 1], add_colorbar=True, cmap=cm.balance, vmin=-rel_diff_max, vmax=rel_diff_max, robust=True)
    axes[1, 1].set_title(rf"Relative Difference: ({predicted_latex} - {actual_latex})/{actual_latex}")
    axes[1, 1].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[1, 1].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # Save (drop tight_layout — it conflicts with the manually-added colorbar)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def compare_speed(ds, field: tuple[str, str, str, str], name, imgs_save_dir, save_name, latex=None, pad=0.02, diff=False,
                  cbmin=None, cbmax=None) -> Path:
    if latex is None:
        latex = field
    save_path = imgs_save_dir / save_name
    predicted_x_field, predicted_y_field, actual_x_field, actual_y_field = field

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(rf"{name} : {save_name}", fontsize=12)

    da_pred_x = ds[predicted_x_field]
    da_pred_y = ds[predicted_y_field]
    da_actual_x = ds[actual_x_field]
    da_actual_y = ds[actual_y_field]

    pred_speed = np.sqrt(da_pred_x**2 + da_pred_y**2)
    actual_speed = np.sqrt(da_actual_x**2 + da_actual_y**2)

    valid_pred = pred_speed.notnull()
    valid_actual = actual_speed.notnull()
    valid = valid_pred & valid_actual

    xv = da_pred_x["x"].values[valid.any("y")]
    yv = da_pred_x["y"].values[valid.any("x")]
    xmin, xmax = xv.min(), xv.max()
    ymin, ymax = yv.min(), yv.max()

    dx = (xmax - xmin) * pad
    dy = (ymax - ymin) * pad

    # Shared color limits for the top two plots (overridable via cbmin/cbmax)
    shared_min = float(cbmin) if cbmin is not None else float(min(pred_speed.min(), actual_speed.min()))
    shared_max = float(cbmax) if cbmax is not None else float(max(pred_speed.max(), actual_speed.max()))

    # 1,1 - Predicted speed (no individual colorbar)
    p1 = pred_speed.plot.imshow(
        ax=axes[0, 0], add_colorbar=False, vmin=shared_min, vmax=shared_max, cmap=cm.thermal
    )
    axes[0, 0].set_title(r"Predicted Speed")
    axes[0, 0].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[0, 0].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # 1,2 - Actual speed (no individual colorbar)
    p2 = actual_speed.plot.imshow(
        ax=axes[0, 1], add_colorbar=False, vmin=shared_min, vmax=shared_max, cmap=cm.haline
    )
    axes[0, 1].set_title(r"Actual Speed")
    axes[0, 1].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[0, 1].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # Single shared colorbar for the top row
    fig.colorbar(p1, ax=axes[0, :], shrink=0.8, label="Speed")

    # 2,1 - Absolute difference
    speed_diff = pred_speed - actual_speed
    diff_max = float(np.abs(speed_diff).max())
    p3 = speed_diff.plot.imshow(
        ax=axes[1, 0], add_colorbar=True,
        cmap=cm.balance, vmin=-diff_max, vmax=diff_max,
    )
    axes[1, 0].set_title(r"Absolute Difference")
    axes[1, 0].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[1, 0].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # 2,2 - Relative difference
    speed_rel_diff = (pred_speed - actual_speed) / actual_speed
    rel_max = float(np.abs(speed_rel_diff).max())
    p4 = speed_rel_diff.plot.imshow(
        ax=axes[1, 1], add_colorbar=True,
        cmap=cm.balance, vmin=-rel_max, vmax=rel_max,
    )
    axes[1, 1].set_title(r"Relative Difference")
    axes[1, 1].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[1, 1].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # Save (drop tight_layout — it conflicts with the manually-added colorbar)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_side_by_side(ds, field: tuple[str, str], name, imgs_save_dir, save_name, latex=None, pad=0.02, diff=False,
                      log_first=False, log_second=False, cbmin=None, cbmax=None) -> Path:
    if latex is None:
        latex = field
    save_path = imgs_save_dir / save_name
    first_field, second_field = field
    first_latex, second_latex = latex

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(rf"{name} : {save_name}", fontsize=12)

    da_first = ds[first_field]
    da_second = ds[second_field]

    valid_first = da_first.notnull()
    valid_second = da_second.notnull()
    valid = valid_first | valid_second

    xv = da_first["x"].values[valid.any("y")]
    yv = da_first["y"].values[valid.any("x")]
    xmin, xmax = xv.min(), xv.max()
    ymin, ymax = yv.min(), yv.max()

    dx = (xmax - xmin) * pad
    dy = (ymax - ymin) * pad

    internal_cmap = cm.haline
    if diff:
        internal_cmap = cm.diff

    def _limits_kwargs(use_log):
        # With a norm, limits must go into the norm (xarray rejects norm + vmin/vmax combos)
        if use_log:
            return {"norm": SymLogNorm(linthresh=1, vmin=cbmin, vmax=cbmax)}
        return {"vmin": cbmin, "vmax": cbmax}

    # Plot first field
    p1 = da_first.plot.imshow(ax=axes[0], add_colorbar=True, cmap=internal_cmap, **_limits_kwargs(log_first))
    if p1.colorbar is not None:
        p1.colorbar.set_label(rf"{first_latex}")
    axes[0].set_title(rf"{first_latex}")
    axes[0].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[0].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # Plot second field
    p2 = da_second.plot.imshow(ax=axes[1], add_colorbar=True, cmap=internal_cmap, **_limits_kwargs(log_second))
    if p2.colorbar is not None:
        p2.colorbar.set_label(rf"{second_latex}")
    axes[1].set_title(rf"{second_latex}")
    axes[1].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[1].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def histogram(ds, field: tuple[str], name, imgs_save_dir, save_name, latex=None, pad=0.02, diff=False, log_x=False,
              cbmin=None, cbmax=None) -> Path:
    # cbmin / cbmax are accepted for a uniform signature; no colorbar here, so they are unused.
    if latex is None:
        latex = field
    save_path = imgs_save_dir / save_name
    data_field = field
    da = ds[data_field]

    # Create figure with 2x1 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(rf"{name} : Histogram of {latex}", fontsize=12)

    # Flatten data and remove NaN values
    values = da.values.flatten()
    values = values[~np.isnan(values)]

    # Common bin edges
    if log_x:
        values = values[values > 0]  # Remove non-positive values for log x-axis
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 50)
    else:
        bins = 50

    # Linear scale histogram (top)
    axes[0].hist(values, bins=bins, color='skyblue', edgecolor='black')
    axes[0].set_title(r"Linear Scale Histogram")
    axes[0].set_xlabel(rf"{latex}")
    axes[0].set_ylabel("Frequency")
    if log_x:
        axes[0].set_xscale('log')

    # Log scale histogram (bottom)
    axes[1].hist(values, bins=bins, color='salmon', edgecolor='black', log=True)
    axes[1].set_title(r"Log-Log Histogram" if log_x else r"Log Y Scale Histogram")
    axes[1].set_xlabel(rf"{latex}")
    axes[1].set_ylabel("Frequency (log)")
    if log_x:
        axes[1].set_xscale('log')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_stress_to_strain_rate(ds, field: tuple[str, str], name, imgs_save_dir, save_name, latex=None, pad=0.02, diff=False,
                               cbmin=None, cbmax=None) -> Path:
    # cbmin / cbmax are accepted for a uniform signature; no colorbar here, so they are unused.
    if latex is None:
        latex = field
    save_path = imgs_save_dir / save_name
    stress_field, strain_rate_field = field
    stress_latex, strain_rate_latex = latex

    stress = ds[stress_field]
    strain_rate = ds[strain_rate_field]

    valid = stress.notnull() & strain_rate.notnull()
    stress_vals = stress.values[valid]
    strain_rate_vals = strain_rate.values[valid]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(rf"{name}: {stress_latex} vs {strain_rate_latex}", fontsize=12)

    # --- Linear scale plot ---
    axes[0].scatter(strain_rate_vals, stress_vals, s=10, alpha=0.5)

    # Linear fit: stress = m * strain_rate + b
    lin_title = "Linear Scale"
    lin_mask = np.isfinite(strain_rate_vals) & np.isfinite(stress_vals)
    if lin_mask.sum() >= 2:
        x = strain_rate_vals[lin_mask]
        y = stress_vals[lin_mask]
        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        x_line = np.linspace(x.min(), x.max(), 100)
        axes[0].plot(x_line, m * x_line + b, color="red", lw=2,
                     label=rf"fit: y = {m:.3g}x + {b:.3g}")
        axes[0].legend()
        lin_title = rf"Linear Scale (R² = {r2:.3f})"

    axes[0].set_xlabel(rf"{strain_rate_latex}")
    axes[0].set_ylabel(rf"{stress_latex}")
    axes[0].set_title(lin_title)
    axes[0].grid(True, which="both", ls="--")

    # --- Log-Log scale plot ---
    axes[1].scatter(strain_rate_vals, stress_vals, s=10, alpha=0.5)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')

    # Power-law fit in log space: log(stress) = n * log(strain_rate) + log(A)
    log_title = "Log-Log Scale"
    log_mask = (
        np.isfinite(strain_rate_vals) & np.isfinite(stress_vals)
        & (strain_rate_vals > 0) & (stress_vals > 0)
    )
    if log_mask.sum() >= 2:
        log_x = np.log10(strain_rate_vals[log_mask])
        log_y = np.log10(stress_vals[log_mask])
        n, log_A = np.polyfit(log_x, log_y, 1)
        log_y_pred = n * log_x + log_A
        ss_res = np.sum((log_y - log_y_pred) ** 2)
        ss_tot = np.sum((log_y - log_y.mean()) ** 2)
        r2_log = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        x_fit = np.logspace(log_x.min(), log_x.max(), 100)
        y_fit = 10 ** log_A * x_fit ** n
        axes[1].plot(x_fit, y_fit, color="red", lw=2,
                     label=rf"fit: y = {10**log_A:.3g} · x^{{{n:.3g}}}")
        axes[1].legend()
        log_title = rf"Log-Log Scale (R² = {r2_log:.3f})"

    axes[1].set_xlabel(rf"{strain_rate_latex} (log scale)")
    axes[1].set_ylabel(rf"{stress_latex} (log scale)")
    axes[1].set_title(log_title)
    axes[1].grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

class Plot:
    def __init__(self, function, field, label, latex=None, cbmin = None, cbmax = None):
        self.field = field
        self.function = function
        self.label = label
        self.latex = field
        self.cbmin = cbmin
        self.cbmax = cbmax
        if latex is not None:
            self.latex = latex

    def __call__(self, ds, name, imgs_save_dir):
        # ds, field: tuple[str, str], name, imgs_save_dir, save_name
        return self.function(
            ds=ds, field=self.field, name=name, imgs_save_dir=imgs_save_dir,
            save_name=self.label, latex=self.latex, cbmin = self.cbmin, cbmax = self.cbmax,
        ), f"{name} - {self.label} - {self.field}"