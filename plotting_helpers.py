from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def plot_field(ds, field: tuple[str], name, imgs_save_dir, save_name, pad=0.02) -> Path:
    save_path = imgs_save_dir / save_name
    data_field = field
    da = ds[data_field]
    valid = da.notnull()
    xv = da["x"].values[valid.any("y")]
    yv = da["y"].values[valid.any("x")]
    xmin, xmax = xv.min(), xv.max()
    ymin, ymax = yv.min(), yv.max()

    p = da.plot.imshow(aspect=1, size=10, add_colorbar=True)

    dx = (xmax - xmin) * pad
    dy = (ymax - ymin) * pad

    p.axes.set_title(f"{name} : {field}", fontsize=10)
    p.axes.set_xlim(float(xmin) - dx, float(xmax) + dx)
    p.axes.set_ylim(float(ymin) - dy, float(ymax) + dy)

    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def plot_log_field(ds, field: tuple[str], name, imgs_save_dir, save_name, pad=0.02) -> Path:
    save_path = imgs_save_dir / save_name
    data_field = field
    da = ds[data_field]
    valid = da.notnull()
    xv = da["x"].values[valid.any("y")]
    yv = da["y"].values[valid.any("x")]
    xmin, xmax = xv.min(), xv.max()
    ymin, ymax = yv.min(), yv.max()

    p = da.plot.imshow(aspect=1, size=10, add_colorbar=True, norm=LogNorm())

    dx = (xmax - xmin) * pad
    dy = (ymax - ymin) * pad

    p.axes.set_title(f"{name} : {field} (log scale)", fontsize=10)
    p.axes.set_xlim(float(xmin) - dx, float(xmax) + dx)
    p.axes.set_ylim(float(ymin) - dy, float(ymax) + dy)

    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def compare_fields(ds, field: tuple[str, str], name, imgs_save_dir, save_name, pad=0.02) -> Path:
    save_path = imgs_save_dir / save_name
    predicted_field, actual_field = field

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{name} : {save_name}", fontsize=12)

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

    # Shared color limits for the top two plots
    shared_min = float(min(da_pred.min(), da_actual.min()))
    shared_max = float(max(da_pred.max(), da_actual.max()))

    # 1,1 - Predicted field (no individual colorbar)
    p1 = da_pred.plot.imshow(
        ax=axes[0, 0], add_colorbar=False, vmin=shared_min, vmax=shared_max
    )
    axes[0, 0].set_title(f"Predicted: {predicted_field}")
    axes[0, 0].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[0, 0].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # 1,2 - Actual field (no individual colorbar)
    p2 = da_actual.plot.imshow(
        ax=axes[0, 1], add_colorbar=False, vmin=shared_min, vmax=shared_max
    )
    axes[0, 1].set_title(f"Actual: {actual_field}")
    axes[0, 1].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[0, 1].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # Single shared colorbar for the top row
    fig.colorbar(p1, ax=axes[0, :], shrink=0.8, label=predicted_field)

    # 2,1 - Difference (predicted - actual)
    diff = da_pred - da_actual
    diff_max = float(np.abs(diff).max())
    p3 = diff.plot.imshow(ax=axes[1, 0], add_colorbar=True, cmap = "seismic", vmin = -diff_max, vmax = diff_max)
    axes[1, 0].set_title(f"Difference: {predicted_field} - {actual_field}")
    axes[1, 0].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[1, 0].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # 2,2 - Relative difference ((predicted - actual)/actual)
    rel_diff = (da_pred - da_actual) / da_actual
    rel_diff_max = float(np.abs(rel_diff).max())
    p4 = rel_diff.plot.imshow(ax=axes[1, 1], add_colorbar=True, cmap = "seismic", vmin = -rel_diff_max, vmax = rel_diff_max, robust = True)
    axes[1, 1].set_title(f"Relative Difference: ({predicted_field} - {actual_field})/{actual_field}")
    axes[1, 1].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[1, 1].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # Save (drop tight_layout — it conflicts with the manually-added colorbar)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path

def compare_speed(ds, field: tuple[str, str, str, str], name, imgs_save_dir, save_name, pad=0.02) -> Path:
    save_path = imgs_save_dir / save_name
    predicted_x_field, predicted_y_field, actual_x_field, actual_y_field = field

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{name} : {save_name}", fontsize=12)

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

    # Shared color limits for the top two plots
    shared_min = float(min(pred_speed.min(), actual_speed.min()))
    shared_max = float(max(pred_speed.max(), actual_speed.max()))

    # 1,1 - Predicted speed (no individual colorbar)
    p1 = pred_speed.plot.imshow(
        ax=axes[0, 0], add_colorbar=False, vmin=shared_min, vmax=shared_max
    )
    axes[0, 0].set_title(f"Predicted Speed")
    axes[0, 0].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[0, 0].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # 1,2 - Actual speed (no individual colorbar)
    p2 = actual_speed.plot.imshow(
        ax=axes[0, 1], add_colorbar=False, vmin=shared_min, vmax=shared_max
    )
    axes[0, 1].set_title(f"Actual Speed")
    axes[0, 1].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[0, 1].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # Single shared colorbar for the top row
    fig.colorbar(p1, ax=axes[0, :], shrink=0.8, label="Speed")

    # 2,1 - Absolute difference
    speed_diff = pred_speed - actual_speed
    diff_max = float(np.abs(speed_diff).max())
    p3 = speed_diff.plot.imshow(
        ax=axes[1, 0], add_colorbar=True,
        cmap="seismic", vmin=-diff_max, vmax=diff_max,
    )
    axes[1, 0].set_title(f"Absolute Difference")
    axes[1, 0].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[1, 0].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # 2,2 - Relative difference
    speed_rel_diff = (pred_speed - actual_speed) / actual_speed
    rel_max = float(np.abs(speed_rel_diff).max())
    p4 = speed_rel_diff.plot.imshow(
        ax=axes[1, 1], add_colorbar=True,
        cmap="seismic", vmin=-rel_max, vmax=rel_max,
    )
    axes[1, 1].set_title(f"Relative Difference")
    axes[1, 1].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[1, 1].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # Save (drop tight_layout — it conflicts with the manually-added colorbar)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path

def plot_side_by_side(ds, field: tuple[str, str], name, imgs_save_dir, save_name, pad=0.02, log_first = False, log_second = False, ) -> Path:
    save_path = imgs_save_dir / save_name
    first_field, second_field = field

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"{name} : {save_name}", fontsize=12)

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

    # Plot first field
    p1 = da_first.plot.imshow(ax=axes[0],  add_colorbar=True, norm=(LogNorm() if log_first else None))
    axes[0].set_title(first_field)
    axes[0].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[0].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # Plot second field
    p2 = da_second.plot.imshow(ax=axes[1],  add_colorbar=True, norm=(LogNorm() if log_second else None))
    axes[1].set_title(second_field)
    axes[1].set_xlim(float(xmin) - dx, float(xmax) + dx)
    axes[1].set_ylim(float(ymin) - dy, float(ymax) + dy)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def histogram(ds, field: tuple[str], name, imgs_save_dir, save_name, pad=0.02, log_x=True) -> Path:
    save_path = imgs_save_dir / save_name
    data_field = field
    da = ds[data_field]

    # Create figure with 2x1 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(f"{name} : Histogram of {data_field}", fontsize=12)

    # Flatten data and remove NaN values
    values = da.values.flatten()
    values = values[~np.isnan(values)]
    values = values[values > 0]  # Remove non-positive values for log x-axis

    # Common bin edges
    if log_x:
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 50)
    else:
        bins = 50

    # Linear scale histogram (top)
    axes[0].hist(values, bins=bins, color='skyblue', edgecolor='black')
    axes[0].set_title(f"Linear Scale Histogram")
    axes[0].set_xlabel(data_field)
    axes[0].set_ylabel("Frequency")
    if log_x:
        axes[0].set_xscale('log')

    # Log scale histogram (bottom)
    axes[1].hist(values, bins=bins, color='salmon', edgecolor='black', log=True)
    axes[1].set_title(f"Log-Log Histogram" if log_x else "Log Y Scale Histogram")
    axes[1].set_xlabel(data_field)
    axes[1].set_ylabel("Frequency (log)")
    if log_x:
        axes[1].set_xscale('log')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def plot_stress_to_strain_rate(ds, field: tuple[str, str], name, imgs_save_dir, save_name, pad=0.02) -> Path:
    save_path = imgs_save_dir / save_name
    stress_field, strain_rate_field = field

    stress = ds[stress_field]
    strain_rate = ds[strain_rate_field]

    valid = stress.notnull() & strain_rate.notnull()
    stress_vals = stress.values[valid]
    strain_rate_vals = strain_rate.values[valid]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(f"{name}: {stress_field} vs {strain_rate_field}", fontsize=12)

    axes[0].scatter(strain_rate_vals, stress_vals, s=10, alpha=0.5)
    axes[0].set_xlabel(strain_rate_field)
    axes[0].set_ylabel(stress_field)
    axes[0].set_title("Linear Scale")
    axes[0].grid(True, which="both", ls="--")

    axes[1].scatter(strain_rate_vals, stress_vals, s=10, alpha=0.5)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(f"{strain_rate_field} (log scale)")
    axes[1].set_ylabel(f"{stress_field} (log scale)")
    axes[1].set_title("Log-Log Scale")
    axes[1].grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

class Plot:
    def __init__(self, function, field, label):
        self.field = field
        self.function = function
        self.label = label

    def __call__(self, ds, name, imgs_save_dir):
        return self.function(ds, self.field, name, imgs_save_dir, self.label), f"{name} - {self.label} - {self.field}"
