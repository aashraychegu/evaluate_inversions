from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
import cmocean
from cmocean import cm


def get_plot_info(ds, field):
    plot_info = []
    x_vals = []
    y_vals = []
    for i in ds:
        x_vals.append(i["loc"][0])
        y_vals.append(i["loc"][1])
        plot_info.append(dict(
            title=f"{i['minuend']}\n-\n{i['subtrahend']}",
            loc=i["loc"],
            data=i["zarr"][field],
        ))
    x_max = max(x_vals)
    y_max = max(y_vals)
    return plot_info, x_max, y_max


def draw_triangle_plot(plot_info, x_max, y_max, name, save_path, pad,
                     norm=None, cmap=cm.balance):

    ncols = x_max
    nrows = y_max + 1

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 5 * nrows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.set_axis_off()

    mappable = None
    for info in plot_info:
        x, y = info["loc"]
        ax = axes[y-1, x-1]
        ax.set_axis_on()

        da = info["data"]
        im = da.plot.imshow(
            ax=ax,
            norm=norm,
            cmap=cmap,
            add_colorbar=False,
            add_labels=False,
        )
        mappable = im  

        ax.set_title(info["title"], fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    if mappable is not None:
        cbar = fig.colorbar(
            mappable, ax=axes.ravel().tolist(),
            fraction=0.025, pad=pad,
        )
        cbar.set_label(name)

    fig.suptitle(name, fontsize=14)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_field(ds, field: tuple[str], name, imgs_save_dir, save_name, latex = None, pad=0.02) -> Path:
    save_path = imgs_save_dir / save_name
    plot_info, x_max, y_max = get_plot_info(ds, field)
    # tqdm.write("  maxes:", x_max, y_max)
    all_vals = np.concatenate([info["data"].values.ravel() for info in plot_info])
    vmax = np.nanmax(np.abs(all_vals))
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    return draw_triangle_plot(
        plot_info, x_max, y_max, name, save_path, pad,
        norm=norm, cmap=cm.balance,
    )


def plot_log_field(ds, field: tuple[str], name, imgs_save_dir, save_name, latex = None, pad=0.02) -> Path:
    save_path = imgs_save_dir / save_name
    plot_info, x_max, y_max = get_plot_info(ds, field)

    all_vals = np.concatenate([info["data"].values.ravel() for info in plot_info])
    vmax = np.nanmax(np.abs(all_vals))
    finite = all_vals[np.isfinite(all_vals) & (all_vals != 0)]
    linthresh = np.nanpercentile(np.abs(finite), 5) if finite.size else 1e-6
    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)

    return draw_triangle_plot(
        plot_info, x_max, y_max, name, save_path, pad,
        norm=norm, cmap=cm.balance,
    )