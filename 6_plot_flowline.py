from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pyproj
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rxr
from functools import partial
from scipy.interpolate import RegularGridInterpolator

parser = ArgumentParser()
parser.add_argument("--folder", required = True)
folder = Path(parser.parse_args().folder)

subfolders = sorted(list(folder.glob("*/")))


def find_central_flowline(ds):
    h = ds["regrid_h"]
    u = ds["regrid_u"]
    v = ds["regrid_v"]
    tau_eff = ds["tau_eff"]
    e_eff = ds["e_eff"]

    stacked = h.stack(cell=("y", "x"))
    imax = stacked.argmax("cell", skipna=True)     
    cell = stacked["cell"][imax]
    max_x = float(cell["x"].item())
    max_y = float(cell["y"].item())
    max_h = float(stacked[imax].item())
    seed = (max_x, max_y, max_h, 0, None, None)
    points = []
    points.append(seed)

    dl = 10
    while True:
        x,y,h,distance, tau_eff_back,e_eff_back = points[-1]
                    
        u_here = u.interp(x = x, y = y, method = "linear").item()
        v_here = v.interp(x = x, y = y, method = "linear").item()
        tau_eff_here = tau_eff.interp(x = x, y = y, method = "linear").item()
        e_eff_here = e_eff.interp(x = x, y = y, method = "linear").item()

        if tau_eff_back is None or e_eff_back is None:
            points[-1] = (x, y, h, distance + dl, tau_eff_here, e_eff_here)
        
        if not (np.isfinite(u_here) and np.isfinite(v_here)):
            break
        speed = np.hypot(u_here, v_here)
        if speed == 0:
            break
        
        dx = dl * u_here / speed
        dy = dl * v_here / speed
        x_prime, y_prime = x+dx, y+dy
        
        points.append((x_prime, y_prime,h, distance + dl, tau_eff_here, e_eff_here))
    return np.array(points)



for subfolder in (pbar := tqdm(subfolders, smoothing = 0, position = 0, dynamic_ncols= True)):
    pbar.set_description(f"Plotting: {subfolder.name}")

    ds = xr.open_zarr(subfolder / f"evaluations.zarr", consolidated = False)
    img_dir = subfolder / "eval_images"
    img_dir.mkdir(exist_ok = True)

    line = find_central_flowline(ds) # (N points, 6); x, y, h, distance, tau_eff, e_eff; N can be one
    
    if line.ndim != 2 or line.shape[0] < 2:
        tqdm.write(f"{subfolder.name}: flowline too short {line.shape}, skipping plot")
        continue

    x_, y_, h_, dist_, tau_, e_ = (line[:, i] for i in range(6))
    dist_km = dist_ / 1000.0

    fig = plt.figure(figsize=(30, 21))
    gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[3, 3, 3, 15, 15])

    # --- subplot 1: h vs distance ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dist_km, h_, "-", color="tab:blue")
    ax1.set_ylabel("h")
    ax1.set_xlabel("distance along flowline (km)")
    ax1.set_title(f"{subfolder.name}: thickness along central flowline")
    ax1.grid(True, ls=":", alpha=0.5)

    # --- subplot 2: tau_eff vs distance ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dist_km, tau_, "-", color="tab:red")
    ax2.set_ylabel(r"$\tau_{eff}$")
    ax2.set_xlabel("distance along flowline (km)")
    ax2.grid(True, ls=":", alpha=0.5)

    # --- subplot 3: e_eff vs distance ---
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(dist_km, e_, "-", color="tab:green")
    ax3.set_ylabel(r"$\dot\varepsilon_{eff}$")
    ax3.set_xlabel("distance along flowline (km)")
    ax3.grid(True, ls=":", alpha=0.5)

    # --- subplot 4: tau_eff vs e_eff scatter + line ---
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(e_, tau_, "-", color="gray", lw=1, alpha=0.6, zorder=1)
    sc4 = ax4.scatter(e_, tau_, c=dist_km, cmap="viridis", s=25, zorder=2)
    ax4.set_xlabel(r"effective strain rate  $\dot\varepsilon_{eff}$")
    ax4.set_ylabel(r"effective stress  $\tau_{eff}$")
    ax4.set_title("stress vs strain rate (linear)")
    ax4.grid(True, ls=":", alpha=0.5)
    fig.colorbar(sc4, ax=ax4, label="distance along flowline (km)")

    # --- subplot 5: log-log tau_eff vs e_eff scatter + line ---
    ax5 = fig.add_subplot(gs[4])
    valid = np.isfinite(e_) & np.isfinite(tau_) & (e_ > 0) & (tau_ > 0)
    ax5.plot(e_[valid], tau_[valid], "-", color="gray", lw=1, alpha=0.6, zorder=1)
    sc5 = ax5.scatter(e_[valid], tau_[valid], c=dist_km[valid],
                      cmap="viridis", s=25, zorder=2)
    ax5.set_xscale("log")
    ax5.set_yscale("log")
    ax5.set_xlabel(r"effective strain rate  $\dot\varepsilon_{eff}$")
    ax5.set_ylabel(r"effective stress  $\tau_{eff}$")
    ax5.set_title("stress vs strain rate (log-log)")
    ax5.grid(True, which="both", ls=":", alpha=0.5)
    fig.colorbar(sc5, ax=ax5, label="distance along flowline (km)")

    fig.tight_layout()
    fig.savefig(img_dir / "flowline_diagnostics.png", dpi=150)
    plt.close(fig)

    tqdm.write(f"{subfolder.name} : {line.shape}")
