from row_plotting_helpers import plot_field, plot_log_field, Plot
from pathlib import Path
import xarray as xr
import rioxarray as rxr
from argparse import ArgumentParser
import tomllib
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--folder", required = True)
parser.add_argument("--config-name", default = "diff.toml")
parser.add_argument("--output-dir", default = "/oak/stanford/groups/cyaolai/AashrayChegu/DIFFICE_DIFFS")
args = parser.parse_args()

folder = Path(args.folder)
config_path = folder / args.config_name
output_dir = Path(args.output_dir) / folder.name
output_dir.mkdir(exist_ok = True)

config = tomllib.loads(config_path.read_text())
comparisons = config["comparisons"]


linear_plots = {
    "Delta tau_eff":    Plot(function=plot_field,        
                             field="diff",      
                             cbmin = None, 
                             cbmax = None, 
                             label="Delta tau_eff",                
                             latex=r"$\tau_{eff,2} = \tau_{eff,1}$"       
                            ),
    "tau_Eff from deltas": Plot(function=plot_field,        
                             field="foil",      
                             cbmin = None, 
                             cbmax = None, 
                             label="tau_eff from deltas",                
                             latex=r"$2(\mu_2 - \mu_1)*(\dot{\varepsilon_2} - \dot{\varepsilon_1})$"       
                            ),
    "mu2 times edot2": Plot(function=plot_field,        
                             field="f",      
                             cbmin = None, 
                             cbmax = None, 
                             label="mu2 times edot2",                
                             latex=r"$2(\mu_2 * \dot{\varepsilon_2})$"       
                            ),
    "-mu2 times edot1": Plot(function=plot_field,        
                             field="o",      
                             cbmin = None, 
                             cbmax = None, 
                             label="-mu2 times edot1",                
                             latex=r"$-2(\mu_2 * \dot{\varepsilon_1})$"       
                            ),
    "-mu1 times edot2": Plot(function=plot_field,        
                             field="i",      
                             cbmin = None, 
                             cbmax = None, 
                             label="-mu1 times edot2",                
                             latex=r"$-2(\mu_1 * \dot{\varepsilon_2})$"       
                            ),
    "mu1 times edot1": Plot(function=plot_field,        
                             field="l",      
                             cbmin = None, 
                             cbmax = None, 
                             label="mu1 times edot1",                
                             latex=r"$(\mu_1 * \dot{\varepsilon_1})$"       
                            ),
}
log_plots = {
    "Log Delta tau_eff":    Plot(function=plot_log_field,        
                             field="diff",      
                             cbmin = None, 
                             cbmax = None, 
                             label="Delta tau_eff",                
                             latex=r"$\tau_{eff,2} = \tau_{eff,1}$"       
                            ),
    "Log tau_Eff from deltas": Plot(function=plot_log_field,        
                             field="foil",      
                             cbmin = None, 
                             cbmax = None, 
                             label="tau_eff from deltas",                
                             latex=r"$2(\mu_2 - \mu_1)*(\dot{\varepsilon_2} - \dot{\varepsilon_1})$"       
                            ),
    "Log mu2 times edot2": Plot(function=plot_log_field,        
                             field="f",      
                             cbmin = None, 
                             cbmax = None, 
                             label="mu2 times edot2",                
                             latex=r"$2(\mu_2 * \dot{\varepsilon_2})$"       
                            ),
    "Log -mu2 times edot1": Plot(function=plot_log_field,        
                             field="o",      
                             cbmin = None, 
                             cbmax = None, 
                             label="-mu2 times edot1",                
                             latex=r"$-2(\mu_2 * \dot{\varepsilon_1})$"       
                            ),
    "Log -mu1 times edot2": Plot(function=plot_log_field,        
                             field="i",      
                             cbmin = None, 
                             cbmax = None, 
                             label="-mu1 times edot2",                
                             latex=r"$-2(\mu_1 * \dot{\varepsilon_2})$"       
                            ),
    "Log mu1 times edot1": Plot(function=plot_log_field,        
                             field="l",      
                             cbmin = None, 
                             cbmax = None, 
                             label="mu1 times edot1",                
                             latex=r"$(\mu_1 * \dot{\varepsilon_1})$"       
                            ),
}


def get_values(t_2: xr.Dataset, t_1: xr.Dataset) -> xr.Dataset:
    mu_2,mu_1 = t_2["mu"], t_1["mu"]
    edot_eff_2, edot_eff_1 = t_2["e_eff"], t_1["e_eff"] 

    d_edot = edot_eff_2 - edot_eff_1
    d_mu = mu_2 - mu_1

    return {
    "diff": t_2["tau_eff"] - t_1["tau_eff"],
    "foil":  2* d_edot * d_mu,
    "f":     2 * mu_2 * edot_eff_2,
    "o":    -2 * mu_2 * edot_eff_1,
    "i":    -2 * mu_1 * edot_eff_2,
    "l":     2 * mu_2 * edot_eff_2
}
    

def do_plots(plots, plot_type):
    for label, zarrs in tqdm(comparisons.items(), smoothing=0, position=1,):

        first_path = folder / zarrs["minuend"] / "evaluations.zarr"
        second_path = folder / zarrs["subtrahend"] / "evaluations.zarr"

        first_zarr = xr.open_zarr(first_path, consolidated=False)
        second_zarr = xr.open_zarr(second_path, consolidated=False)

        first_aligned, second_aligned = xr.align(first_zarr, second_zarr, join="inner")

        n_rows, n_cols = len(plots), 1

        fig, ax = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 8 * n_rows),
            squeeze=False,           # always 2D, even for 1 row / 1 column
            layout="constrained",    # keeps the per-row colorbars tidy
        )

        values = get_values(first_zarr, second_zarr)

        for row,(name, plot) in tqdm(enumerate(plots.items())):
            plot({"values": values}, ax[row])

        parent_dir = output_dir / label
        save_path =  parent_dir / f"{plot_type} Stress Difference Components.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

do_plots(linear_plots,"Linear")
do_plots(log_plots, plot_type="Log")