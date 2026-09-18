from pathlib import Path
import tomllib
from argparse import ArgumentParser
from itertools import product
from pprint import pprint
import xarray as xr
from plotting_helpers import Plot
from triangle_plotting_helpers import plot_field, plot_log_field
from tqdm import tqdm

plot_vars = [
    Plot(
        function=plot_field,
        field="mu",
        label="Viscosity",
    ),
    Plot(
        function=plot_log_field,
        field="mu",
        label="Log Viscosity",
    ),
    Plot(
        function=plot_field,
        field="e_eff",
        label="Effective Strain Rate",
    ),
    Plot(
        function=plot_log_field,
        field="e_eff",
        label="Log Effective Strain Rate",
    ),
    Plot(
        function=plot_field,
        field="e_xx",
        label="Strain Rate XX Component",
    ),
    Plot(
        function=plot_field,
        field="e_xy",
        label="Strain Rate XY Component",
    ),
    Plot(
        function=plot_field,
        field="e_yy",
        label="Strain YY Component",
    ),
    Plot(
        function=plot_field,
        field="tau_eff",
        label="Effective Stress",
    ),
    Plot(
        function=plot_field,
        field="tau_xx",
        label="Stress XX Component",
    ),
    Plot(
        function=plot_field,
        field="tau_xy",
        label="Stress XY Component",
    ),
    Plot(
        function=plot_field,
        field="tau_yy",
        label="Stress YY Component",
    ),
    Plot(
        function=plot_field,
        field="tau_1",
        label="Stress Primary Component",
    ),
    Plot(
        function=plot_field,
        field="tau_2",
        label="Stress Secondary Component",
    ),
    Plot(
        function=plot_field,
        field="R_xx",
        label="Extensional Stress (R_xx)",
    ),
]

parser = ArgumentParser()
parser.add_argument("--folder", required = True)
parser.add_argument("--comparison-folder", default = "/oak/stanford/groups/cyaolai/AashrayChegu/DIFFICE_COMPARISONS")
parser.add_argument("--mapping-file", default = "diff.toml")
parser.add_argument("--comparisons-file", default = "comparisons.toml")


args = parser.parse_args()

folder = Path(args.folder)
comparison_folder = Path(args.comparison_folder)
mapping_file = folder / args.mapping_file
comparisons_file = folder / args.comparisons_file

label_to_diffs = tomllib.loads(mapping_file.read_text())["comparisons"]
comparisons = tomllib.loads(comparisons_file.read_text())["comparisons"]

mappings = {}
for label, diffs in label_to_diffs.items():
    mappings[(diffs["minuend"],diffs["subtrahend"])] = label

filtered_comparisons = {}
for label, comparison_list in comparisons.items():
    all_comparisons = list(product(enumerate(comparison_list["list"]),repeat = 2))
    filtered_comparisons[label] = []
    for comparison in all_comparisons:
        loc1, inv1 = comparison[0]
        loc2, inv2 = comparison[1]
        diff = (inv1,inv2)
        if (name := mappings.get(diff, False)):
            filtered_comparisons[label].append(dict(minuend = inv1, subtrahend = inv2, name = name, loc = (loc1,loc2)))

filtered_comparisons = {key:value for key,value in filtered_comparisons.items() if len(value) > 0}

output_folder = comparison_folder / folder.name
output_folder.mkdir(exist_ok = True)

for label, plot_list in (pbar1:=tqdm(list(filtered_comparisons.items()),smoothing = 0, position = 0)):
    pbar1.set_description(f"{label}")
    save_folder = output_folder / label
    save_folder.mkdir(exist_ok = True)

    for plot in tqdm(plot_list,smoothing = 0, position = 1,desc = "Loading Zarrs"):
        plot["zarr"] = xr.open_zarr( folder / plot["name"] / f"diff.zarr", consolidated = False)

    for plot in (pbar2:=tqdm(plot_vars, smoothing = 0, position = 1, desc = "Building Plots")):
        save_path, description = plot(plot_list, name = label, imgs_save_dir = save_folder)
        pbar2.set_description(description)
        
