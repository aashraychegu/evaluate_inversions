from pathlib import Path
import xarray as xr
import rioxarray as rxr
from argparse import ArgumentParser
import tomllib
import shutil
import numpy as np
from tqdm import tqdm

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

for label, zarrs in tqdm(comparisons.items(), smoothing=0, position=1,):
    first_path = folder / zarrs["minuend"] / "evaluations.zarr"
    second_path = folder / zarrs["subtrahend"] / "evaluations.zarr"

    first_zarr = xr.open_zarr(first_path, consolidated=False)
    second_zarr = xr.open_zarr(second_path, consolidated=False)

    first_aligned, second_aligned = xr.align(first_zarr, second_zarr, join="inner")

    data_vars = {}
    for data_var in tqdm(list(first_zarr.data_vars), smoothing=0, position=0,):
        if data_var not in second_zarr.data_vars:
            continue

        minuend = first_aligned[data_var]
        subtrahend = second_aligned[data_var]

        # tqdm.write(f"{data_var}; {minuend.dtype} {subtrahend.dtype}")
        if minuend.dtype == bool or subtrahend.dtype == bool:
            data_vars[data_var] = minuend | subtrahend
        else:
            diff = minuend - subtrahend
            data_vars[data_var] = diff

        if data_var == "mu":
            data_vars["log(delta mu)"] = np.log(minuend) - np.log(subtrahend)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=first_aligned.coords,
    )

    # Preserve CRS if present on the source datasets
    ds = ds.rio.write_crs("EPSG:3031")

    parent_dir = output_dir / label
    parent_dir.mkdir(exist_ok = True)

    out_path = parent_dir / f"diff.zarr"
    if out_path.exists():
        shutil.rmtree(out_path)
    ds.to_zarr(out_path, mode="w", consolidated=False)

print(shutil.copy(config_path, output_dir))

