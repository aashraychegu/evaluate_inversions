# pyrefly: ignore [missing-import]
import tomllib
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser
from tabulate import tabulate
from tqdm import tqdm

def dashpad(inp):
    if len(inp) == 1:
        return (inp[0], "ALL-ALL") 

parser = ArgumentParser()
parser.add_argument("--folder",required=True)

args = parser.parse_args()
folder = Path(args.folder)
folder = folder.resolve().absolute()
save_location = folder

if not folder.exists():
    folder = Path("")
subfolders = sorted(list(folder.iterdir()))

def mean_abs(x):
    return np.mean(np.abs(x))

def median_abs(x):
    return np.median(np.abs(x))

parse = tomllib.loads(Path("./mapping.toml").read_text())
    

table = []
for count, subfolder in tqdm(list(enumerate(subfolders))):
    resid_file = subfolder / "field_and_residual_data.npz"
    if not resid_file.exists(): continue
    data = np.load(resid_file)
    inversion_name = subfolder.name
    shelf, raw_velocity, raw_thickness = inversion_name.split("__") 
    velocity_product, velocity_start, velocity_end = parse[raw_velocity]
    thickness_product, thickness_start, thickness_end = parse[raw_thickness]
    res_x, res_y = data["residual_x"], data["residual_y"]
    mean_res_x, mean_res_y = mean_abs(res_x)  , mean_abs(res_y)
    med_res_x , med_res_y  = median_abs(res_x), median_abs(res_y)
    table.append([count,shelf, velocity_product, velocity_start, velocity_end,thickness_product, thickness_start, thickness_end,mean_res_x, mean_res_y, med_res_x , med_res_y])

headers = [  "#", 
                        "Shelf", 
                        "Velocity\nProduct",
                        "Velocity\nStart", 
                        "Velocity\nEnd", 
                        "Thickness\nProduct", 
                        "Thickness\nStart", 
                        "Thickness\nEnd", 
                        "Mean-Abs\nX Residual", 
                        "Mean-Abs\nY Residual", 
                        "Median-Abs\nX Residual", 
                        "Median-Abs\nY Residual", 
                    ]

table_str = tabulate(table,headers = headers,tablefmt="psql")
print(f"For: {folder.name}")
print(table_str)

headers = [string.replace("\n"," ") for string in headers]
pd.DataFrame(table, columns=headers).to_csv(save_location / f"residuals.csv", index=False)
(save_location / "Residuals Table.txt").write_text(table_str)
