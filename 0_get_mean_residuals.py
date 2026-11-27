import tomllib
import numpy as np
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser
from tabulate import tabulate
from tqdm import tqdm

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
    return np.nanmean(np.abs(x))

def median_abs(x):
    return np.nanmedian(np.abs(x))

def std_abs(x):
    return np.nanstd(np.abs(x))

parse = tomllib.loads(Path("./mapping.toml").read_text())
    

table = []
for count, subfolder in tqdm(list(enumerate(subfolders))):
    resid_file = subfolder / "field_and_residual_data.npz"
    if not resid_file.exists(): continue
    data = np.load(resid_file)
    inversion_name = subfolder.name
    shelf, raw_velocity, raw_thickness = inversion_name.split("__") 
    res_x, res_y = data["residual_x_fine"], data["residual_y_fine"]
    mean_res_x, mean_res_y = mean_abs(res_x)  , mean_abs(res_y)
    med_res_x , med_res_y  = median_abs(res_x), median_abs(res_y)
    std_res_x, str_res_y   = std_abs(res_x)   , std_abs(res_y) 
    table.append([count,shelf, raw_velocity, raw_thickness, mean_res_x, mean_res_y, med_res_x , med_res_y,std_res_x, str_res_y])

headers = [  "#", 
                        "Shelf", 
                        "Velocity\nProduct",
                        "Thickness\nProduct", 
                        "Mean-Abs\nX Residual", 
                        "Mean-Abs\nY Residual", 
                        "Median-Abs\nX Residual", 
                        "Median-Abs\nY Residual", 
                        "Std X Residual",
                        "Std Y Residual"
                    ]

table_str = tabulate(table,headers = headers,tablefmt="psql")
print(f"For: {folder.name}")
print(table_str)

headers = [string.replace("\n"," ") for string in headers]
pd.DataFrame(table, columns=headers).to_csv(save_location / f"residuals.csv", index=False)
(save_location / "Residuals Table.txt").write_text(table_str)
