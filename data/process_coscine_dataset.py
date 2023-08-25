import os
import pickle

from pathlib import Path
from tqdm import tqdm

name = "20230808_OPF__lv_grid_403634__mt_mvl_400V_20kV_t35040"

path = os.path.join("datasets", name, "Train")

new_path = os.path.join(name + "_just_opf")
Path(new_path).mkdir(parents=True, exist_ok=True)

pickle_filenames = os.listdir(path)

for filename in tqdm(pickle_filenames, desc="Saving just the OPF part"):
    if filename.endswith(".p"):
        with open(os.path.join(path, filename), "rb") as f:
            pfsolved, unsolved, opfsolved = pickle.load(f)

        with open(os.path.join(new_path, filename), "wb") as f:
            pickle.dump(opfsolved, f)
