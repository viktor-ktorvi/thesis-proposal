import os
import pickle

from pathlib import Path
from tqdm import tqdm

name = "20230823_OPF__path_lv_sample2__mt_mvl_400V_20kV_t350"

path = os.path.join("datasets", name)

new_path = os.path.join("datasets", name + "_just_opf")
Path(new_path).mkdir(parents=True, exist_ok=True)

pickle_filenames = os.listdir(path)

for filename in tqdm(pickle_filenames, desc="Saving just the OPF part"):
    if filename.endswith(".p"):
        with open(os.path.join(path, filename), "rb") as f:
            pfsolved, unsolved, opfsolved = pickle.load(f)

        with open(os.path.join(new_path, filename), "wb") as f:
            pickle.dump(opfsolved, f)
