import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import torch
import tensorflow as tf


from pate_gans import PATE_GANS_AUDIT
from utils import fix_dtypes


seed = 13
np.random.seed(seed)
tf.random.set_seed(seed)
torch.manual_seed(seed)


parser = ArgumentParser()
parser.add_argument('--config', '-C', type=str, default='configs/utility/credit.json', help='Config relative path')
args = parser.parse_args()


# load config
with open(Path(args.config)) as f:
    config = json.load(f)


# data
df_name = config["data"]["df_name"]
save_data_path = config["data"]["save_data_path"]
# generation
epsilon = config["generation"]["epsilon"]
delta = config["generation"]["delta"]
pgs = config["generation"]["pgs"]
if pgs == "all":
    pgs = PATE_GANS_AUDIT.keys()
pgs_kwargs = config["generation"]["pgs_kwargs"]
pgs_fit_kwargs = {pg_name: {} for pg_name in pgs}
n_pgs = config["generation"]["n_pgs"]

dtypes = {"A": "int", "B": "int", "C": "int"}
df_out = pd.DataFrame({"A": [0, 0, 0, 0],
                       "B": [0, 0, 0, 0],
                       "C": [0, 0, 0, 0]}).astype(dtypes)
df_in = pd.DataFrame({"A": [1, 0, 0, 0, 0],
                      "B": [1, 0, 0, 0, 0],
                      "C": [1, 0, 0, 0, 0]}).astype(dtypes)

n_records_out, n_features_out = df_out.shape
n_records_in, n_features_in = df_in.shape
n_teachers = 2

# initialize and fit pate-gan
for pg_name in tqdm(pgs, desc="pg", leave=False):
    pgs_kwargs[pg_name]["epsilon"] = epsilon
    pgs_kwargs[pg_name]["delta"] = delta if pg_name != "PG_ORIGINAL_AUDIT" else int(-np.log10(delta))
    pgs_kwargs[pg_name]["num_teachers"] = n_teachers
    if pg_name in ["PG_SMARTNOISE_AUDIT"]:
        pgs_fit_kwargs[pg_name]["skip_processing"] = True

    if pg_name in ["PG_ORIGINAL_AUDIT", "PG_UPDATED_AUDIT", "PG_TURING_AUDIT", "PG_BORAI_AUDIT"]:
        pgs_kwargs[pg_name]["X_shape"] = (n_records_out, n_features_out)

    for i in tqdm(range(n_pgs), desc="out", leave=False):
        pg_model = PATE_GANS_AUDIT[pg_name](**pgs_kwargs[pg_name])
        pg_model.fit(df_out, **pgs_fit_kwargs[pg_name])
        synth_df_out = pg_model.generate(n_records_out)
        synth_df_out = fix_dtypes(synth_df_out, dtypes)
        synth_df_out.to_pickle(f"{save_data_path}/{pg_name}/out_{i}.pkl.gz", compression="gzip")

    if pg_name in ["PG_ORIGINAL_AUDIT", "PG_UPDATED_AUDIT", "PG_TURING_AUDIT", "PG_BORAI_AUDIT"]:
        pgs_kwargs[pg_name]["X_shape"] = (n_records_in, n_features_in)

    for i in tqdm(range(n_pgs), desc="in", leave=False):
        pg_model = PATE_GANS_AUDIT[pg_name](**pgs_kwargs[pg_name])
        pg_model.fit(df_in, **pgs_fit_kwargs[pg_name])
        synth_df_in = pg_model.generate(n_records_in)
        synth_df_in = fix_dtypes(synth_df_in, dtypes)
        synth_df_in.to_pickle(f"{save_data_path}/{pg_name}/in_{i}.pkl.gz", compression="gzip")
