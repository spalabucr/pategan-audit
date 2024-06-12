import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser

import torch
import tensorflow as tf


from pate_gans import PATE_GANS_AUDIT


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
df_path = config["data"]["df_path"]
save_eval_path = config["data"]["save_eval_path"]
# generation
epsilon = config["generation"]["epsilon"]
delta = config["generation"]["delta"]
teachers_ratio = config["generation"]["teachers_ratio"]
pgs = config["generation"]["pgs"]
pgs_kwargs = config["generation"]["pgs_kwargs"]


df = pd.read_pickle(df_path)
print(f"LOAD: {df_name} [shape {df.shape}] loaded\n")

dtypes = df.dtypes
n_records, n_features = df.shape
n_teachers = max(2, n_records // teachers_ratio)

# initialize df results
teachers_results = defaultdict()


# initialize and fit pate-gan
for pg_name in tqdm(pgs, desc="pg", leave=False):
    pgs_kwargs[pg_name]["epsilon"] = epsilon
    pgs_kwargs[pg_name]["delta"] = delta
    pgs_kwargs[pg_name]["num_teachers"] = n_teachers
    if pg_name in ["PG_UPDATED_AUDIT", "PG_BORAI_AUDIT"]:
        pgs_kwargs[pg_name]["X_shape"] = (n_records, n_features)
    if pg_name in ["PG_SMARTNOISE_AUDIT"]:
        pgs_kwargs[pg_name]["epsilon"] = 3

    pg_model = PATE_GANS_AUDIT[pg_name](**pgs_kwargs[pg_name])
    pg_model.fit(df)

    teachers_results[pg_name] = pg_model.teachers_dict
    with open(save_eval_path, 'wb') as f:
        pickle.dump(teachers_results, f, protocol=pickle.HIGHEST_PROTOCOL)
