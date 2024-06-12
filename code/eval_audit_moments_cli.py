import json
import pickle
import numpy as np
import pandas as pd
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
test_df_path = config["data"]["test_df_path"]
save_eval_path = config["data"]["save_eval_path"]
# generation
delta = config["generation"]["delta"]
teachers_ratio = config["generation"]["teachers_ratio"]

df = pd.read_pickle(df_path)
test_df = pd.read_pickle(test_df_path)
print(f"LOAD: {df_name} [shape {df.shape}] loaded\n")

dtypes = df.dtypes
n_records, n_features = df.shape
n_teachers = max(2, n_records // teachers_ratio)

# initialize df results
moments_results = defaultdict()
epsilons_results = defaultdict()


# initialize and fit pate-gan
pg_name = config["generation"]["pgs"]
pg_kwargs = config["generation"]["pg_kwargs"][pg_name]
pg_kwargs["epsilon"] = np.inf
pg_kwargs["delta"] = delta
pg_kwargs["num_teachers"] = n_teachers

pg_model = PATE_GANS_AUDIT[pg_name](**pg_kwargs)
pg_model.fit(df)

results = {"moments": pg_model.alphas_dict, "epsilons": pg_model.eps_dict}
with open(save_eval_path, 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
