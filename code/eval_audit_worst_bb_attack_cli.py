import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import product
from argparse import ArgumentParser

import torch
import tensorflow as tf


from pate_gans import PATE_GANS_AUDIT
from utils import fix_dtypes, featurize_df_queries, bb_get_auc_est_eps


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
save_eval_path = config["data"]["save_eval_path"]
# generation
epsilon = config["generation"]["epsilon"]
delta = config["generation"]["delta"]
pgs = config["generation"]["pgs"]
if pgs == "all":
    pgs = PATE_GANS_AUDIT.keys()
pgs_kwargs = config["generation"]["pgs_kwargs"]
pgs_fit_kwargs = {pg_name: {} for pg_name in pgs}
n_pgs = config["generation"]["n_pgs"]
# evaluation
alpha = config["evaluation"]["alpha"]
n_train = config["evaluation"]["n_train"]
n_valid = config["evaluation"]["n_valid"]
n_test = config["evaluation"]["n_test"]
n_all = n_train + n_valid + n_test
assert n_all == 1000

dtypes = {"A": "int", "B": "int", "C": "int"}
df_out = pd.DataFrame({"A": [0, 0, 0, 0],
                       "B": [0, 0, 0, 0],
                       "C": [0, 0, 0, 0]}).astype(dtypes)
df_in = pd.DataFrame({"A": [1, 0, 0, 0, 0],
                      "B": [1, 0, 0, 0, 0],
                      "C": [1, 0, 0, 0, 0]}).astype(dtypes)

queries = np.array(list(product([0, 1], repeat=3)))

n_records_out, n_features_out = df_out.shape
n_records_in, n_features_in = df_in.shape
n_teachers = 2


# initialize df results
cols = ["df_name", "pg_name", "epsilon", "auc", "emp_eps_approxdp"]
results_df = pd.DataFrame(columns=cols)


# initialize and fit pate-gan
for pg_name in tqdm(pgs, desc="pg", leave=False):
    pgs_kwargs[pg_name]["epsilon"] = epsilon
    pgs_kwargs[pg_name]["delta"] = delta if pg_name != "PG_ORIGINAL_AUDIT" else int(-np.log10(delta))
    pgs_kwargs[pg_name]["num_teachers"] = n_teachers

    if pg_name in ["PG_SMARTNOISE_AUDIT"]:
        pgs_fit_kwargs[pg_name]["skip_processing"] = True

    if pg_name in ["PG_ORIGINAL_AUDIT", "PG_UPDATED_AUDIT", "PG_TURING_AUDIT", "PG_BORAI_AUDIT"]:
        pgs_kwargs[pg_name]["X_shape"] = (n_records_out, n_features_out)

    out_data = np.zeros([n_all, len(queries)])
    for i in tqdm(range(n_pgs), desc="out", leave=False):
        pg_model = PATE_GANS_AUDIT[pg_name](**pgs_kwargs[pg_name])
        pg_model.fit(df_out, **pgs_fit_kwargs[pg_name])
        synth_df_out = pg_model.generate(n_records_out)
        synth_df_out = fix_dtypes(synth_df_out, dtypes)
        out_data[i] = featurize_df_queries(synth_df_out, queries)

    if pg_name in ["PG_ORIGINAL_AUDIT", "PG_UPDATED_AUDIT", "PG_TURING_AUDIT", "PG_BORAI_AUDIT"]:
        pgs_kwargs[pg_name]["X_shape"] = (n_records_in, n_features_in)

    in_data = np.zeros([n_all, len(queries)])
    for i in tqdm(range(n_pgs), desc="in", leave=False):
        pg_model = PATE_GANS_AUDIT[pg_name](**pgs_kwargs[pg_name])
        pg_model.fit(df_in, **pgs_fit_kwargs[pg_name])
        synth_df_in = pg_model.generate(n_records_in)
        synth_df_in = fix_dtypes(synth_df_in, dtypes)
        in_data[i] = featurize_df_queries(synth_df_in, queries)

    auc, emp_eps_approxdp = bb_get_auc_est_eps(out_data, in_data, n_train, n_valid, n_test, delta, alpha)
    results = pd.DataFrame([[df_name, pg_name, epsilon, auc, emp_eps_approxdp]], columns=cols)
    results_df = pd.concat([results_df, results], ignore_index=True)
    results_df.to_pickle(save_eval_path, compression="gzip")
