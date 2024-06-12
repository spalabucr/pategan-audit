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
from utils import featurize_df_queries, bb_get_auc_est_eps


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
synth_dfs_path = config["data"]["synth_dfs_path"]
save_eval_path = config["data"]["save_eval_path"]
# eval
epsilon = config["evaluation"]["epsilon"]
delta = config["evaluation"]["delta"]
n_train = config["evaluation"]["n_train"]
n_valid = config["evaluation"]["n_valid"]
n_test = config["evaluation"]["n_test"]
pgs = config["evaluation"]["pgs"]
if pgs == "all":
    pgs = PATE_GANS_AUDIT.keys()

# initialize df results
cols = ["df_name", "pg_name", "epsilon", "auc", "emp_eps_approxdp"]
results_df = pd.DataFrame(columns=cols)


n_all = n_train + n_valid + n_test
assert n_all == 1000
queries = np.array(list(product([0, 1], repeat=3)))


for pg_name in tqdm(pgs, desc="pg", leave=False):
    # read dfs
    out_data = np.zeros([n_all, len(queries)])
    for i in range(n_all):
        _out_df = pd.read_pickle(f"{synth_dfs_path}/{pg_name}/out_{i}.pkl.gz")
        out_data[i] = featurize_df_queries(_out_df, queries)

    in_data = np.zeros([n_all, len(queries)])
    for i in range(n_all):
        _in_df = pd.read_pickle(f"{synth_dfs_path}/{pg_name}/in_{i}.pkl.gz")
        in_data[i] = featurize_df_queries(_in_df, queries)

    auc, emp_eps_approxdp = bb_get_auc_est_eps(out_data, in_data, n_train, n_valid, n_test, delta)

    results = pd.DataFrame([[df_name, pg_name, epsilon, auc, emp_eps_approxdp]], columns=cols)
    results_df = pd.concat([results_df, results], ignore_index=True)
    results_df.to_pickle(save_eval_path, compression="gzip")
