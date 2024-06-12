import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import torch
import tensorflow as tf
from sklearn.metrics import roc_auc_score


from pate_gans import PATE_GANS_AUDIT
from utils import get_vuln_records_dists


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
epsilon = config["evaluation"]["epsilon"]
delta = config["evaluation"]["delta"]
teachers_ratio = config["evaluation"]["teachers_ratio"]
pg_name = config["evaluation"]["pg"]
pg_kwargs = config["evaluation"]["pg_kwargs"]

n_vuln_records = config["evaluation"]["n_vuln_records"]
n_pgs = config["evaluation"]["n_pgs"]


df = pd.read_pickle(df_path)
print(f"LOAD: {df_name} [shape {df.shape}] loaded\n")
n_teachers = max(2, len(df) // teachers_ratio)

# initialize df results
cols = ["df_name", "idx", "distance", "pg_name", "epsilon", "out_mean", "in_mean", "auc"]
results_df = pd.DataFrame(columns=cols)

# get vulnerable records
vuln_records_dists = get_vuln_records_dists(df)
vulns_records_ids = np.argsort(vuln_records_dists)[-n_vuln_records:][::-1]


for vuln_record_id in tqdm(vulns_records_ids, desc="vuln_id", leave=False):
    # initialize and fit pate-gan
    pg_kwargs["epsilon"] = epsilon
    pg_kwargs["delta"] = delta if pg_name != "PG_ORIGINAL_AUDIT" else int(-np.log10(delta))
    pg_kwargs["num_teachers"] = n_teachers

    df_in = df.copy()
    df_out = df.copy().drop(vuln_record_id)
    target = df.iloc[[vuln_record_id]]

    in_probs = np.zeros([n_pgs])
    out_probs = np.zeros([n_pgs])

    for i in tqdm(range(n_pgs), desc="pg_it", leave=False):
        if pg_name in ["PG_ORIGINAL_AUDIT", "PG_UPDATED_AUDIT", "PG_TURING_AUDIT", "PG_BORAI_AUDIT"]:
            pg_kwargs["X_shape"] = df_out.shape
        pg_model_out = PATE_GANS_AUDIT[pg_name](**pg_kwargs)
        pg_model_out.fit(df_out)

        if pg_name in ["PG_ORIGINAL_AUDIT", "PG_UPDATED_AUDIT", "PG_TURING_AUDIT", "PG_BORAI_AUDIT"]:
            pg_kwargs["X_shape"] = df_in.shape
        pg_model_in = PATE_GANS_AUDIT[pg_name](**pg_kwargs)
        pg_model_in.fit(df_in)

        # white-box access to discriminator
        out_probs[i] = float(pg_model_out.sd_predict(target))
        in_probs[i] = float(pg_model_in.sd_predict(target))

    mia_preds = np.concatenate([out_probs, in_probs])
    mia_labels = np.array([0] * n_pgs + [1] * n_pgs)
    auc = roc_auc_score(mia_labels, mia_preds)

    results = pd.DataFrame([[df_name, vuln_record_id, vuln_records_dists[vuln_record_id],
                             pg_name, epsilon, out_probs.mean(), in_probs.mean(), auc]], columns=cols)
    results_df = pd.concat([results_df, results], ignore_index=True)
    results_df.to_pickle(save_eval_path, compression="gzip")
