import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import torch
import tensorflow as tf


from pate_gans import PATE_GANS_AUDIT
from utils import wb_get_auc_est_eps


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
save_eval_wb_path = config["data"]["save_eval_wb_path"]
# eval
epsilon = config["evaluation"]["epsilon"]
delta = config["evaluation"]["delta"]
alpha = config["evaluation"]["alpha"]
n_train = config["evaluation"]["n_train"]
n_valid = config["evaluation"]["n_valid"]
n_test = config["evaluation"]["n_test"]
n_all = n_train + n_valid + n_test
pgs = config["evaluation"]["pgs"]
if pgs == "all":
    pgs = PATE_GANS_AUDIT.keys()
pgs_kwargs = config["evaluation"]["pgs_kwargs"]
pgs_kwargs_fit = config["evaluation"]["pgs_kwargs_fit"]

dtypes = {"A": "int", "B": "int", "C": "int"}
df_out = pd.DataFrame({"A": [0, 0, 0, 0, 0],
                       "B": [0, 0, 0, 0, 0],
                       "C": [0, 0, 0, 0, 0]}).astype(dtypes)
df_in = pd.DataFrame({"A": [1, 0, 0, 0, 0, 0],
                      "B": [1, 0, 0, 0, 0, 0],
                      "C": [1, 0, 0, 0, 0, 0]}).astype(dtypes)
target = pd.DataFrame({"A": [1], "B": [1], "C": [1]}).astype(dtypes)

n_records_out, n_features_out = df_out.shape
n_records_in, n_features_in = df_in.shape
n_teachers = 2


wb_cols = ["df_name", "pg_name", "epsilon",
           "out_mean_t", "in_mean_t", "auc_t", "emp_eps_approxdp_t",
           "out_mean", "in_mean", "auc", "emp_eps_approxdp"]
wb_results_df = pd.DataFrame(columns=wb_cols)


for pg_name in tqdm(pgs, desc="pg", leave=False):
    # initialize and fit pate-gan
    pgs_kwargs[pg_name]["epsilon"] = epsilon
    pgs_kwargs[pg_name]["delta"] = delta if pg_name != "PG_ORIGINAL_AUDIT" else int(-np.log10(delta))
    pgs_kwargs[pg_name]["num_teachers"] = n_teachers

    wb_in_probs, wb_out_probs = np.zeros([n_valid + n_test]), np.zeros([n_valid + n_test])
    wb_t_in_probs, wb_t_out_probs = np.zeros([n_valid + n_test]), np.zeros([n_valid + n_test])

    for i in tqdm(range(n_all), desc="pg_it", leave=False):
        if pg_name in ["PG_BORAI_AUDIT", "PG_SMARTNOISE_AUDIT"]:
            pgs_kwargs_fit[pg_name]["skip_processing"] = True

        if pg_name in ["PG_ORIGINAL_AUDIT", "PG_UPDATED_AUDIT", "PG_TURING_AUDIT", "PG_BORAI_AUDIT"]:
            pgs_kwargs[pg_name]["X_shape"] = df_out.shape
        pg_model_out = PATE_GANS_AUDIT[pg_name](**pgs_kwargs[pg_name])
        try:
            pg_model_out.fit(df_out, **pgs_kwargs_fit[pg_name])
        except:
            print("Model training failed, skipping iteration")
            continue

        if pg_name in ["PG_ORIGINAL_AUDIT", "PG_UPDATED_AUDIT", "PG_TURING_AUDIT", "PG_BORAI_AUDIT"]:
            pgs_kwargs[pg_name]["X_shape"] = df_in.shape
        pg_model_in = PATE_GANS_AUDIT[pg_name](**pgs_kwargs[pg_name])
        try:
            pg_model_in.fit(df_in, **pgs_kwargs_fit[pg_name])
        except:
            print("Model training failed, skipping iteration")
            continue

        if pg_name in ["PG_UPDATED_AUDIT", "PG_SYNTHCITY_AUDIT", "PG_BORAI_AUDIT", "PG_SMARTNOISE_AUDIT"]:
            wb_t_out_probs[i] = float(pg_model_out.td_predict(target))
            wb_t_in_probs[i] = float(pg_model_in.td_predict(target))
        wb_out_probs[i] = float(pg_model_out.sd_predict(target))
        wb_in_probs[i] = float(pg_model_in.sd_predict(target))

        # create manual rules for some models/probabilities, "equivalent" to running a classifier
        if pg_name in ["PG_ORIGINAL_AUDIT", "PG_TURING_AUDIT"]:
            if wb_out_probs[i] == 1:
                wb_out_probs[i] = 0
            if wb_in_probs[i] == 1:
                wb_in_probs[i] = 0
        if pg_name in ["PG_UPDATED_AUDIT"]:
            if wb_t_out_probs[i] < 0.25:
                wb_t_out_probs[i] = 0.43
            if wb_t_in_probs[i] < 0.25:
                wb_t_in_probs[i] = 0.43
        if pg_name in ["PG_SYNTHCITY_AUDIT"]:
            if wb_t_out_probs[i] < 0.42:
                wb_t_out_probs[i] = 0.63
            if wb_t_in_probs[i] < 0.42:
                wb_t_in_probs[i] = 0.63

    # white-box eps estimation
    wb_t_auc, emp_eps_approxdp_t = wb_get_auc_est_eps(wb_t_out_probs, wb_t_in_probs, n_valid, n_test, delta, alpha)
    wb_auc, emp_eps_approxdp = wb_get_auc_est_eps(wb_out_probs, wb_in_probs, n_valid, n_test, delta, alpha)

    results = pd.DataFrame([[df_name, pg_name, epsilon,
                             wb_t_out_probs.mean(), wb_t_in_probs.mean(), wb_t_auc, emp_eps_approxdp_t,
                             wb_out_probs.mean(), wb_in_probs.mean(), wb_auc, emp_eps_approxdp]], columns=wb_cols)
    wb_results_df = pd.concat([wb_results_df, results], ignore_index=True)
    wb_results_df.to_pickle(save_eval_wb_path, compression="gzip")
