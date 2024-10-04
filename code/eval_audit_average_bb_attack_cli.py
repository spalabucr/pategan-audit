import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import torch
import tensorflow as tf


from pate_gans import PATE_GANS_AUDIT, PATE_GANS_FIX
from utils import fix_dtypes, featurize_df_naive, get_histogram_domain, featurize_df_histogram, featurize_df_corr, bb_get_auc_est_eps


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
save_eval_bb_path = config["data"]["save_eval_bb_path"]
# eval
epsilon = config["evaluation"]["epsilon"]
delta = config["evaluation"]["delta"]
alpha = config["evaluation"]["alpha"]
teachers_ratio = config["evaluation"]["teachers_ratio"]

vuln_record_id = config["evaluation"]["vuln_record_id"]
n_bins_hist_feat = config["evaluation"]["n_bins_hist_feat"]
n_train = config["evaluation"]["n_train"]
n_valid = config["evaluation"]["n_valid"]
n_test = config["evaluation"]["n_test"]
pgs = config["evaluation"]["pgs"]
if pgs == "all":
    pgs = PATE_GANS_AUDIT.keys()
elif pgs == "fix":
    pgs = PATE_GANS_FIX.keys()
pgs_kwargs = config["evaluation"]["pgs_kwargs"]
pgs_kwargs_fit = config["evaluation"]["pgs_kwargs_fit"]


df = pd.read_pickle(df_path)
dtypes = df.dtypes
print(f"LOAD: {df_name} [shape {df.shape}] loaded\n")
n_teachers = max(2, len(df) // teachers_ratio)


# initialize df results
df_in = df.copy()
df_out = df.copy().drop(vuln_record_id)
target = df.iloc[[vuln_record_id]]


bb_cols = ["df_name", "pg_name", "epsilon",
           "naive_auc", "naive_emp_eps_approxdp", "hist_auc", "hist_emp_eps_approxdp",
           "corr_auc", "corr_emp_eps_approxdp", "ens_auc", "ens_emp_eps_approxdp"]
bb_results_df = pd.DataFrame(columns=bb_cols)

histogram_domain = get_histogram_domain(df, n_bins_hist_feat)
n_bb_features = {"naive": len(featurize_df_naive(df)),
                 "histogram": len(featurize_df_histogram(df, n_bins_hist_feat, histogram_domain)),
                 "corr": len(featurize_df_corr(df))}
n_bb_features["ensemble"] = sum(n_bb_features.values())
n_all = n_train + n_valid + n_test


for pg_name in tqdm(pgs, desc="pg", leave=False):
    # initialize and fit pate-gan
    pgs_kwargs[pg_name]["epsilon"] = epsilon
    pgs_kwargs[pg_name]["delta"] = delta if pg_name != "PG_ORIGINAL_AUDIT" else int(-np.log10(delta))
    pgs_kwargs[pg_name]["num_teachers"] = n_teachers

    bb_na_in_data, bb_na_out_data = np.zeros([n_all, n_bb_features["naive"]]), np.zeros([n_all, n_bb_features["naive"]])
    bb_hi_in_data, bb_hi_out_data = np.zeros([n_all, n_bb_features["histogram"]]), np.zeros([n_all, n_bb_features["histogram"]])
    bb_co_in_data, bb_co_out_data = np.zeros([n_all, n_bb_features["corr"]]), np.zeros([n_all, n_bb_features["corr"]])
    bb_en_in_data, bb_en_out_data = np.zeros([n_all, n_bb_features["ensemble"]]), np.zeros([n_all, n_bb_features["ensemble"]])

    for i in tqdm(range(n_all), desc="pg_it", leave=False):
        if pg_name in ["PG_ORIGINAL_AUDIT", "PG_UPDATED_AUDIT", "PG_TURING_AUDIT", "PG_BORAI_AUDIT"]:
            pgs_kwargs[pg_name]["X_shape"] = df_out.shape
        pg_model_out = PATE_GANS_AUDIT[pg_name](**pgs_kwargs[pg_name]) if pg_name in PATE_GANS_AUDIT else PATE_GANS_FIX[pg_name](**pgs_kwargs[pg_name])
        pg_model_out.fit(df_out, **pgs_kwargs_fit[pg_name])

        if pg_name in ["PG_ORIGINAL_AUDIT", "PG_UPDATED_AUDIT", "PG_TURING_AUDIT", "PG_BORAI_AUDIT"]:
            pgs_kwargs[pg_name]["X_shape"] = df_in.shape
        pg_model_in = PATE_GANS_AUDIT[pg_name](**pgs_kwargs[pg_name]) if pg_name in PATE_GANS_AUDIT else PATE_GANS_FIX[pg_name](**pgs_kwargs[pg_name])
        pg_model_in.fit(df_in, **pgs_kwargs_fit[pg_name])

        # generate data for black box
        synth_df_out = pg_model_out.generate(len(df))
        synth_df_out = fix_dtypes(synth_df_out, dtypes)
        synth_df_in = pg_model_in.generate(len(df))
        synth_df_in = fix_dtypes(synth_df_in, dtypes)

        # naive features
        bb_na_out_data[i] = featurize_df_naive(synth_df_out)
        bb_na_in_data[i] = featurize_df_naive(synth_df_in)
        # histogram features
        bb_hi_out_data[i] = featurize_df_histogram(synth_df_out, n_bins_hist_feat, histogram_domain)
        bb_hi_in_data[i] = featurize_df_histogram(synth_df_in, n_bins_hist_feat, histogram_domain)
        # corr features
        bb_co_out_data[i] = featurize_df_corr(synth_df_out)
        bb_co_in_data[i] = featurize_df_corr(synth_df_in)
        # ensemble features
        bb_en_out_data[i] = np.concatenate((bb_na_out_data[i], bb_hi_out_data[i], bb_co_out_data[i]))
        bb_en_in_data[i] = np.concatenate((bb_na_in_data[i], bb_hi_in_data[i], bb_co_in_data[i]))

    # black-box eps estimation
    bb_na_auc, emp_eps_approxdp_na = bb_get_auc_est_eps(bb_na_out_data, bb_na_in_data, n_train, n_valid, n_test, delta, alpha)
    bb_hi_auc, emp_eps_approxdp_hi = bb_get_auc_est_eps(bb_hi_out_data, bb_hi_in_data, n_train, n_valid, n_test, delta, alpha)
    bb_co_auc, emp_eps_approxdp_co = bb_get_auc_est_eps(bb_co_out_data, bb_co_in_data, n_train, n_valid, n_test, delta, alpha)
    bb_en_auc, emp_eps_approxdp_en = bb_get_auc_est_eps(bb_en_out_data, bb_en_in_data, n_train, n_valid, n_test, delta, alpha)

    results = pd.DataFrame([[df_name, pg_name, epsilon,
                             bb_na_auc, emp_eps_approxdp_na, bb_hi_auc, emp_eps_approxdp_hi,
                             bb_co_auc, emp_eps_approxdp_co, bb_en_auc, emp_eps_approxdp_en]], columns=bb_cols)
    bb_results_df = pd.concat([bb_results_df, results], ignore_index=True)
    bb_results_df.to_pickle(save_eval_bb_path, compression="gzip")
