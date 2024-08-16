import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import product
from argparse import ArgumentParser

import torch
import tensorflow as tf


from pate_gans import PATE_GANS
from utils import CLASSIFIERS, fix_dtypes, preprocess_Xy, run_classifiers


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
epsilons = config["generation"]["epsilons"]
delta = config["generation"]["delta"]
# hyperparams
teachers_ratio_list = config["generation"]["teachers_ratio"]
lamda_list = config["generation"]["lamda"]
alpha_list = config["generation"]["alpha"]
lr_list = config["generation"]["lr"]
batch_size_list = config["generation"]["batch_size"]
hyperparams_list = [teachers_ratio_list, lamda_list,
                    alpha_list, lr_list, batch_size_list]
hyperparams_combinations = list(product(*hyperparams_list))

pgs = config["generation"]["pgs"]
if pgs == "all":
    pgs = PATE_GANS.keys()
pgs_kwargs = config["generation"]["pgs_kwargs"]
n_pgs_per_epsilon = config["generation"]["n_pgs_per_epsilon"]
n_synth_dfs_per_pg = config["generation"]["n_synth_dfs_per_pg"]
# evaluation
classifiers = config["evaluation"]["classifiers"]
if classifiers == "all":
    classifiers = CLASSIFIERS


df = pd.read_pickle(df_path)
test_df = pd.read_pickle(test_df_path)
print(f"LOAD: {df_name} [shape {df.shape}] loaded\n")

dtypes = df.dtypes
n_records, n_features = df.shape

# initialize df results
cols = ["df_name", "pg_name", "epsilon", "num_teachers", "lamda", "alpha", "lr",
        "batch_size", "pg_it", "sd_it"] + [f"{cls}_aucroc" for cls in classifiers]
results_df = pd.DataFrame(columns=cols)


# generate synth df
for epsilon in tqdm(epsilons, desc="epsilon"):
    for pg_name in tqdm(pgs, desc="pg", leave=False):
        for hyperparams_combination in tqdm(hyperparams_combinations, desc="hyp it", leave=False):
            for i_pg in tqdm(range(n_pgs_per_epsilon), desc="pg it", leave=False):
                # manually update model kwargs
                pgs_kwargs[pg_name]["epsilon"] = epsilon
                pgs_kwargs[pg_name]["delta"] = delta

                n_teachers = max(2, n_records // hyperparams_combination[0])
                pgs_kwargs[pg_name]["num_teachers"] = n_teachers
                pgs_kwargs[pg_name]["lamda"] = hyperparams_combination[1]
                pgs_kwargs[pg_name]["alpha"] = hyperparams_combination[2]
                pgs_kwargs[pg_name]["lr"] = hyperparams_combination[3]
                pgs_kwargs[pg_name]["batch_size"] = hyperparams_combination[4]

                # initialize and fit pate-gan
                pg_model = PATE_GANS[pg_name](**pgs_kwargs[pg_name])
                pg_model.fit(df)

                for i_sd in tqdm(range(n_synth_dfs_per_pg), desc="sd it", leave=False):
                    synth_df = pg_model.generate(n_records)
                    synth_df = fix_dtypes(synth_df, dtypes)

                    # run utility evaluation on synth df
                    synth_X, synth_y, test_X, test_y = preprocess_Xy(synth_df, test_df)
                    results = run_classifiers(synth_X, synth_y, test_X, test_y, classifiers)[:, 0]

                    new_results_df = pd.DataFrame([[df_name, pg_name, epsilon, n_teachers] +
                                                   list(hyperparams_combination[1:]) +
                                                   [i_pg, i_sd] + list(results)], columns=cols)

                    results_df = pd.concat([results_df, new_results_df], ignore_index=True)
                    results_df.to_pickle(save_eval_path, compression="gzip")