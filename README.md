## The Elusive Pursuit of Reproducing PATE-GAN: Benchmarking, Auditing, Debugging

This repository contains the source code for the paper The Elusive Pursuit of Replicating PATE-GAN: Benchmarking, Auditing, Debugging by G. Ganev, M.S.M.S. Annamalai, E. De Cristofaro


## Install

The experiments require Python 3.10.
All necessary dependencies are listed in `requirements.txt`.
Since there are conflicts between some libraries, i.e., `synthcity` and `smartnoise-synth`, it is recommended to install the dependencies manually.
Nevertheless, after manual installation, all experiments run successfully.


## Source Code Structure

The source code is broken down into two folders -- `code` and `data`.
We present a brief summary:

1. `code` contains all code necessary for running the experiments.
It is further broken down into:

  1. `pate_gans`: includes the code of the six PATE-GAN implementations taken from their corresponding repos ([original](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/pategan/PATE_GAN.py), [updated](https://github.com/vanderschaarlab/mlforhealthlabpub/blob/main/alg/pategan/pate_gan.py), [synthcity](https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/plugins/privacy/plugin_pategan.py), [turing](https://github.com/alan-turing-institute/reprosyn/blob/main/src/reprosyn/methods/gans/pate_gan.py), [borealis](https://github.com/BorealisAI/private-data-generation/blob/master/models/pate_gan.py), and [smartnoise](https://github.com/opendp/smartnoise-sdk/blob/main/synth/snsynth/pytorch/nn/pategan.py)).
  2. `utils`: includes the utility functions necessary for both utility and privacy evaluations.
  3. `configs`: includes the configuration files necessary for running both utility and privacy evaluations.
  3. `python scripts & jupyter notebook`: includes the python scripts for running the utility/privacy experiments and the jupyter notebooks for visualizing the results (more details below).

2. `data` contains the four public datasets used in our evaluations (Kaggle Credit, Kaggle Cervical Cancer, UCI ISOLET, and UCI Epileptic Seizure) as well as the `results` folder, in which all the results from the utility/privacy evaluations and saved.


## Run Experiments
All experiments and tables/plots in the paper can be replicated by running the following code.

### 1. Utility Benchmark
The utility results are presented and discussed in Section 5.
The utility scripts are:

* `code/eval_utility_cli.py`
* `code/eval_utility_teachers_cli.py`

To (re)create the files in `data/results/utility`, one can run the commands in `scripts_utility.txt` from `code`.

### 2. Privacy Evaluation (including Privacy Auditing)
The privacy evaluations results are presented and discussed in Section 6.
The privacy evaluation scripts are:

* `code/eval_audit_teachers_seen_cli.py`
* `code/eval_audit_teachers_loss_cli.py`
* `code/eval_audit_moments_cli.py`
* `code/eval_audit_worst_bb_attack_cli.py`
* `code/eval_audit_select_vuln_records_cli.py`
* `code/eval_audit_average_bb_attack_cli.py`

To (re)create the files in `data/results/audit`, one can run the commands in `scripts_audit.txt` from `code`.

### 3. Visualizations
The tables/plots are presented throughout the paper.
To (re)create them (once all the scripts above are run and the results saved in `data/results`), one can run the jupyter notebook `code/nb_plot.ipynb`.
