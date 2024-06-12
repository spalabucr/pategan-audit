import math
import numpy as np
from tqdm import tqdm
import concurrent.futures
from scipy.optimize import root_scalar
from scipy.stats import binomtest, norm
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from privacy_estimates import AttackResults, compute_eps_lo_hi


def wb_get_auc_est_eps(out_probs, in_probs, n_valid, n_test, delta):
    mia_preds = np.concatenate([out_probs, out_probs])
    mia_labels = np.array([0] * (n_valid + n_test) + [1] * (n_valid + n_test))
    auc = roc_auc_score(mia_labels, mia_preds)

    scoress = np.concatenate([mia_preds.reshape(-1, 1), mia_labels.reshape(-1, 1)], axis=1)
    emp_eps = estimate_eps(scoress, n_valid, alpha=0.1, delta=delta, method='cp', n_procs=1)
    scoress_inv_thresh = np.concatenate([1 - mia_preds.reshape(-1, 1), mia_labels.reshape(-1, 1)], axis=1)
    emp_eps_inv_thresh = estimate_eps(scoress_inv_thresh, n_valid, alpha=0.1, delta=delta, method='cp', n_procs=1)
    emp_eps = max(emp_eps, emp_eps_inv_thresh)

    return auc, emp_eps


def bb_get_auc_est_eps(out_data, in_data, n_train, n_valid, n_test, delta):
    # prepare train data
    train_data = np.concatenate([out_data[: n_train], in_data[: n_train]])
    train_labels = np.array([0] * n_train + [1] * n_train)

    # train classifier
    clf = RandomForestClassifier()
    clf.fit(train_data, train_labels)

    mia_preds = np.concatenate([clf.predict_proba(out_data[n_train:])[:, 1],
                                clf.predict_proba(in_data[n_train:])[:, 1]])
    mia_labels = np.array([0] * (n_valid + n_test) + [1] * (n_valid + n_test))
    auc = roc_auc_score(mia_labels, mia_preds)

    # estimate epsilon
    scoress = np.concatenate([mia_preds.reshape(-1, 1), mia_labels.reshape(-1, 1)], axis=1)
    emp_eps = estimate_eps(scoress, n_valid, alpha=0.1, delta=delta, method='cp', n_procs=1)
    scoress_inv_thresh = np.concatenate([1 - mia_preds.reshape(-1, 1), mia_labels.reshape(-1, 1)], axis=1)
    emp_eps_inv_thresh = estimate_eps(scoress_inv_thresh, n_valid, alpha=0.1, delta=delta, method='cp', n_procs=1)
    emp_eps = max(emp_eps, emp_eps_inv_thresh)

    return auc, emp_eps


def estimate_eps(scoress, n_valid, alpha=0.1, delta=0, method='all', n_procs=32, choose_thresh='holdout'):
    # find optimal threshold using validation set
    scoress_in, scoress_out = scoress[scoress[:, 1] == 1], scoress[scoress[:, 1] == 0]
    mia_scores_valid = np.concatenate([scoress_in[:n_valid, 0], scoress_out[:n_valid, 0]])
    mia_labels_valid = np.concatenate([scoress_in[:n_valid, 1], scoress_out[:n_valid, 1]])
    opt_t, _ = compute_eps_lower_from_mia(mia_scores_valid, mia_labels_valid, alpha=alpha, delta=delta, method=method, n_procs=n_procs)

    # calculate epsilon based on optimal threshold on test set
    mia_scores_test = np.concatenate([scoress_in[n_valid:, 0], scoress_out[n_valid:, 0]])
    mia_labels_test = np.concatenate([scoress_in[n_valid:, 1], scoress_out[n_valid:, 1]])
    tp = np.sum(mia_scores_test[mia_labels_test == 1] >= opt_t)
    fp = np.sum(mia_scores_test[mia_labels_test == 0] >= opt_t)
    fn = np.sum(mia_scores_test[mia_labels_test == 1] < opt_t)
    tn = np.sum(mia_scores_test[mia_labels_test == 0] < opt_t)

    results = AttackResults(FN=fn, FP=fp, TN=tn, TP=tp)
    return compute_eps_lower_single(results, alpha, delta, method=method)


def compute_eps_lower_from_mia(scores, labels, alpha, delta, method='all', n_procs=32):
    """
    Compute lower bound for epsilon using privacy estimation procedure described in https://proceedings.mlr.press/v202/zanella-beguelin23a/zanella-beguelin23a.pdf
    Step 1: Fix a set of significant decision thresholds (from TAPAS)
    Step 2: For each threshold, calculate TP, FP, TN, FN and estimate epsilon lower bound using different methods at a given significance level alpha and delta
    Step 3: Output the maximum epsilon lower bound
    """
    scores, labels = np.array(scores), np.array(labels)
    significant_thresholds = np.sort(np.unique(scores))

    resultss = []
    for t in significant_thresholds:
        tp = np.sum(scores[labels == 1] >= t)
        fp = np.sum(scores[labels == 0] >= t)
        fn = np.sum(scores[labels == 1] < t)
        tn = np.sum(scores[labels == 0] < t)

        results = AttackResults(FN=fn, FP=fp, TN=tn, TP=tp)
        resultss.append((t, results))

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor, \
         tqdm(total=len(resultss), leave=False) as pbar:

        futures = {}
        for results in resultss:
            t, curr_results = results
            futures[executor.submit(compute_eps_lower_single, curr_results, alpha, delta, method)] = t

        max_eps_lo, max_t = None, None
        for future in concurrent.futures.as_completed(futures):
            curr_max_eps_lo = future.result()
            t = futures[future]
            if not math.isnan(curr_max_eps_lo) and (max_eps_lo is None or curr_max_eps_lo > max_eps_lo):
                max_eps_lo = curr_max_eps_lo
                max_t = t
            pbar.update(1)

    return max_t, max_eps_lo


def compute_eps_lower_single(results, alpha, delta, method='all'):
    """
    Given TP, FP, TN, FN, estimate epsilon lower bound using different methods at a given significance level alpha and delta
    """
    if method == 'GDP':
        return eps_l_GDP(results, alpha, delta)

    eps_lo_jb, eps_lo_b, eps_lo_j = -1, -1, -1

    # calculations can run into numerical errors hence wrapped around try, except blocks
    if method == 'all' or method == 'zb':
        try:
            eps_lo_jb, _ = compute_eps_lo_hi(count=results, delta=delta, alpha=alpha, method="joint-beta")
        except Exception:
            pass

    if method == 'all' or method == 'cp':
        try:
            eps_lo_b, _ = compute_eps_lo_hi(count=results, delta=delta, alpha=alpha, method="beta")
        except Exception:
            pass

    if method == 'all' or method == 'jeff':
        try:
            eps_lo_j, _ = compute_eps_lo_hi(count=results, delta=delta, alpha=alpha, method="jeffreys")
        except Exception:
            pass

    curr_max_eps_lo = max(eps_lo_jb, eps_lo_b, eps_lo_j)
    return curr_max_eps_lo


# Privacy accounting using GDP
def eps_l_GDP(results, alpha, delta):
    # Step 1: calculate CP upper bound on FPR and FNR at significance level alpha
    _, fpr_r = binomtest(int(results.FP), int(results.N)).proportion_ci(confidence_level=1 - alpha / 2)
    _, fnr_r = binomtest(int(results.FN), int(results.P)).proportion_ci(confidence_level=1 - alpha / 2)

    # Step 2: calculate lower bound on mu-GDP
    mu_l = norm.ppf(1 - fpr_r) - norm.ppf(fnr_r)

    if mu_l < 0:
        # GDP is not defined for mu < 0
        return 0

    try:
        # Step 3: convert mu-GDP to (eps, delta)-DP using Equation (6) from Tight Auditing DPML paper
        def eq6(epsilon):
            return norm.cdf(-epsilon / mu_l + mu_l / 2) - np.exp(epsilon) * norm.cdf(-epsilon / mu_l - mu_l / 2) - delta

        sol = root_scalar(eq6, bracket=[0, 50], method='brentq')
        eps_l = sol.root
    except Exception:
        eps_l = 0

    return eps_l
