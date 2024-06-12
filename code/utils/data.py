import numpy as np
import pandas as pd


def wrap(els, els_len=None):
    if type(els) == list:
        return np.array([[el] for el in els])
    else:
        return np.array([[els] for _ in range(els_len)])


def concat_results(results_df, df_name, pg_name, epsilon, i_pg, i_sd, classifiers, results):
    all_results = np.concatenate((wrap(df_name, len(classifiers)),
                                  wrap(pg_name, len(classifiers)),
                                  wrap(epsilon, len(classifiers)),
                                  wrap(i_pg, len(classifiers)),
                                  wrap(i_sd, len(classifiers)),
                                  wrap(classifiers),
                                  results
                                  ), axis=1)
    new_results_df = pd.DataFrame(all_results, columns=results_df.columns)
    results_df = pd.concat([results_df, new_results_df], ignore_index=True)
    return results_df.astype({"auroc": float, "auprc": float})


def fix_dtypes(df, dtypes):
    if type(df) != pd.DataFrame:
        df = pd.DataFrame(df, columns=dtypes.keys())

    for col, dtype in dtypes.items():
        if dtype in ["int"]:
            df[col] = np.round(df[col]).astype(int)

    df = df.astype(dtypes)
    return df


def featurize_df_queries(df, queries):
    features = np.zeros(len(queries))
    for i, query in enumerate(queries):
        features[i] = (df == query).all(axis=1).sum()
    return features.astype(int)


def featurize_df_naive(df):
    features = np.zeros(5 * len(df.columns))
    for i, col in enumerate(df.columns):
        col_data = df[col]
        features[i * 5: (i + 1) * 5] = [col_data.min(),
                                        col_data.max(),
                                        col_data.mean(),
                                        col_data.median(),
                                        col_data.var()]
    return features


def get_histogram_domain(df, n_bins):
    domain = {"min_max": {}, "order": {}}
    for col in df.columns:
        col_data = df[col]
        if col_data.nunique() > n_bins:
            domain["min_max"][col] = (col_data.min(), col_data.max())
        else:
            domain["order"][col] = col_data.value_counts().index.to_list()
    return domain


def featurize_df_histogram(df, n_bins, domain):
    features = np.empty(0)
    for col in df.columns:
        col_data = df[col]
        if col in domain["min_max"].keys():
            bins = np.linspace(domain["min_max"][col][0], domain["min_max"][col][1], n_bins + 1)
            col_features = pd.Series(0, index=pd.IntervalIndex.from_breaks(bins, closed="right"))
            col_features.update(col_data.value_counts(normalize=True, bins=bins))
            col_features = col_features.to_numpy()
        else:
            col_data = col_data.astype(pd.CategoricalDtype(categories=domain["order"][col]))
            col_features = col_data.value_counts(normalize=True)[domain["order"][col]].to_numpy()
        features = np.concatenate((features, col_features))
    return features


def featurize_df_corr(df):
    corrs = df.corr().to_numpy()
    mask = np.triu_indices_from(corrs, k=1)
    features = corrs[mask]
    return features
