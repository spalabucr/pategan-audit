import numpy as np
from tqdm import tqdm
from scipy import stats

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor

from sklearn.metrics import roc_auc_score, average_precision_score

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


CLASSIFIERS = [
    "LogisticRegression",
    "RandomForest",
    "GaussianNB",
    "BernoulliNB",
    "LinearSVM",
    "DecisionTree",
    "LDA",
    "AdaBoost",
    "Bagging",
    "GBM",
    "MLP",
    "XGB",
]


def preprocess_Xy(df, test_df):
    min_max_scaler = MinMaxScaler()
    train_X, train_y = df.iloc[:, :-1], df.iloc[:, -1].to_numpy()
    train_X = min_max_scaler.fit_transform(train_X)

    test_X, test_y = test_df.iloc[:, :-1], test_df.iloc[:, -1].to_numpy()
    test_X = min_max_scaler.transform(test_X)

    return train_X, train_y, test_X, test_y


def run_classifiers(train_X, train_y, test_X, test_y, classifiers=CLASSIFIERS, majority=False):
    if majority:
        classifiers += ["majority"]
    results = np.zeros([len(classifiers), 2])

    for i, model_name in enumerate(tqdm(classifiers, desc='clf', leave=False)):
        if model_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=2000)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier()
        elif model_name == 'GaussianNB':
            model = GaussianNB()
        elif model_name == 'BernoulliNB':
            model = BernoulliNB()
        elif model_name == 'LinearSVM':
            model = LinearSVC(max_iter=10000)
        elif model_name == 'DecisionTree':
            model = DecisionTreeClassifier()
        elif model_name == 'LDA':
            model = LinearDiscriminantAnalysis()
        elif model_name == 'AdaBoost':
            model = AdaBoostClassifier()
        elif model_name == 'Bagging':
            model = BaggingClassifier()
        elif model_name == 'GBM':
            model = GradientBoostingClassifier()
        elif model_name == 'MLP':
            model = MLPClassifier(max_iter=1000)
        elif model_name == 'XGB':
            model = XGBRegressor()

        if len(np.unique(train_y)) > 1 and model_name != "majority":
            model.fit(train_X, train_y)
            if model_name == 'LinearSVM':
                predict_y = model.decision_function(test_X)
            elif model_name == 'XGB':
                predict_y = model.predict(np.asarray(test_X))
            else:
                predict_y = model.predict_proba(test_X)[:, 1]

            if np.isnan(predict_y).any():
                predict_y[np.isnan(predict_y)] = stats.mode(train_y)[0]

        else:
            predict_y = np.array([stats.mode(train_y)[0]] * len(test_y))

        auc = roc_auc_score(y_true=test_y, y_score=predict_y)
        apr = average_precision_score(y_true=test_y, y_score=predict_y)
        results[i, :] = [auc, apr]

    return results
