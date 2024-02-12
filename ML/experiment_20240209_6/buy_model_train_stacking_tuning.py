import json
import math
from typing import Dict, TypedDict
import sys

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier

sys.path.append('/root/chan.py')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, make_scorer, log_loss
from sklearn.model_selection import GridSearchCV

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np
from joblib import dump, load


def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (precision_score(y, yPred),
            roc_auc_score(y, yPred, average='macro'))


def train_buy_model(code, begin_time, end_time):
    X, y = load_svmlight_file(f"buy_feature_{code}_bsp5_20240209_6.libsvm")  # load sample
    # X = X.toarray()
    print(np.unique(y, return_counts=True))

    clf3 = make_pipeline(
         SimpleImputer(strategy="constant", fill_value=np.nan),
         xgb.XGBClassifier(objective='binary:logistic',
                             scale_pos_weight=3.66, reg_lambda=20, subsample=0.9, random_state=1, missing=np.nan)
    )
    clf4 = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        BalancedRandomForestClassifier(random_state=1, sampling_strategy="all", replacement=True,
                                       bootstrap=False)
    )
    # clf4 = BalancedRandomForestClassifier(n_estimators=300, random_state=1, sampling_strategy="all", replacement=True,
    #                                    bootstrap=False)
    # clf5 = MLPClassifier(random_state=1, max_iter=1000)
    # lr = LogisticRegression(class_weight='balanced')
    # sclf = StackingClassifier(classifiers=[clf3, clf4],
    #                           use_probas=True, average_probas=False,
    #                           meta_classifier=lr)
    lr = LogisticRegression(class_weight='balanced')
    sclf = StackingClassifier(classifiers=[clf3, clf4],
                              use_probas=True, average_probas=False,
                              meta_classifier=lr)
    cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    # for clf, label in zip([clf3, clf4, sclf], ['xgb', 'blancedRF', 'StackingClassifier']):
    # for clf, label in zip([sclf], ['StackingClassifier']):
    #     auc_scores = model_selection.cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    #     print("roc_auc: % 0.2f(+ / - % 0.2f) [ % s]" % (auc_scores.mean(), auc_scores.std(), label))
    #     precision_scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='precision')
    #     print("precision: % 0.2f(+ / - % 0.2f) [ % s]" % (precision_scores.mean(), precision_scores.std(), label))

    # sclf.get_params()
    params = {'pipeline-1__xgbclassifier__learning_rate': [0.05, 0.1, 0.15, 0.2],
               'pipeline-1__xgbclassifier__n_estimators': [20, 30, 40, 50],
              'pipeline-1__xgbclassifier__max_depth': [20, 30, 40],
              'pipeline-2__balancedrandomforestclassifier__n_estimators': [200, 300]}

    # LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True, labels=[0, 1])
    grid = GridSearchCV(estimator=sclf,
                        param_grid=params,
                        scoring='precision',
                        cv=5,
                        refit=True,
                        verbose=100,
                        n_jobs=-1)
    grid.fit(X, y)

    cv_keys = ('mean_test_score', 'std_test_score', 'params')
    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r],
                 grid.cv_results_[cv_keys[1]][r] / 2.0,
                 grid.cv_results_[cv_keys[2]][r]))

    print('Best parameters: %s' % grid.best_params_)
    print('precision: %.2f' % grid.best_score_)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    sclf.fit(X_train, y_train)
    # sclf.fit(X, y)
    preds = sclf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    preds_binary = (preds > 0.55).astype(int)
    precision = precision_score(y_test, preds_binary, pos_label=1)
    print(f"test AUC: {auc}")
    print(f"test precision: {precision}")
    print('调整阈值后正样本比例： ', sum(preds_binary) / len(preds_binary), sum(preds_binary), len(preds_binary))

    # 对整个数据集训练
    # dump(sclf, f'buy_model_{code}_bsp5_20240209_6.joblib')
    # clf_loaded = load(f'buy_model_{code}_bsp5_20240209_6.joblib')
    # preds = clf_loaded.predict_proba(X_test)[:, 1]
    # auc = roc_auc_score(y_test, preds)
    # preds_binary = (preds > 0.6).astype(int)
    # precision = precision_score(y_test, preds_binary, pos_label=1)
    # print(f"test AUC: {auc}")
    # print(f"test precision: {precision}")
    # print('调整阈值后正样本比例： ', sum(preds_binary) / len(preds_binary), sum(preds_binary), len(preds_binary))

    print(X.shape[0])
    print('正负样本比例：', np.unique(y, return_counts=True)[1][0] / np.unique(y, return_counts=True)[1][1])
    print('正样本占所有样本的比例：', np.unique(y, return_counts=True)[1][1] /
          (np.unique(y, return_counts=True)[1][1] + np.unique(y, return_counts=True)[1][0]))
    print('预测正样本比例: ', len(preds[preds > 0.5]) / len(preds))
    print('测试集正样本比例: ', len(y_test[y_test > 0.5]) / len(y_test), len(y_test))
    # feature_importance = bst.get_score(importance_type='weight')
    # sorted_dict = sorted(feature_importance.items(), key=lambda x: x[1])
    # print(sorted_dict)


if __name__ == '__main__':
    train_buy_model(code='MNQmain', begin_time="2019-05-20 00:00:00", end_time="2024-01-18 00:00:00")
