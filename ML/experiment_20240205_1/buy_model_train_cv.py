


import json
import math
from typing import Dict, TypedDict
import sys

sys.path.append('/root/chan.py')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold



from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, TREND_TYPE, BSP_TYPE
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver
from Test.config import Config
from candlestick import candlestick


def train_buy_model(code,  begin_time, end_time):
    X, y = load_svmlight_file(f"buy_feature_{code}_20240205_1.libsvm")    # load sample
    print(np.unique(y, return_counts=True))

    nonzero_counts = X.getnnz(axis=0)
    for i, count in enumerate(nonzero_counts):
        if count == 0:
            print(f"第{i}列全为0 ！！！！")

    # Define parameters
    if code == 'MNQmain':
        param = {'max_depth': 30, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'scale_pos_weight': 4.3, 'reg_lambda': 20, 'subsample': 0.9}
        # {'lambda': 0.0042019991850679415, 'subsample': 0.9, 'colsample_bytree': 1.0, 'learning_rate': 0.15,
         # 'max_depth': 48, 'scale_pos_weight': 3.5333778544033967, 'min_child_weight': 4}
    elif code == 'MRTYmain':
        param = {'max_depth': 20, 'eta': 0.05, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'scale_pos_weight': 2.8, 'subsample': 0.9}
    else:
        param = {'max_depth': 20, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'scale_pos_weight': 2.3, 'subsample': 0.9}

    # KFold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=13)

    aucs = []
    precisions = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Train model
        print(param)
        evals_result = {}
        bst = xgb.train(
            param,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=[(dtest, "test")],
            evals_result=evals_result,
            early_stopping_rounds=10,
            verbose_eval=True,
        )
        bst.save_model(f"buy_model_{code}_20240205_1.json")

        # Evaluate model
        preds = bst.predict(dtest)
        preds_binary = (preds > 0.6).astype(int)
        auc = roc_auc_score(y_test, preds)
        precision = precision_score(y_test, preds_binary, pos_label=1)
        precisions.append(precision)
        aucs.append(auc)
        print('cv result: ', precision, auc)

    print("Mean AUC: ", np.mean(aucs))
    print(f"Mean precision: {np.mean(precisions)}")

    print('调整阈值后正样本比例： ', sum(preds_binary) / len(preds_binary), sum(preds_binary), len(preds_binary))
    print(X.shape[0])
    print('正负样本比例：', np.unique(y, return_counts=True)[1][0] / np.unique(y, return_counts=True)[1][1])
    print('正样本占所有样本的比例：', np.unique(y, return_counts=True)[1][1] / (np.unique(y, return_counts=True)[1][1]+np.unique(y, return_counts=True)[1][0]))
    print('预测正样本比例: ', len(preds[preds > 0.5]) / len(preds))
    print('测试集正样本比例: ', len(y_test[y_test > 0.5]) / len(y_test), len(y_test))
    feature_importance = bst.get_score(importance_type='weight')
    sorted_dict = sorted(feature_importance.items(), key=lambda x: x[1])
    print(sorted_dict)
    with open(f'buy_feature_{code}_20240205_1.meta', 'r') as file:
        fea_dict = json.load(file)
    fea_name_importance_dict = {}
    for k, v in fea_dict.items():
        score = feature_importance['f' + str(v)] if 'f' + str(v) in feature_importance else 0
        fea_name_importance_dict[k] = score
    print(sorted(fea_name_importance_dict.items(), key=lambda x: x[1]))



if __name__ == '__main__':
    train_buy_model(code='MNQmain', begin_time="2019-05-20 00:00:00", end_time="2024-01-18 00:00:00")
