


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


from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, TREND_TYPE, BSP_TYPE
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver
from Test.config import Config
from candlestick import candlestick


def train_buy_model(code,  begin_time, end_time):
    X, y = load_svmlight_file(f"buy_feature_{code}_20240202_1.libsvm")    # load sample
    print(np.unique(y, return_counts=True))

    nonzero_counts = X.getnnz(axis=0)
    for i, count in enumerate(nonzero_counts):
        if count == 0:
            print(f"第{i}列全为0 ！！！！")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define parameters
    if code == 'MNQmain':
        param = {'max_depth': 30, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'scale_pos_weight': 4.2, 'reg_lambda': 20,  'subsample': 0.9}
    elif code == 'MRTYmain':
        param = {'max_depth': 20, 'eta': 0.05, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'scale_pos_weight': 2.8, 'subsample': 0.9}
    else:
        param = {'max_depth': 20, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'scale_pos_weight': 2.3, 'subsample': 0.9}
        # param = {'max_depth': 20, 'eta': 0.05, 'objective': 'reg:logistic', 'eval_metric': 'rmse'}

    # 调参开始
    # param = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': "hist", 'device': 'cuda'}
    # cv_params = {
    #     'n_estimators': np.linspace(10, 200, 10, dtype=int),
    #     'max_depth': [10, 15, 20, 25, 30, 35, 40],
    #     'eta': [0.05, 0.1, 0.15, 0.2, 0.3],
    #     'min_child_weight': np.linspace(1, 10, 10, dtype=int),
    #     'gamma': np.linspace(0, 1, 10),
    #     'subsample': np.linspace(0, 1, 11),
    #     'colsample_bytree': np.linspace(0, 1, 11),
    #     'reg_lambda': np.linspace(0, 100, 11),
    #     'scale_pos_weight': np.linspace(1, 20, 1)
    # }
    # classifiy_model = xgb.XGBClassifier(**param)  # 注意这里的两个 * 号！
    # gs = GridSearchCV(classifiy_model, cv_params, refit=True, verbose=10, cv=5, scoring='roc_auc')
    # gs.fit(X, y)  # X为训练数据的特征值，y为训练数据的label
    # # 性能测评
    # print("参数的最佳取值：:", gs.best_params_)
    # print("最佳模型得分:", gs.best_score_)
    # 调参结束

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
    bst.save_model(f"buy_model_{code}_20240202_1.json")

    # Evaluate model
    preds = bst.predict(dtest)
    auc = roc_auc_score(y_test, preds)
    preds_binary = (preds > 0.6).astype(int)
    precision = precision_score(y_test, preds_binary, pos_label=1)
    print(f"test AUC: {auc}")
    print(f"test precision: {precision}")
    print('调整阈值后正样本比例： ', sum(preds_binary) / len(preds_binary), sum(preds_binary), len(preds_binary))
    print(X.shape[0])
    print('正负样本比例：', np.unique(y, return_counts=True)[1][0] / np.unique(y, return_counts=True)[1][1])
    print('正样本占所有样本的比例：', np.unique(y, return_counts=True)[1][1] / (np.unique(y, return_counts=True)[1][1]+np.unique(y, return_counts=True)[1][0]))
    print('预测正样本比例: ', len(preds[preds > 0.5]) / len(preds))
    print('测试集正样本比例: ', len(y_test[y_test > 0.5]) / len(y_test), len(y_test))
    feature_importance = bst.get_score(importance_type='weight')
    sorted_dict = sorted(feature_importance.items(), key=lambda x: x[1])
    print(sorted_dict)


if __name__ == '__main__':
    train_buy_model(code='MNQmain', begin_time="2019-05-20 00:00:00", end_time="2024-01-18 00:00:00")
