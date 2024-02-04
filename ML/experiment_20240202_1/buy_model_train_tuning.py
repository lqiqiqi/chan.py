


import json
import math
from typing import Dict, TypedDict
import sys

sys.path.append('/root/chan.py')

import optuna
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
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.1, random_state=13)

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define parameters
    if code == 'MNQmain':
        param = {'max_depth': 30, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'scale_pos_weight': 1.9, 'reg_lambda': 20}
    elif code == 'MRTYmain':
        param = {'max_depth': 20, 'eta': 0.05, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'scale_pos_weight': 2.8, 'subsample': 0.9}
    else:
        param = {'max_depth': 20, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'scale_pos_weight': 2.3, 'subsample': 0.9}
        # param = {'max_depth': 20, 'eta': 0.05, 'objective': 'reg:logistic', 'eval_metric': 'rmse'}

    # 调参开始
    # optuna调参+训练模型
    def objectives(trial):
        param = {
            'eval_metric': 'auc',
            'objective': 'binary:logistic',
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'subsample': trial.suggest_categorical('subsample', [0.4, 0.6, 0.8, 0.9, 1.0]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.5, 0.7, 0.9, 1.0]),
            'eta': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]),
            # "num_boost_round": trial.suggest_int('n_estimators', 30, 1000),
            'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
            'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }
        evals_result = {}
        model = xgb.train(
                param,
                num_boost_round=1000,  # trial.suggest_int('num_boost_round', 10, 1000)
                dtrain=dtrain,
                evals=[(deval, "eval")],
                evals_result=evals_result,
                early_stopping_rounds=10,
                verbose_eval=True,
            )
        preds = model.predict(deval)
        auc = roc_auc_score(y_eval, preds)
        return auc

    studyxgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=0))
    studyxgb.optimize(objectives, n_trials=1000)

    trial = studyxgb.best_trial
    params_best = dict(trial.params.items())
    print(params_best)
    # params_best['random_seed'] = 0

    # bst = xgb.XGBClassifier(**params_best, enable_categorical=True)  # xgb.XGBRegressor(**param, enable_categorical=True)

    # 打印最佳参数
    print('study.best_params score:', studyxgb.best_trial.value)
    print('Number of finished trials:', len(studyxgb.trials))
    print('Best trial:', studyxgb.best_trial.params)
    print('study.best_params:', studyxgb.best_params)
    # 调参结束
    #
    # # Train model
    # print(param)
    params_best['eval_metric'] = 'auc'
    evals_result = {}
    bst = xgb.train(
        params_best,
        dtrain=dtrain,
        num_boost_round=1000,  # trial.suggest_int('num_boost_round', 10, 1000)
        evals=[(dtest, "test")],
        evals_result=evals_result,
        early_stopping_rounds=10,
        verbose_eval=True,
    )
    # bst.save_model(f"buy_model_{code}_20240202_1.json")
    # Evaluate model
    preds = bst.predict(dtest)
    auc = roc_auc_score(y_test, preds)
    preds_binary = (preds > 0.5).astype(int)
    precision = precision_score(y_test, preds_binary, pos_label=1)
    print(f"test AUC: {auc}")
    print(f"test precision: {precision}")
    print('调整阈值后正样本比例： ', sum(preds_binary) / len(preds_binary), sum(preds_binary), len(preds_binary))
    print(X.shape[0])
    print('正负样本比例：', np.unique(y, return_counts=True)[1][0] / np.unique(y, return_counts=True)[1][1])
    print('正样本占所有样本的比例：', np.unique(y, return_counts=True)[1][1] / (np.unique(y, return_counts=True)[1][1]+np.unique(y, return_counts=True)[1][0]))
    print('预测正样本比例: ', len(preds[preds > 0.5]) / len(preds))
    print('测试集正样本比例: ', len(y_test[y_test > 0.5]) / len(y_test))
    feature_importance = bst.get_score(importance_type='weight')
    sorted_dict = sorted(feature_importance.items(), key=lambda x: x[1])
    print(sorted_dict)


if __name__ == '__main__':
    train_buy_model(code='MNQmain', begin_time="2019-05-20 00:00:00", end_time="2024-01-18 00:00:00")
