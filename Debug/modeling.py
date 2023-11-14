#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/4 17:28
# @Author  : rockieluo
# @File    : modeling.py

import os
import pandas as pd
from joblib import dump
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sktime.split import ExpandingWindowSplitter

folder_path = '../Data'
csv_files = []

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path) and file_name.endswith('.csv'):
        csv_files.append(pd.read_csv(file_path, index_col=0).reset_index(drop=True))
df = pd.concat(csv_files, ignore_index=True)

# df = pd.read_csv('../Data/09868.csv', index_col=0).reset_index(drop=True)

# model = make_pipeline(MinMaxScaler(), (max_iter=50000, tol=1e-6))
model = make_pipeline(MinMaxScaler(), LogisticRegression())
# model = make_pipeline(StandardScaler(), MLPClassifier(random_state=1, max_iter=5000))
# model = make_pipeline(GradientBoostingClassifier())
draw_dict = {'prediction': [], 'return_rate': []}


cv = ExpandingWindowSplitter(initial_window=1489, fh=[1])
for train_index, test_index in cv.split(df):
    print('窗口示意：训练集', train_index, ' 测试集：', test_index)
    train_x_split_df = df.drop('return_rate', axis=1).values[train_index]
    train_y_split_df = df['return_rate'][train_index]
    test_x_split_df = df.drop('return_rate', axis=1).values[test_index]
    test_y_split_df = df['return_rate'][test_index]

    model.fit(train_x_split_df, train_y_split_df)
    y_prediction_series = model.predict(test_x_split_df)

    prediction = y_prediction_series[0]
    draw_dict['prediction'].append(prediction)
    draw_dict['return_rate'].extend(list(test_y_split_df))

draw_y_df = pd.DataFrame(draw_dict)
pred_pos_cnt = len(draw_y_df[draw_y_df['prediction'] > 0])
true_pos_cnt = len(draw_y_df[(draw_y_df['prediction'] > 0) & (draw_y_df['return_rate'] > 0)])

# model.fit(df.drop('return_rate', axis=1), df['return_rate'])
# df['prediction'] = model.predict(df.drop('return_rate', axis=1))
# df['error'] = df['return_rate'] - df['prediction']
# print(df)
# pred_pos_cnt = len(df[df['prediction'] > 0])
# true_pos_cnt = len(df[(df['prediction'] > 0) & (df['return_rate'] > 0)])
# print('开仓正确率', true_pos_cnt/pred_pos_cnt*100)

model_filename = '../Model/09868_nn.pkl'
dump(model, model_filename)
