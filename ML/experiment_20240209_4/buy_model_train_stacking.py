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
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import numpy as np
from joblib import dump, load


def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (precision_score(y, yPred),
            roc_auc_score(y, yPred, average='macro'))


def train_buy_model(code, begin_time, end_time):
    X, y = load_svmlight_file(f"buy_feature_{code}_bsp4_20240209_4.libsvm")  # load sample
    # X = X.toarray()
    print(np.unique(y, return_counts=True))

    # clf1 = KNeighborsClassifier(n_neighbors=1)
    # clf2 = RandomForestClassifier(random_state=1, class_weight='balanced')
    # clf3 = make_pipeline(
    #     SimpleImputer(strategy="constant", fill_value=np.nan),
    #     xgb.XGBClassifier(max_depth=30, n_estimators=20, learning_rate=0.05, objective='binary:logistic',
    #                          scale_pos_weight=3.6, reg_lambda=20, subsample=0.9, random_state=1, missing=np.nan)
    # )
    # clf4 = make_pipeline(
    #     SimpleImputer(strategy="constant", fill_value=0),
    #     BalancedRandomForestClassifier(n_estimators=200, random_state=1, sampling_strategy="all", replacement=True,
    #                                    bootstrap=False)
    # )
    clf3 = make_pipeline(
         SimpleImputer(strategy="constant", fill_value=np.nan),
         xgb.XGBClassifier(max_depth=20, n_estimators=40, learning_rate=0.1, objective='binary:logistic',
                             scale_pos_weight=3.6, reg_lambda=20, subsample=0.9, random_state=1, missing=np.nan)
    )
    clf4 = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        BalancedRandomForestClassifier(n_estimators=200, random_state=1, sampling_strategy="all", replacement=True,
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
    # sclf = StackingClassifier(estimators=[('xgbclassifier', clf3), ('balancedrandomforestclassifier', clf4)],
    #                           stack_method="predict_proba",
    #                           final_estimator=lr)

    cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    # for clf, label in zip([clf3, clf4, sclf], ['xgb', 'blancedRF', 'StackingClassifier']):
    for clf, label in zip([sclf], ['StackingClassifier']):
    #     auc_scores = model_selection.cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    #     print("roc_auc: % 0.2f(+ / - % 0.2f) [ % s]" % (auc_scores.mean(), auc_scores.std(), label))
        precision_scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='precision')
        print("precision: % 0.2f(+ / - % 0.2f) [ % s]" % (precision_scores.mean(), precision_scores.std(), label))
        balanced_accuracys = model_selection.cross_val_score(clf, X, y, cv=5, scoring='balanced_accuracy')
        print("balanced_accuracys: % 0.2f(+ / - % 0.2f) [ % s]" % (balanced_accuracys.mean(), balanced_accuracys.std(), label))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    sclf.fit(X_train, y_train)
    # sclf.fit(X, y)
    preds = sclf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    threshold = 0.6
    preds_binary = (preds > threshold).astype(int)
    precision = precision_score(y_test, preds_binary, pos_label=1)
    print(f"test AUC: {auc}")
    print(f"test precision: {precision}")
    print(f'调整阈值{threshold}后正样本比例： ', sum(preds_binary) / len(preds_binary), sum(preds_binary), len(preds_binary))

    # dump(sclf, f'buy_model_{code}_bsp4_20240209_4.joblib')
    clf_loaded = load(f'buy_model_{code}_bsp4_20240209_4.joblib')
    preds = clf_loaded.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    preds_binary = (preds > 0.5).astype(int)
    precision = precision_score(y_test, preds_binary, pos_label=1)
    print(f"test AUC: {auc}")
    print(f"test precision: {precision}")
    print('调整阈值后正样本比例： ', sum(preds_binary) / len(preds_binary), sum(preds_binary), len(preds_binary))

    print(X.shape[0])
    print('正负样本比例：', np.unique(y, return_counts=True)[1][0] / np.unique(y, return_counts=True)[1][1])
    print('正样本占所有样本的比例：', np.unique(y, return_counts=True)[1][1] /
          (np.unique(y, return_counts=True)[1][1] + np.unique(y, return_counts=True)[1][0]))
    print('预测正样本比例: ', len(preds[preds > 0.5]) / len(preds))
    print('测试集正样本比例: ', len(y_test[y_test > 0.5]) / len(y_test), len(y_test))
    # feature_importance = bst.get_score(importance_type='weight')
    # sorted_dict = sorted(feature_importance.items(), key=lambda x: x[1])
    # print(sorted_dict)

    # 计算 ROC 曲线
    # from sklearn.metrics import roc_curve, auc
    # import matplotlib.pyplot as plt
    # fpr, tpr, thresholds = roc_curve(y_test, preds)
    # roc_auc = auc(fpr, tpr)
    #
    # # 绘制 ROC 曲线
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

if __name__ == '__main__':
    train_buy_model(code='MNQmain', begin_time="2019-05-20 00:00:00", end_time="2024-01-18 00:00:00")
