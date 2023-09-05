# -*- coding: utf-8 -*-
"""
This program is used to implement Multi-dimensional node evaluator
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from minepy import MINE

# Input data
X, y = make_classification(n_samples=1000, n_features=50, random_state=42)

# L1 regularization
l1 = SelectKBest(score_func=lambda X, y: np.abs(Lasso(alpha=0.01).fit(X, y).coef_), k=10).fit(X, y)
l1_rank = np.argsort(l1.scores_)

# L2 regularization
l2 = SelectKBest(score_func=lambda X, y: np.abs(Ridge(alpha=0.01).fit(X, y).coef_), k=10).fit(X, y)
l2_rank = np.argsort(l2.scores_)

# Linear Regression
lr = SelectKBest(score_func=lambda X, y: np.abs(LinearRegression().fit(X, y).coef_), k=10).fit(X, y)
lr_rank = np.argsort(lr.scores_)

# Stability selection - Lasso
rlog = Lasso(alpha=0.01, selection='random').fit(X, y)
stability_rank = np.argsort(np.abs(rlog.coef_))

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
rf_rank = np.argsort(rf.feature_importances_)

# Correlation coefficient
correlation_coef = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
correlation_rank = np.argsort(np.abs(correlation_coef))

# MIC
def mic(X, y):
    m = MINE()
    mic_scores = []
    for i in range(X.shape[1]):
        m.compute_score(X[:, i], y)
        mic_scores.append(m.mic())
    return np.array(mic_scores)

mic_rank = np.argsort(mic(X, y))

# Aggregating rankings using RRA
k = X.shape[1]  # number of features
ranks = [l1_rank, l2_rank, lr_rank, stability_rank, rf_rank, correlation_rank, mic_rank]
final_scores = np.zeros(k)
for rank in ranks:
    for i in range(k):
        final_scores[rank[i]] += (i / k) ** 2

final_ranking = np.argsort(final_scores)

print("Final feature ranking:", final_ranking)
