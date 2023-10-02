# -*- coding: utf-8 -*-
"""
This program is used to implement Multi-dimensional node evaluator
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from minepy import MINE

class FeatureEvaluator:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def _get_kbest_rank(self, score_func, k=10):
        select_k_best = SelectKBest(score_func=score_func, k='all').fit(self.X, self.y)
        return np.argsort(select_k_best.scores_)

    def _mic(self):
        m = MINE()
        mic_scores = []
        for i in range(self.X.shape[1]):
            m.compute_score(self.X[:, i], self.y)
            mic_scores.append(m.mic())
        return np.array(mic_scores)

    def get_feature_ranking(self):
        # L1 regularization
        l1_rank = self._get_kbest_rank(lambda X, y: np.abs(Lasso(alpha=0.01).fit(X, y).coef_))

        # L2 regularization
        l2_rank = self._get_kbest_rank(lambda X, y: np.abs(Ridge(alpha=0.01).fit(X, y).coef_))

        # Linear Regression
        lr_rank = self._get_kbest_rank(lambda X, y: np.abs(LinearRegression().fit(X, y).coef_))

        # Stability selection - Lasso
        rlog = Lasso(alpha=0.01, selection='random').fit(self.X, self.y)
        stability_rank = np.argsort(np.abs(rlog.coef_))

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(self.X, self.y)
        rf_rank = np.argsort(rf.feature_importances_)

        # Correlation coefficient
        correlation_coef = np.array([np.corrcoef(self.X[:, i], self.y)[0, 1] for i in range(self.X.shape[1])])
        correlation_rank = np.argsort(np.abs(correlation_coef))

        # MIC
        mic_rank = np.argsort(self._mic())

        # Aggregating rankings using RRA
        k = self.X.shape[1]  # number of features
        ranks = [l1_rank, l2_rank, lr_rank, stability_rank, rf_rank, correlation_rank, mic_rank]
        final_scores = np.zeros(k)
        for rank in ranks:
            for i in range(k):
                final_scores[rank[i]] += (i / k) ** 2

        return np.argsort(final_scores)

# # Example usage:
# X, y = make_classification(n_samples=1000, n_features=50, random_state=42)
# evaluator = FeatureEvaluator(X, y)
# final_ranking = evaluator.get_feature_ranking()
# print("Final feature ranking:", final_ranking)
