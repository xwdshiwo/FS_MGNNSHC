# -*- coding: utf-8 -*-


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from group_lasso import GroupLasso
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
import pandas as pd

# data = datasets.load_breast_cancer()
# X = data.data
# y = data.target

import warnings

warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from group_lasso import GroupLasso
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def read_all_xlsx_in_directory(directory):
    all_dataframes = {} 
    
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):  
            file_path = os.path.join(directory, filename)
            all_dataframes[filename] = pd.read_excel(file_path,header=None)
    
    return all_dataframes

directory_path = r'C:\Users\wobuhekele\Documents'  
dataframes = read_all_xlsx_in_directory(directory_path)

for name, df in dataframes.items():
    print(f"Data from {name}:")
    data = df
    
    data2 = load_breast_cancer()
    X2 = data2.data
    y2 = data2.target
    
    data = data
    X = data.iloc[1::,1::]
    y = data.iloc[0,1::]
    X = X.T
    
    X = np.array(X)
    y = LabelEncoder().fit_transform(y)
    y = np.array(y,dtype=int)

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X.T)  
        wcss.append(kmeans.inertia_)
    
  
    diffs = np.diff(wcss)

    n_clusters = np.argmin(diffs) + 2
    # n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X.T) 
    group_ids = kmeans.labels_
    

    alpha = 0.1  
    group_lasso = GroupLasso(groups=group_ids, group_reg=alpha, l1_reg=alpha, n_iter=1000, tol=1e-7)
    group_lasso.fit(X, y)
    

    group_norms = [np.linalg.norm(group_lasso.coef_[group_ids == i]) for i in np.unique(group_ids)]
    sorted_group_ids = np.argsort(group_norms)[::-1]
    
    selected_feature_indices = []
    for group_id in sorted_group_ids:
        selected_feature_indices.extend(np.where(group_ids == group_id)[0])
        if len(selected_feature_indices) >= 10:
            break
    

    if len(selected_feature_indices) > 10:
        lasso_selector = Lasso(alpha=alpha)
        lasso_selector.fit(X[:, selected_feature_indices], y)
        sorted_features = np.argsort(np.abs(lasso_selector.coef_))[::-1]
        selected_feature_indices = [selected_feature_indices[i] for i in sorted_features[:10]]
    
    X_selected = X[:, selected_feature_indices]
    

    clf = SVC(kernel="linear")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    acc_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='f1')
    
    print(f'5-fold CV Accuracy: {np.mean(acc_scores):.3f}')
    print(f'5-fold CV F1 Score: {np.mean(f1_scores):.3f}')
