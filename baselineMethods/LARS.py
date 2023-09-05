# -*- coding: utf-8 -*-


import numpy as np
from sklearn import datasets
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoLars, lars_path
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
    
    # data2 = load_breast_cancer()
    # X2 = data2.data
    # y2 = data2.target
    
    data = data
    X = data.iloc[1::,1::]
    y = data.iloc[0,1::]
    X = X.T
    
    X = np.array(X,dtype=float)
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
    

    alphas, active, coef_path = lars_path(X, y, method='lasso', return_path=True)
    

    active_groups = []
    for idx in active:
        group = group_ids[idx]
        if group not in active_groups:
            active_groups.append(group)
    

    selected_features = [i for i, group in enumerate(group_ids) if group in active_groups[:10]]
    X_selected = X[:, selected_features]
    

    clf = SVC(kernel="linear")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    acc_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='f1')
    
    print(f'5-fold CV Accuracy: {np.mean(acc_scores):.3f}')
    print(f'5-fold CV F1 Score: {np.mean(f1_scores):.3f}')
