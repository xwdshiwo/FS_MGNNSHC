# -*- coding: utf-8 -*-

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
    

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X.T)
    

    for reg in np.logspace(-3, 3, 7):
        gl = GroupLasso(groups=labels, group_reg=reg, l1_reg=0)
        gl.fit(X, y)
        
        if np.sum(gl.sparsity_mask_) > 10:
            break
    

    X_reduced = X[:, gl.sparsity_mask_]
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X_reduced, y)
    

    clf = SVC(kernel='linear')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(clf, X_selected, y, cv=kf, scoring='accuracy')
    f1_scores = cross_val_score(clf, X_selected, y, cv=kf, scoring='f1')
    
    print('filename:\n',name)
    print(f"Optimal number of clusters based on the elbow method: {n_clusters}")
    print(f"Selected Features: {np.where(selector.get_support())[0]}")
    print(f"Average Accuracy: {np.mean(acc_scores):.4f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
