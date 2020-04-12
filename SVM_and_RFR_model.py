# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:52:06 2020

@author: Joey(Trumer)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import *
from sklearn import metrics
import pandas as pd
from sklearn import decomposition
from sklearn.decomposition import PCA
from function import *
from scipy.stats import pearsonr
from sklearn import preprocessing
import os
from sklearn.externals import joblib

pca_flag = 1#use pca, if the flag is set to 1
k_fold_flag = 0#1:k-fold cross validation
prediction_target = 0#0:predict compressor stroke, 1:predict PV
machine_learning_type = 1#1:SVR, 2:RFR

#load data
df = pd.read_excel(os.path.join(os.getcwd(), 'data.xlsx'),sheetname=0)
input_val = df.values
if prediction_target == 0:
    X = input_val[:, 0:3]
    y = input_val[:, 3]
elif prediction_target == 1:
    X = input_val[:, 0:4]
    y = input_val[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train_oringial = X_train

if pca_flag == 1:
    #pca
    pca = PCA(3)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
else:
    #min-max normalization
    min_max_scaled = preprocessing.MinMaxScaler().fit(X_train)
    X_train = min_max_scaled.transform(X_train)
    X_test = min_max_scaled.transform(X_test)
    
#Initialize SVR and RFR
if prediction_target == 0:
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.7,epsilon=0.01)
    rfr = RandomForestRegressor(random_state=2,n_estimators=182,max_depth=92,min_samples_leaf=1,min_samples_split=2,max_features=2)
elif prediction_target == 1:
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1,epsilon=0.1)
    rfr = RandomForestRegressor(random_state=2,n_estimators=142,max_depth=60,min_samples_leaf=1,min_samples_split=2,max_features=2)

#k-fold cross validation and grid search
if k_fold_flag == 1:
    if machine_learning_type == 1:
        svr_rbf_param_grid = {'C':[1000, 100, 10], 'gamma':[0.01, 0.05, 0.1], 'epsilon':[0.01, 0.001, 0.005]}
        grid = GridSearchCV(SVR(kernel='rbf',epsilon=0.01,gamma=0.05), param_grid=svr_rbf_param_grid, cv=5, scoring='neg_mean_squared_error')
    elif machine_learning_type == 2:
        rfr_param_grid = {'n_estimators':[10,50,100], 'max_features':[2,3], 'max_depth':[5, 10, 20], 'mini_samples_split':[5,10,20]}
        grid = GridSearchCV(rfr, param_grid=rfr_param_grid, cv=5, scoring='neg_mean_squared_error')
    
    grid.fit(X_train, y_train)
    print(grid.grid_scores_)
    print(grid.best_params_)
    print(grid.best_score_)

else:
    #model training
    if machine_learning_type == 1:
        if not os.path.exists(os.path.join(os.getcwd(), r'SVR_train_result')):
            os.makedirs(os.path.join(os.getcwd(), r'SVR_train_result'))
        
        y_train_model = svr_rbf.fit(X_train, y_train)
        if prediction_target == 0:
            write_std_PCA_norl(X_train_oringial, os.path.join(os.getcwd(), r'SVR_train_result\SVR_Compressor_stroke_train.xlsx'))
            joblib.dump(svr_rbf, os.path.join(os.getcwd(), r'SVR_train_result\SVR_Compressor_stroke_model.m'))
        elif prediction_target == 1:
            write_std_PCA_norl(X_train_oringial, os.path.join(os.getcwd(), r'SVR_train_result\SVR_PV_train.xlsx'))
            joblib.dump(svr_rbf, os.path.join(os.getcwd(), r'SVR_train_result\SVR_PV_model.m'))
        y_test_prediction = y_train_model.predict(X_test)
    elif machine_learning_type == 2:
        if not os.path.exists(os.path.join(os.getcwd(), r'RFR_train_result')):
            os.makedirs(os.path.join(os.getcwd(), r'RFR_train_result'))
        y_train_model = rfr.fit(X_train, y_train)
        if prediction_target == 0:
            write_std_PCA_norl(X_train_oringial, os.path.join(os.getcwd(), r'RFR_train_result\RFR_Compressor_stroke_train.xlsx'))
            joblib.dump(rfr, os.path.join(os.getcwd(), r'RFR_train_result\RFR_Compressor_stroke_model.m'))
        elif prediction_target == 1:
            write_std_PCA_norl(X_train_oringial, os.path.join(os.getcwd(), r'RFR_train_result\RFR_PV_train.xlsx'))
            joblib.dump(rfr, os.path.join(os.getcwd(), r'RFR_train_result\RFR_PV_model.m'))
        y_test_prediction = y_train_model.predict(X_test)

    #generalization analysis in the testing set
    MSE = metrics.mean_squared_error(y_test, y_test_prediction)
    MAE = metrics.mean_absolute_error(y_test, y_test_prediction)
    RE = np.abs(y_test-y_test_prediction)/y_test
    r = pearsonr(y_test, y_test_prediction)
    print(MSE,MAE,r)
