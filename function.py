# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 18:31:57 2020

@author: lenovo
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

def write_std_PCA_norl(input_array, adr):
    '''write data to excel'''
    saved_vars = pd.DataFrame(input_array)
    
    writer = pd.ExcelWriter(adr)
    saved_vars.to_excel(writer, 'write_std_PCA_norl', float_format='%.5f')
    writer.save()
    

def variable_summaries(var,name="summaries"):
    '''display setting for tensorboard'''
    with tf.name_scope(str(name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#A function that arranges the elements of an array in order from smallest to largest
def findSmallest(arr):
    smallest = arr[0]
    smallest_index = 0
    for i in range(1, len(arr)):
        smallest = arr[i]
        smallest_index = i
    return smallest_index

def selectionSort(arr):
    newArr = []
    for i in range(len(arr)):
        smallest = findSmallest(arr)
        newArr.append(arr.pop(smallest))
    return newArr


def meanX(dataX):
    '''calculate mean value'''
    return np.mean(dataX,axis=0)

def pca(XMat, k):
    '''PCA for BPNN based model'''
    average = meanX(XMat) 
    std_ = np.std(XMat, axis=0, ddof=1)
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = (XMat - avgs)/std_
    covX = np.cov(data_adjust.T) 
    featValue, featVec=  np.linalg.eig(covX)  
    index = np.argsort(-featValue) 
    finalData = []
    if k > n:
        print ("k must lower than feature number")
    else:
        selectVec = np.matrix(featVec.T[index[:k]]) 
        pca_proportion = np.sum(featValue[index[:k]]) / np.sum(featValue[index])
        finalData = data_adjust * selectVec.T 
        reconData = (finalData * selectVec) + average  
    return finalData, reconData, average, std_, selectVec, pca_proportion

def normalization(input_array):
    input_array_max = np.max(input_array, axis=0)
    input_array_min = np.min(input_array, axis=0)
    
    normalized_output = (input_array-input_array_min) / (input_array_max-input_array_min)
    return normalized_output, input_array_max, input_array_min

def standardization(input_array):
    '''z-score normalization'''
    input_array_mean = np.mean(input_array, axis=0)
    input_array_std = np.std(input_array, axis=0)
    standardized_output = (input_array-input_array_mean) / input_array_std
    return standardized_output

def smooth_curve(points, factor=0.94):
    '''smooth data'''
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
 
def mae_plot(epoch,mae):
    '''draw smoothed data plot'''
    fig=plt.figure()
    ax1=fig.add_subplot(2,2,1)
    x=[i for i in range(1,epoch+1)]
    ax1.plot(x,mae)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation MAE')
    ax2=fig.add_subplot(2,2,2)
    smooth_mae_history = smooth_curve(mae[0:])
    ax2.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.tight_layout()
    plt.show()
    return np.array(smooth_mae_history)

def transformation(data, adr, pca_flag):
    '''preprosessing for SVR and RFR'''
    X_train_df = pd.read_excel(adr)
    X_train = X_train_df.values
    if pca_flag == 1:
        #pca
        pca_ = PCA(3)
        pca_.fit(X_train)
        X_train = pca_.transform(X_train)
        out_ = pca_.transform(data)
    else:
        #min-max normalization
        min_max_scaled = preprocessing.MinMaxScaler().fit(X_train)
        X_train = min_max_scaled.transform(X_train)
        out_ = min_max_scaled.transform(data)
    return out_
