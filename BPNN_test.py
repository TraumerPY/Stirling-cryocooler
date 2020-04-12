# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:11:34 2019

@author: lenovo
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from PV_prediction_setting import Setting as PV_setting
from Compressor_stroke_prediction_setting import Setting as CS_setting
from function import *
import matplotlib.pyplot as plt


#load BPNN based model
def read_from_model_function(ckpt_path,input_norl_x,input_node,node_used, back_out_node,pca_falg, pca_dimension):
    if pca_falg == 1:
        input_node_shape = pca_dimension
    else:
        input_node_shape = input_node
    print('ckpt_path', ckpt_path)
    tf.reset_default_graph()
    W1 = tf.Variable(np.arange(input_node_shape*node_used[0]).reshape((input_node_shape,node_used[0])), dtype=tf.float32, name="W1")
    b1 = tf.Variable(tf.zeros([node_used[0]]), dtype=tf.float32, name='layer/bias1/Variable')
    W2 = tf.Variable(np.arange(node_used[0]*back_out_node).reshape((node_used[0],back_out_node)), dtype=tf.float32, name="W2")
    b2 = tf.Variable(tf.zeros([back_out_node]), dtype=tf.float32, name='layer/bias2/Variable')
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        print('ckpt_path',ckpt_path)
        W1_c = sess.run(W1)
        b1_c = sess.run(b1)
        W2_c = sess.run(W2)
        b2_c = sess.run(b2)
    
        x = tf.placeholder(tf.float32, [None,input_node_shape])
        L1_a = tf.matmul(x, W1_c) + b1_c
        L1 = tf.nn.relu(L1_a)
        prediction = (tf.matmul(L1,W2_c) + b2_c)    
        result = sess.run(prediction, feed_dict={x:input_norl_x})
    return result
    
#data preprocessing and call read_from_model_function for BPNN based model
def back_read_from_model(model_df_adr=os.getcwd(), std_df_adr=os.getcwd(), PCA_df_adr=os.getcwd(),model_type=0,type_flag=0, main_flag=0, input_set=None):

    if model_type == 1:
        setting = CS_setting()
    else:
        setting = PV_setting()        
        
    input_node = setting.input_node
    lr = setting.fixed_lr
    batch = setting.fixed_batch
    back_out_node = setting.out_node
    back_input_node = setting.input_node
    back_node_used = setting.node_used
    back_train_writer_path = setting.train_writer_path

    pca_falg = setting.pca_falg
    pca_dimension = setting.pca_dimension
    
    #load proprocessing data 
    if type_flag == 1:
        std_df = pd.read_excel(std_df_adr)
        std_val = std_df.values
        PCA_df = pd.read_excel(PCA_df_adr)
        PCA_val = PCA_df.values
    else:
        min_df = pd.read_excel(std_df_adr)
        min_ = min_df.values
        max_df = pd.read_excel(PCA_df_adr)
        max_ = max_df.values

    ckpt_path = back_train_writer_path + '_' + str(batch) + '_' + str(lr) + r"\my_net\save_net.ckpt"

    
    if main_flag == 1:
        model_val = input_set
    else:
        model_df = pd.read_excel(model_df_adr)#
        model_val = model_df.values
    if type_flag == 1:
        #model with PCA proprocessing
        average = std_val[0:input_node, 0].reshape((1,-1))
        std_ = std_val[input_node:input_node*2, 0].reshape((1,-1))

        input_x_norl = (model_val[:, 0:input_node] - average)/std_
        PCA_vector = PCA_val

        back_input_norl_x_PCA = np.dot(input_x_norl, PCA_vector.T)
        out = read_from_model_function(ckpt_path=ckpt_path,input_norl_x=back_input_norl_x_PCA,input_node=back_input_node,node_used=back_node_used, back_out_node=back_out_node,pca_falg=pca_falg, pca_dimension=pca_dimension)
    else:
        #model with min-max normalization
        back_input_norl_x = (model_val[:, 0:input_node] - min_.reshape(1,-1)) / (max_.reshape(1,-1) - min_.reshape(1,-1))
        out = read_from_model_function(ckpt_path=ckpt_path,input_norl_x=back_input_norl_x,input_node=back_input_node,node_used=back_node_used, back_out_node=back_out_node,pca_falg=pca_falg, pca_dimension=pca_dimension)
        
    if main_flag == 0:
        err = np.abs(out-model_val[:, input_node].reshape((-1,1)))/model_val[:, input_node].reshape((-1,1))
    
        plt.subplot(523)
        plt.plot(err, 'b',label='err2')
        plt.legend()
        plt.show()
        
        return out,err,model_val
    return out

                
        


if __name__ == '__main__':
    plt.figure(52, figsize=(23,30))

    model_df_adr = r'D:\my_file\python_material\lianxi\Stirling_cryocooler_regression_model\Compressor_stroke_prediction_BPNN_train_result\102030_32_0.002\test_file.xls'  
    std_df_adr = r"D:\my_file\python_material\lianxi\Stirling_cryocooler_regression_model\Compressor_stroke_prediction_BPNN_train_result\102030_32_0.002\std_vars.xlsx"#min
    PCA_df_adr = r"D:\my_file\python_material\lianxi\Stirling_cryocooler_regression_model\Compressor_stroke_prediction_BPNN_train_result\102030_32_0.002\PCA_vars.xlsx"#max
    model_type = 1#predict the PV power, when the type is set to 0
    type_flag = 1#use PCA, if the flag is set to 1
    back_train_writer_path = r'D:\my_file\python_material\lianxi\Stirling_cryocooler_regression_model\Compressor_stroke_prediction_BPNN_train_result\102030'
    back_out1,err1,model_val1 = back_read_from_model(model_df_adr=model_df_adr, std_df_adr=std_df_adr, PCA_df_adr=PCA_df_adr, model_type=model_type, type_flag=type_flag)



