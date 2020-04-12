# -*- coding: utf-8 -*-
"""
back
Created on Tue Aug 20 10:07:12 2019
@author: Joey
"""

import tensorflow as tf
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dill
import pickle
import os
from function import *
from Compressor_stroke_prediction_setting import Setting as CS_setting
from PV_prediction_setting import Setting as PV_setting
#load from setting file
Setting = CS_setting()
lr_search_flag = Setting.lr_search_flag
fixed_lr = Setting.fixed_lr
lr_start = Setting.lr_start
lr_end = Setting.lr_end
lr_interval = Setting.lr_interval
lr_decay_flag = Setting.lr_decay_flag

batch_search_flag = Setting.batch_search_flag
fixed_batch = Setting.fixed_batch
batch_start = Setting.batch_start
batch_end = Setting.batch_end
batch_interval = Setting.batch_interval

file_type = Setting.file_type
input_filename = Setting.input_filename
train_writer_path = Setting.train_writer_path

split_group = Setting.split_group
split_batch_size = Setting.split_batch_size
epoch_range = Setting.epoch_range
input_node = Setting.input_node

out_node = Setting.out_node

node_used = Setting.node_used

rows = Setting.rows

kfold_flag = Setting.kfold_flag

SGD_flag = Setting.SGD_flag

pca_falg = Setting.pca_falg
pca_dimension = Setting.pca_dimension

test_split_index_list = []
limit_test_flag =Setting.limit_test_flag

rows_start = 0
rows_end = rows

lr_list = []
if lr_search_flag == 1:
    lr_list.append(fixed_lr)
    lr_map = lr_list
else:
    lr_map = np.arange(lr_start, lr_end, lr_interval)

batch_list = []
if batch_search_flag == 1:
    batch_list.append(fixed_batch)
    batch_map = batch_list
else:
    batch_map = np.arange(batch_start, batch_end, batch_interval)  

#reset the tensorflow    
tf.reset_default_graph()
#create neural network
with tf.name_scope('input'):
    if pca_falg == 1:
        input_node_shape = pca_dimension
    else:
        input_node_shape = input_node

    #define placeholder
    x = tf.placeholder(tf.float32, [None,input_node_shape])
    y = tf.placeholder(tf.float32, [None,out_node])

with tf.name_scope('layer'):
    with tf.name_scope('Weight1'):
        W1 = tf.get_variable("W1", shape=[input_node_shape, node_used[0]], initializer=tf.random_uniform_initializer(minval=0, maxval=6, seed=None, dtype=tf.float32))
        variable_summaries(W1)
    with tf.name_scope('bias1'):
        b1 = tf.Variable(tf.zeros([node_used[0]])+0.0001)
        variable_summaries(b1)
    with tf.name_scope('w1x_plus_b1_activated1'):
        L1_a = tf.matmul(x, W1) + b1
        L1 = tf.nn.relu(L1_a)
        variable_summaries(L1_a, name="L1_a")
        variable_summaries(L1, name="L1")
        L1_dropout = tf.nn.dropout(L1, 1)

    with tf.name_scope('Weight2'):
        W2 = tf.get_variable("W2", shape=[node_used[0], out_node], initializer=tf.random_uniform_initializer(minval=0, maxval=10, seed=None, dtype=tf.float32))
        variable_summaries(W2)
    with tf.name_scope('bias2'):
        b2 = tf.Variable(tf.zeros([out_node]))
        variable_summaries(b2)
    with tf.name_scope('w2x_plus_b2_activated2'):
        prediction = (tf.matmul(L1_dropout,W2) + b2)

with tf.name_scope('loss'):
    l2_reg = tf.contrib.layers.l1_regularizer(0.05)
    l2_loss = tf.contrib.layers.apply_regularization(regularizer=l2_reg, weights_list=[W1, W2])
    delta = tf.square(y - prediction) 
    loss = tf.reduce_mean(delta) 
    Graph_weights = tf.GraphKeys.WEIGHTS

    tf.summary.scalar('loss', loss)


#merge all the summaries togethor
merged = tf.summary.merge_all()
#create saver for the BPNN based model
saver = tf.train.Saver(max_to_keep=80)
#use MBGD and k-fold cross validation to create the BPNN model
def create_model_function(function_partial_train_data,function_val_data,function_n_batch,xiangweicha_change_num,function_num_path,function_batch_size):    
    val_loss_ii_list = []
    train_loss_list = []
    #start the tensorflow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(function_num_path, sess.graph)
        
        if pca_falg == 1:
            val_cut = pca_dimension
        else:
            val_cut = input_node
        val_x = function_val_data[:, 0:val_cut]
        val_y = function_val_data[:, val_cut]
        val_y = val_y.reshape(-1, 1)    

        for epoch in range(epoch_range):
            for batch in range(function_n_batch):
                if SGD_flag == 1:
                    train_batch_mask = np.random.choice(function_partial_train_data.shape[0], function_batch_size)
                else:
                    train_batch_mask = range(function_partial_train_data.shape[0])
    
                train_val_rand = function_partial_train_data[train_batch_mask]
    
                #divide the training data into input parameters part and output part
                train_val_x = train_val_rand[:, 0:val_cut]
                train_val_y = train_val_rand[:, val_cut]
                train_val_y = train_val_y.reshape(-1, 1)

                #run the Optimizer
                train_summary, _ = sess.run([merged, train_step], feed_dict={x:train_val_x,y:train_val_y,})
            val_loss_ii = sess.run(loss, feed_dict={x:val_x, y:val_y})
            val_loss_ii_list.append(val_loss_ii)

            train_prediction = sess.run(prediction, feed_dict={x:train_val_x})              
            train_loss = sess.run(loss, feed_dict={x:train_val_x,y:train_val_y})
            train_writer.add_summary(train_summary, epoch)
            epoch += 1
                
            train_loss_list.append(train_loss)
            
            #curves are generated after 10 epochs to see if the trend 
            if epoch%10 == 0 or epoch == (epoch_range):
                if kfold_flag == 1:
                    print('processing fold:'+str(ii)+' '+'epoch:'+str(epoch)+' val_loss_ii:'+str(val_loss_ii)+' train_loss:'+str(train_loss), ' node:', str(node_used[0]))
                else:
                    print('epoch:'+str(epoch)+' val_loss_ii:'+str(val_loss_ii)+' train_loss:'+str(train_loss))
                plt.figure(52, figsize=(23,8))
                
                plt.subplot(521)
                plt.plot(train_prediction, 'b',label='train_prediction')
                plt.plot(train_val_y, 'r', label='train_real')
                plt.legend()
                
                plt.subplot(522)
                train_prediction_err1 = np.abs(train_prediction - train_val_y) / np.abs(train_val_y)
                plt.plot(train_prediction_err1, 'r',label='train_prediction_RE')
                plt.legend()
    
    
                plt.subplot(524)
                plt.plot(train_loss_list[-300:], 'b',label='train_loss')
                plt.legend()
     
                plt.subplot(523)
                plt.plot(val_loss_ii_list[-300:], 'b',label='val_loss')
                plt.legend()
     
                plt.subplot(526)
                plt.plot(train_loss_list[700:], 'b',label='train_loss')
                plt.legend()
     
                plt.subplot(525)
                plt.plot(val_loss_ii_list[700:], 'b',label='val_loss')
                plt.legend()
                plt_save_path = os.path.join(function_num_path, file_type + '.jpg')
                plt.savefig(plt_save_path)
                
                plt.show() 
                
                print('-----------------------------------------------------------------------------------------------------------------------------')
        if kfold_flag == 0:
            save_path = saver.save(sess, os.path.join(function_num_path,"my_net/save_net.ckpt"))            
    pkl_path = os.path.join(function_num_path, 'objs.pkl')
    return val_loss_ii_list, train_loss_list,pkl_path

    
#load data
df = pd.read_excel(input_filename,sheetname=0)
input_val = df.values    
input_val = input_val[rows_start:rows_end, :]

#create empty array
std_vars_array = np.zeros((1,4)).reshape((1,-1))
PCA_vars_save = np.zeros((1,2))

test_val_all = np.zeros((1,input_node+2))
test_val_all_xiangweicha = np.zeros((1,input_node+2))

for batch_size in batch_map:# search for optimal batch size 
    
    for lr in lr_map:# search for optimal learning rate
        with tf.name_scope('train'):
            if lr_decay_flag == 1:
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(lr, global_step=global_step, decay_steps=100, decay_rate=0.5)
            else:
                learning_rate = lr
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
        
        #calculate the weights and biases  
        var_list_w = [var for var in tf.trainable_variables() if "W" in var.name]    
        var_list_b = [var for var in tf.trainable_variables() if "b" in var.name]
        
        #claculate the gradients  
        gradient_w = tf.train.AdamOptimizer(lr).compute_gradients(loss=loss, var_list=var_list_w)
        gradient_b = tf.train.AdamOptimizer(lr).compute_gradients(loss=loss, var_list=var_list_b)
        
        #add the gradients to summarry
        for idx, itr_g in enumerate(gradient_w):
            variable_summaries(itr_g[0], "layer%d-w-grad"%idx)
        for idx, itr_g in enumerate(gradient_b):
            variable_summaries(itr_g[0], "layer%d-b-grad"%idx)
        
    
        input_val = input_val[:, 0:(input_node+1)]
        num_path = train_writer_path + '_' + str(batch_size) + '_' + str(lr)
        print('batch_size:',batch_size,  'lr:', lr)
        isExists = os.path.exists(num_path)
        if not isExists:
            os.makedirs(num_path)
            
        #divide the data in training set and tesing set
        train_split_index_all = np.arange(rows)
        train_split_index_all = train_split_index_all.tolist()
        if limit_test_flag == 1:
            test_split_index_list = []
            split_k = rows // split_group
            for s in range(split_k):
                test_split_mask_range = s*split_group
                test_split_batch_mask = np.random.choice(split_group, split_batch_size, replace=False)
                            
                test_split_index = test_split_mask_range + test_split_batch_mask
                test_split_index = test_split_index.tolist()
                test_split_out = selectionSort(test_split_index)
                test_split_index_list.extend(test_split_out)
            for t in test_split_index_list:
                train_split_index_all.remove(t)
                
            test_val = input_val[test_split_index_list]    
            train_input = input_val[train_split_index_all]  
        else:
            test_index = np.random.choice(rows, 1200, replace=False)
            test_index = test_index.tolist()
            for t in test_index:
                train_split_index_all.remove(t)
            test_val = input_val[test_index]    
            train_input = input_val[train_split_index_all]  
            
        #save testing data and training data    
        test_val_all = test_val[:, 0:(input_node+1)] 
        test_val_path = os.path.join(num_path, r'test_file.xls')
        write_std_PCA_norl(test_val_all,test_val_path)
        train_val_path = os.path.join(num_path, r'train_file.xls')
        write_std_PCA_norl(train_input,train_val_path)
        all_feature_path = train_writer_path + '_' + str(batch_size) +  '_' + str(lr)
        if pca_falg == 1:
            
            #PCA for the training data
            input_val_PCA_result, _, average, std_, selectVec, pca_proportion = pca(train_input[:,0:input_node],pca_dimension)
            print('pca_proportion',str(pca_proportion))
        else:
            average = None
            std_ = None
            selectVec = None
            pca_proportion = None
            input_val_PCA_result, input_array_max, input_array_min = normalization(train_input[:,0:input_node])
            write_std_PCA_norl(input_array_max,os.path.join(all_feature_path, 'input_array_max'+'.xlsx'))
            write_std_PCA_norl(input_array_min,os.path.join(all_feature_path, 'input_array_min'+'.xlsx'))
            
        np.set_printoptions(threshold=40000)
        train_out_for_concatenate = train_input[:,input_node].reshape((-1,1))
        train_input_concatenate = np.concatenate((input_val_PCA_result,train_out_for_concatenate),axis=1)
        
        #shuffle the training data
        index = [i for i in range(train_input_concatenate.shape[0])]
        np.random.shuffle(index)
        train_val = train_input_concatenate[index]
        train_val_shuffle_path = os.path.join(num_path, r'train_shuffle_file.xls')
        write_std_PCA_norl(train_input[index],train_val_shuffle_path)
        
            
        if SGD_flag == 1:
            n_batch = train_val.shape[0] // batch_size
        else:
            n_batch = 1
        
        if pca_falg == 1:
            std_vars_array_add = np.concatenate((average,std_), axis=0)
            
            isExists = os.path.exists(all_feature_path)
            if not isExists:
                os.makedirs(all_feature_path)
            write_std_PCA_norl(std_vars_array_add,os.path.join(all_feature_path, 'std_vars'+'.xlsx'))#第二位模型次数
            write_std_PCA_norl(selectVec,os.path.join(all_feature_path, 'PCA_vars'+'.xlsx'))
        
        #5-fold cross validation           
        k = 5
        num_val_samples = len(train_val) // k
        #增加for循环循环
        val_loss = []
        val_loss_ii_array = np.zeros((1,epoch_range))
        train_loss_array = np.zeros((1,epoch_range))
        if kfold_flag == 1:
            for ii in range(k):
                val_data = train_val[ii*num_val_samples:(ii+1)*num_val_samples, :]#依次把k份数据中的每一份作为校验数据集
                partial_train_data = np.concatenate([train_val[:ii*num_val_samples], train_val[(ii+1)*num_val_samples:]], axis=0)#把剩下的k-1份作为校验数据集

                val_loss_ii_list, train_loss_list,pkl_path =create_model_function(function_partial_train_data=partial_train_data,function_val_data=val_data,function_n_batch=n_batch,xiangweicha_change_num=0,function_num_path = num_path,function_batch_size=batch_size)
                val_loss_ii_array_temp = np.array(val_loss_ii_list).reshape(1, -1)
                val_loss_ii_array = np.concatenate((val_loss_ii_array, val_loss_ii_array_temp), axis=0)
                train_loss_temp = np.array(train_loss_list).reshape(1, -1)
                train_loss_array = np.concatenate((train_loss_array, train_loss_temp), axis=0)
            val_loss_mean = np.mean(val_loss_ii_array[1:], axis=0)
            train_loss_mean = np.mean(train_loss_array[1:], axis=0)

            smooth_mae_history = mae_plot(epoch_range, val_loss_mean[0:])

            plt.figure(22, figsize=(23,15))
            plt.subplot(421)
            plt.plot(val_loss_mean, 'r',label='val_loss_mean')
            plt.plot(train_loss_mean, 'b',label='train_loss_mean')
            plt.legend()
            plt.subplot(422)
            plt.plot(smooth_mae_history[150:], 'r',label='smooth_mae_history')
            plt.legend()
            plt_save_path = os.path.join(num_path, file_type + '_val' + '.jpg')
            plt.savefig(plt_save_path)
                
            plt.show()
            
            with open(pkl_path, 'wb') as f:
                pickle.dump([val_loss_ii_array,train_loss_array,val_loss_mean, train_loss_mean, smooth_mae_history], f)    
        else:
            val_data = train_val[0:num_val_samples, :]#validation data
            partial_train_data = np.concatenate([train_val[:0], train_val[num_val_samples:]], axis=0)#train data
            _,_,pkl_path = create_model_function(function_partial_train_data=partial_train_data,function_val_data=val_data,function_n_batch=n_batch,xiangweicha_change_num=0,function_num_path = num_path,function_batch_size=batch_size)
            
            