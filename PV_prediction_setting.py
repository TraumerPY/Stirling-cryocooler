# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:39:08 2019
神经网络相关参数设置
@author: lenovo
forward指PV功回归
"""
import os
class Setting:
    def __init__(self):
        self.map_range = '102030'
        #setting about learning rate
        self.lr_search_flag = 1#serch learning rate(5-cross validation), if the flag is set to 0
        self.fixed_lr = 0.002# fix the learning rate
        self.lr_start = 0.01#the minimun learning rate of search grid
        self.lr_end = 0.03#the maximun learning rate of search grid
        self.lr_interval = 0.01
        self.lr_decay_flag = 1#use learning rate decaying, if the flag is set to 1
        #setting about batch size
        self.batch_search_flag = 1#serch batch size(5-cross validation), if the flag is set to 0
        self.fixed_batch = 32#fix batch size 
        self.batch_start = 50#the minimun batch size of search grid
        self.batch_end = 120#the maximun batch size of search grid
        self.batch_interval = 10
        
        
        self.file_type = self.map_range#data mapped range
        self.input_filename = os.path.join(os.getcwd(), 'data.xlsx')#Stirling cryocooler data file path
        
        self.train_writer_father_path = os.path.join(os.getcwd(), 'PV_prediction_BPNN_train_result')

        self.split_group = 10 #used to partition the testing set
        self.split_batch_size = 2 #Extract the sub-testing set from each partition group of the testing set
        self.epoch_range = 4000#800 3500
        self.input_node = 4

        self.out_num = self.input_node
        self.out_node = 1

        self.node_used = [12, 10]
        self.cut_head = 10
        self.cut_tail = 80

        self.train_writer_path = os.path.join(self.train_writer_father_path , self.file_type)
        
        self.rows =  int(self.file_type[0:2]) * int(self.file_type[2:4]) * int(self.file_type[4:6])
        
        
        self.kfold_flag = 0# run 5_cross_validation, if the flag is set to 1

        self.SGD_flag = 1# use MBGD method, if the flag is set to 1
        
        self.pca_falg = 1# use pca, if the flag is set to 1
        self.pca_dimension = 3
        
        self.back_model_type = 0
        self.limit_test_flag = 1#randomly generate the tesing set, if the flag is set to 1
