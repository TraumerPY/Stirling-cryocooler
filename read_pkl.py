# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:48:09 2019

@author: Joey(Trumer)
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


#train_writer_path = os.path.join(r'D:\my_file\python_material\lianxi\STIRLING_SAGE_train\tf_stirling\new_mod\analyze\back_analyze' , file_type)
#pkl_path = os.path.join(train_writer_path, 'objs.pkl')
pkl_path1 = r'D:\my_file\python_material\lianxi\STIRLING_SAGE_train\tf_stirling\new_mod\analyze\forward\AdamOptimizer_epoch\102030out1_32_0.002\objs.pkl'
pkl_path2 = r'D:\my_file\python_material\lianxi\STIRLING_SAGE_train\tf_stirling\new_mod\analyze\back\AdamOptimizer_result500\102030out1_70_0.006\objs.pkl'
pkl_path3 = r'D:\my_file\python_material\lianxi\STIRLING_SAGE_train\tf_stirling\new_mod\analyze\back\AdamOptimizer_batchsearch\102030out1_70_0.006\objs.pkl'
pkl_path4 = r'D:\my_file\python_material\lianxi\STIRLING_SAGE_train\tf_stirling\new_mod\analyze\back\AdamOptimizer_batchsearch\102030out1_80_0.006\objs.pkl'
pkl_path5 = r'D:\my_file\python_material\lianxi\STIRLING_SAGE_train\tf_stirling\new_mod\analyze\back\AdamOptimizer_batchsearch\102030out1_90_0.006\objs.pkl'
pkl_path6 = r'D:\my_file\python_material\lianxi\STIRLING_SAGE_train\tf_stirling\new_mod\analyze\back\AdamOptimizer_batchsearch\102030out1_100_0.006\objs.pkl'
pkl_path7 = r'D:\my_file\python_material\lianxi\STIRLING_SAGE_train\tf_stirling\new_mod\analyze\back\AdamOptimizer_batchsearch\102030out1_110_0.006\objs.pkl'


pkl_path_map = [pkl_path1, pkl_path2, pkl_path3, pkl_path4, pkl_path5, pkl_path6, pkl_path7]

def read_pkl_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        val_loss_ii_array,train_loss_array,val_loss_mean, train_loss_mean, smooth_mae_history = pickle.load(f)
        return smooth_mae_history[0:]
        
#for i in pkl_path_map:
#    smooth_mae_history = read_pkl_data(i)
#    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history, label=str(i)[-20:])
#    plt.legend()
#plt.show()
 
       
a = read_pkl_data(pkl_path1)[0:]     
b = read_pkl_data(pkl_path2)[0:]     
c = read_pkl_data(pkl_path3)[0:]     
d = read_pkl_data(pkl_path4)[0:]     
e = read_pkl_data(pkl_path5)[0:]     
f = read_pkl_data(pkl_path6)[0:]     
g = read_pkl_data(pkl_path7)[0:]     
   
   
plt.plot(range(1, len(a) + 1), a, label='a')

plt.legend() 
plt.show()
       
