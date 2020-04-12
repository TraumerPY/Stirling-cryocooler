# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:11:34 2019

@author: lenovo
"""
import numpy as np
import os
from BPNN_test import *
from sklearn.externals import joblib
        
if __name__ == '__main__':
    machine_learning_type = 2#0:BPNN, 1:SVR, 2:RFR
    Q = 3.226
    expander_stroke_floor = 1
    expander_stroke_ceiling =2
    expander_stroke_interpol_val = 0.05
    
    phase_shift_floor = 50
    phase_shift_ceiling = 100
    phase_shift_interpol_val = 1
    
    input_list = []
    input_array = np.zeros((3,1))
    for i1 in np.arange(expander_stroke_floor,expander_stroke_ceiling+expander_stroke_interpol_val,expander_stroke_interpol_val):
        for i2 in np.arange(phase_shift_floor,phase_shift_ceiling+phase_shift_interpol_val,phase_shift_interpol_val):
            xiangweicha_change_num = i2
            input_list.append(Q)
            input_list.append(i2)
            input_list.append(i1)
            input_list_to_array = np.array(input_list).reshape((-1,1))
            input_array = np.concatenate((input_array,input_list_to_array),axis=1)
            input_list = []
    input_set = input_array.T[1:,:]    
    
    if machine_learning_type == 0:
        

        std_df_adr_CS = os.path.join(os.getcwd(), r'Compressor_stroke_prediction_BPNN_train_result\102030_32_0.002\std_vars.xlsx')#minimun of the training test/variance and average of the training test
        PCA_df_adr_CS = os.path.join(os.getcwd(), r"Compressor_stroke_prediction_BPNN_train_result\102030_32_0.002\PCA_vars.xlsx")#maximun of the training test/pca vector of the training test
        model_type = 1#predict the PV power, when the type is set to 0
        type_flag = 1#use PCA, if the flag is set to 1
        back_train_writer_path_CS = os.path.join(os.getcwd(), r'Compressor_stroke_prediction_BPNN_train_result\102030')
        Compressor_stroke_prediction = back_read_from_model(main_flag=1, input_set=input_set, std_df_adr=std_df_adr_CS, PCA_df_adr=PCA_df_adr_CS, model_type=model_type, type_flag=type_flag)
    
        std_df_adr_PV = os.path.join(os.getcwd(), r"PV_prediction_BPNN_train_result\102030_32_0.002\std_vars.xlsx")#min
        PCA_df_adr_PV = os.path.join(os.getcwd(), r"PV_prediction_BPNN_train_result\102030_32_0.002\PCA_vars.xlsx")#max
        back_train_writer_path_CS = os.path.join(os.getcwd(), r'PV_prediction_BPNN_train_result\102030')
        model_type = 0
        type_flag = 1
        PV_regression_input_set = np.concatenate((input_set, Compressor_stroke_prediction), axis=1)
        PV_prediction = back_read_from_model(main_flag=1, input_set=PV_regression_input_set, std_df_adr=std_df_adr_PV, PCA_df_adr=PCA_df_adr_PV, model_type=model_type, type_flag=type_flag)

    elif machine_learning_type == 1:
        type_flag = 1
        Compressor_stroke_SVR = joblib.load(os.path.join(os.getcwd(), r'SVR_train_result\SVR_Compressor_stroke_model.m'))
        input_set_transformation = transformation(input_set,os.path.join(os.getcwd(), r'SVR_train_result\SVR_Compressor_stroke_train.xlsx'),pca_flag=type_flag)
        Compressor_stroke_prediction = Compressor_stroke_SVR.predict(input_set_transformation)
  
        type_flag = 1
        PV_regression_input_set = np.concatenate((input_set, Compressor_stroke_prediction.reshape((-1,1))), axis=1)
        PV_regression_input_set_transformation = transformation(PV_regression_input_set,os.path.join(os.getcwd(), r'SVR_train_result\SVR_PV_train.xlsx'),pca_flag=type_flag)
        PV_SVR = joblib.load(os.path.join(os.getcwd(), r'SVR_train_result\SVR_PV_model.m'))
        PV_prediction = PV_SVR.predict(PV_regression_input_set_transformation)       


    elif machine_learning_type==2:
                
        type_flag = 1
        Compressor_stroke_RFR = joblib.load(os.path.join(os.getcwd(), r'RFR_train_result\RFR_Compressor_stroke_model.m'))
        input_set_transformation = transformation(input_set,os.path.join(os.getcwd(), r'RFR_train_result\RFR_Compressor_stroke_train.xlsx'),pca_flag=type_flag)
        Compressor_stroke_prediction = Compressor_stroke_RFR.predict(input_set_transformation)
  
        type_flag = 1
        PV_regression_input_set = np.concatenate((input_set, Compressor_stroke_prediction.reshape((-1,1))), axis=1)
        PV_regression_input_set_transformation = transformation(PV_regression_input_set,os.path.join(os.getcwd(), r'RFR_train_result\RFR_PV_train.xlsx'),pca_flag=type_flag)
        PV_RFR = joblib.load(os.path.join(os.getcwd(), r'RFR_train_result\RFR_PV_model.m'))
        PV_prediction = PV_RFR.predict(PV_regression_input_set_transformation)       
        
    

    output = np.concatenate((PV_regression_input_set, PV_prediction.reshape((-1, 1))), axis=1)
    output_reshape_byPV = output[np.lexsort(output.T)]


