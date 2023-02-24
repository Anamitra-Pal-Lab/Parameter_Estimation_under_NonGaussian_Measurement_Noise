
## Importing necessary libraries

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math 
import sklearn
from sklearn import mixture
from sklearn import preprocessing
import numpy as np
from scipy import stats
import sys
from sklearn.metrics import accuracy_score
import time
import numpy.linalg as la 
from random import randrange
import numpy as np
import numdifftools as nd



def Create_Toy_data_based_on_parameters(min_range,max_range, nsamp, x_t):
    d11_t = np.random.uniform(min_range,max_range,size=nsamp) 
    d11_t=d11_t.reshape(1,-1).T
    d12_t = np.random.uniform(min_range,max_range,size=nsamp) 
    d12_t=d12_t.reshape(1,-1).T
    d13_t = np.random.uniform(min_range,max_range,size=nsamp) 
    d13_t=d13_t.reshape(1,-1).T
    d14_t = np.random.uniform(min_range,max_range,size=nsamp) 
    d14_t=d14_t.reshape(1,-1).T
    
    D_t = np.hstack((d11_t, d12_t, d13_t, d14_t)) # Entire D true matrix
    
    
    x_t=np.asarray(x_t)
    x_t=x_t.reshape(1,-1).T   
    x = x_t
    c_t = D_t.dot(x_t) ## Calculating the true c parameter values from D and x true values
    
    return(d11_t, d12_t, d13_t, d14_t, c_t, x_t)


def Read_PMU_data_true_values(file_name):# filename includes path and filename, load the input data as .xlsx file of 8 pages - Ipr, Ipi, Iqr, Iqi, Vpr, Vpi, Vqr, and Vqi respectively
    
    df_Ipr = pd.read_excel(file_name, sheet_name = 'Sheet1', header=None)
    print('Scanned')
    df_Ipi = pd.read_excel(file_name, sheet_name = 'Sheet2', header=None)
    print('Scanned')
    df_Iqr = pd.read_excel(file_name, sheet_name = 'Sheet3', header=None)
    print('Scanned')
    df_Iqi = pd.read_excel(file_name, sheet_name = 'Sheet4', header=None)
    print('Scanned')
    
    df_Vpr = pd.read_excel(file_name, sheet_name = 'Sheet5', header=None)
    print('Scanned')
    df_Vpi = pd.read_excel(file_name, sheet_name = 'Sheet6', header=None)
    print('Scanned')
    df_Vqr = pd.read_excel(file_name, sheet_name = 'Sheet7', header=None)
    print('Scanned')
    df_Vqi = pd.read_excel(file_name, sheet_name = 'Sheet8', header=None)
    print('Scanned')

    
    ### Converting the dataframe to numpy arrays for ease of usage
    np_Ipi  =  df_Ipi.values
    np_Iqr  =  df_Iqr.values
    np_Iqi  =  df_Iqi.values
    np_Ipr  =  df_Ipr.values
            
    np_Vpr  =  df_Vpr.values
    np_Vpi  =  df_Vpi.values
    np_Vqr  =  df_Vqr.values
    np_Vqi  =  df_Vqi.values
    
    return(np_Ipr, np_Ipi, np_Iqr, np_Iqi, np_Vpr, np_Vpi, np_Vqr, np_Vqi)

def Create_D_matrix_TLPE(np_Vpr, np_Vpi, np_Vqr, np_Vqi, bp):
    
    d11_t = np_Vpr[bp]
    d12_t = np_Vpi[bp]
    d13_t = np_Vqr[bp]
    d14_t = np_Vqi[bp]
    d11_t = d11_t.reshape((1,-1)).T
    d12_t = d12_t.reshape((1,-1)).T
    d13_t = d13_t.reshape((1,-1)).T
    d14_t = d14_t.reshape((1,-1)).T
    
    nsamp=len(d11_t)
        
    d21_t = d12_t
    d22_t = -d11_t
    d23_t = d14_t
    d24_t = -d13_t
    
    d31_t = d13_t
    d32_t = d14_t
    d33_t = d11_t
    d34_t = d12_t
    
    d41_t = d14_t
    d42_t = -d13_t
    d43_t = d12_t
    d44_t = -d11_t
    
    D1=np.hstack((d11_t, d12_t, d13_t, d14_t))
    D2=np.hstack((d21_t, d22_t, d23_t, d24_t))
    D3=np.hstack((d31_t, d32_t, d33_t, d34_t))
    D4=np.hstack((d41_t, d42_t, d43_t, d44_t))
    
    D_t = np.vstack((D1, D2, D3, D4))

    return(D_t)