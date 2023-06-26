
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
import random


def Create_TiSy_GMM_noise_for_D_and_c(order_comp, Mu_vector, Sigma_vector, weight_vector):
    # # concatenate the two lists
    # order_comp = zeros + ones
    # # shuffle the list randomly
    # random.shuffle(order_comp)
    # # print the final list
    # print(order_comp)
    
    c_e = []
    
    d11_e = []
    d12_e = []
    d13_e = []
    d14_e = []
    
    for j in range(len(order_comp)):
        if(order_comp[j] == 0):
            c_e.append(stats.norm.rvs(loc=Mu_vector[0], scale=Sigma_vector[0], size=1))
            d11_e.append(stats.norm.rvs(loc=Mu_vector[0], scale=Sigma_vector[0], size=1))
            d12_e.append(stats.norm.rvs(loc=Mu_vector[0], scale=Sigma_vector[0], size=1))
            d13_e.append(stats.norm.rvs(loc=Mu_vector[0], scale=Sigma_vector[0], size=1))
            d14_e.append(stats.norm.rvs(loc=Mu_vector[0], scale=Sigma_vector[0], size=1))
        else:
            c_e.append(stats.norm.rvs(loc=Mu_vector[1], scale=Sigma_vector[1], size=1))
            d11_e.append(stats.norm.rvs(loc=Mu_vector[1], scale=Sigma_vector[1], size=1))
            d12_e.append(stats.norm.rvs(loc=Mu_vector[1], scale=Sigma_vector[1], size=1))
            d13_e.append(stats.norm.rvs(loc=Mu_vector[1], scale=Sigma_vector[1], size=1))
            d14_e.append(stats.norm.rvs(loc=Mu_vector[1], scale=Sigma_vector[1], size=1))
            
    c_e = np.asarray(c_e)
    d11_e = np.asarray(d11_e)
    d12_e = np.asarray(d12_e)
    d13_e = np.asarray(d13_e)
    d14_e = np.asarray(d14_e)
    return(c_e, d11_e, d12_e, d13_e, d14_e)




# Function which actually creates a GMM vector
def GMM_Noise_vector_Generation(sample_size, Mu_vector, Sigma_vector, weight_vector): ## The GMM error 
    print('starting')
    m = len(Mu_vector) # m - number of Gaussian components in the GMM
    GMM_vector_ordered = [] # Initializing the GMM variable
    for i in range(m): # Through each iteration, 
        data_i = stats.norm.rvs(loc=Mu_vector[i], scale=Sigma_vector[i], size=round(weight_vector[i]*sample_size)) 
        data_i_list = np.ndarray.tolist(data_i)
        GMM_vector_ordered = GMM_vector_ordered+(data_i_list) # concatenating the GMM variable with noise data points from the new Gaussian component 

    GMM_vector_ordered_np_array = np.asarray(GMM_vector_ordered) # Converting the list to numpy array
    GMM_vector_ordered_np_array = GMM_vector_ordered_np_array.reshape(-1, 1)
    
    GMM_vector_randomized = GMM_vector_ordered_np_array 
    np.random.shuffle(GMM_vector_randomized) # Randomizing the noise vector
    GMM_vector_output =GMM_vector_randomized # Naming the output
    return(GMM_vector_output)  


def Create_GMM_noise_vectors_for_D_and_c(Mu_vector, Sigma_vector, weight_vector, sample_size):
    c_e = GMM_Noise_vector_Generation(sample_size, Mu_vector, Sigma_vector, weight_vector) ## The function which creates the GMM error vector using the decided parameters
    c_e = c_e.reshape(1,-1).T
    # GMM Noise in d11 
    d11_e =GMM_Noise_vector_Generation(sample_size, Mu_vector, Sigma_vector, weight_vector) ## The function which creates the GMM error vector using the decided parameters
    d11_e = d11_e.reshape(1,-1).T
    # GMM Noise in d12 
    d12_e =GMM_Noise_vector_Generation(sample_size, Mu_vector, Sigma_vector, weight_vector) ## The function which creates the GMM error vector using the decided parameters
    d12_e = d12_e.reshape(1,-1).T
    d13_e = GMM_Noise_vector_Generation(sample_size, Mu_vector, Sigma_vector, weight_vector) ## The function which creates the GMM error vector using the decided parameters
    d13_e = d13_e.reshape(1,-1).T
    # GMM Noise in d12 
    d14_e = GMM_Noise_vector_Generation(sample_size, Mu_vector, Sigma_vector, weight_vector) ## The function which creates the GMM error vector using the decided parameters
    d14_e = d14_e.reshape(1,-1).T
    # Concatenating the d11_e and d12_e to create D_e matrix
    D_e = np.hstack((d11_e, d12_e, d13_e, d14_e)) 
    return(d11_e, d12_e, d13_e, d14_e, c_e)