
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

from EGLE_Generate_Dataset import Create_Toy_data_based_on_parameters
from EGLE_Create_Noisy_Measurements import GMM_Noise_vector_Generation
from EGLE_Create_Noisy_Measurements import Create_GMM_noise_vectors_for_D_and_c
from EGLE_Create_Noisy_Measurements import Create_TiSy_GMM_noise_for_D_and_c

from EGLE_Estimation import EGLE_Initialization_PS
from EGLE_Estimation import EGLE_Noise_char_Estimation


from EGLE_Performance_comparison import Total_Least_Squares_SVD
from EGLE_Performance_comparison import Least_Squares
from EGLE_Performance_comparison import ARE



def EGLE_Parameter_Estimation_Step(x_curr_hat, secondary_loop_max_iter):
    # EGLE - Parameter Estimation
    g_x_GMM_gen_Jacobi_hat = nd.Jacobian(g_x_GMM_hat_gen)
    Jacobi_at_x_curr = g_x_GMM_gen_Jacobi_hat(x_curr_hat)
    Jacobi_at_x_curr = Jacobi_at_x_curr.reshape(4,4)
    
    for j1 in range(secondary_loop_max_iter):
        x_new_hat = x_curr_hat - np.linalg.pinv(Jacobi_at_x_curr).dot(g_x_GMM_hat_gen(x_curr_hat))
        x_curr_hat = x_new_hat
    
    # # # print(np.linalg.norm(g_x_GMM_gen_Jacobi_hat(x_new_hat)))
    x_curr_hat = x_new_hat
    return(x_curr_hat)


def g_x_GMM_hat_gen(x):
    global p
    g_hat = np.zeros((p,1))

    
    k_sigma_net_Gj_hat_gen = []
    for q1 in range(m):
        k_sigma_net_G1_hat_gen = 0
        for q in range(p):
            k_sigma_net_G1_hat_gen = k_sigma_net_G1_hat_gen + (Sigma_DGj_hat_list[q1]**2)*x[q]**2
        k_sigma_net_G1_hat_gen = k_sigma_net_G1_hat_gen + (Sigma_cGj_hat_list[q1]**2) 
        k_sigma_net_Gj_hat_gen.append(k_sigma_net_G1_hat_gen)        

    
    k_mu_net_Gj_hat_gen = []
    for q1 in range(m):
        k_mu_net_G1_hat_gen = 0
        for q in range(p):
            k_mu_net_G1_hat_gen = k_mu_net_G1_hat_gen - (mu_DGj_hat_list[q1])*x[q]
        k_mu_net_G1_hat_gen = k_mu_net_G1_hat_gen + (mu_cGj_hat_list[q1]) 
        k_mu_net_Gj_hat_gen.append(k_mu_net_G1_hat_gen)
    
    Lambda_Gj_hat_gen = []
    Lambda_G_curr_hat_gen = 0
    for q1 in range(m):
        #Lambda_G_curr_hat_gen = (1/k_sigma_net_Gj_hat_gen[q1])*(-D_1j_hat_list[q1].reshape(1,-1).T.dot(x)+c_j_hat_list[q1]-k_mu_net_Gj_hat_gen[q1])
        Lambda_G_curr_hat_gen = (1/k_sigma_net_Gj_hat_gen[q1])*(-D_1j_hat_list[q1].dot(x)+c_j_hat_list[q1]-k_mu_net_Gj_hat_gen[q1])
        Lambda_Gj_hat_gen.append(Lambda_G_curr_hat_gen)

    for q1 in range(p):
        g_hat[q1]=0
        for j in range(m):
            g_hat[q1]=g_hat[q1]+(D_1j_hat_list[j][:,q1].reshape(1,-1).T+x[q1]*Sigma_DGj_hat_list[j]**2*Lambda_Gj_hat_gen[j]-mu_DGj_hat_list[j]).T.dot(Lambda_Gj_hat_gen[j])  
            
    return(g_hat)




###############################################################################
#####################   Main Starts here   ##########################   
###############################################################################

##################################################################################################
## Seeding to obtain replicable results
seed_value= 250
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# # 4. Set the `tensorflow` pseudo-random generator at a fixed value
# import tensorflow as tf
# tf.random.set_seed(seed_value)
##################################################################################################

# Generating the true D values (between -1 and 1)
min_range = -1
max_range= 1
nsamp  = 10000
## True parameter values
x_t = [1,2,-1,4] # give the true parameter as a list


#####################   1. True data generation   ##########################  
[d11_t, d12_t, d13_t, d14_t, c_t, x_t] = Create_Toy_data_based_on_parameters(min_range,max_range, nsamp, x_t)

#####################   2. Noise generation   ##########################  
### Generating error vectors
# Creating the GMM noise vector to be added in the c variable
# The noise characteristics
mu_c1 = -0.00
mu_c2 = 0.05
mu_c3 = 0.08

sigma_c1 = 0.03*0.5
sigma_c2 = 0.03*0.5
sigma_c3 = 0.03*0.5

weight_c1 = 0.3
weight_c2 = 0.4
weight_c3 = 0.3

Mu_vector = [mu_c1, mu_c2, mu_c3 ]
Sigma_vector = [sigma_c1, sigma_c2, sigma_c3]
weight_vector = [weight_c1, weight_c2, weight_c3]

sample_size = nsamp
seed_number = 10
[d11_e, d12_e, d13_e, d14_e, c_e] = Create_GMM_noise_vectors_for_D_and_c(Mu_vector, Sigma_vector, weight_vector, sample_size)


# zeros = [0] * round(weight_vector[0]*int(sample_size))
# ones = [1] * round(weight_vector[1]*int(sample_size))
# # concatenate the two lists
# order_comp = zeros + ones
# # shuffle the list randomly
# random.shuffle(order_comp)
# [c_e, d11_e, d12_e, d13_e, d14_e] = Create_TiSy_GMM_noise_for_D_and_c(order_comp, Mu_vector, Sigma_vector, weight_vector)
#####################   3. Noisy data  = True data + Noise   ##########################   

d11 = d11_t + d11_e
d12 = d12_t + d12_e 
d13 = d13_t + d13_e
d14 = d14_t + d14_e 

D = np.hstack((d11, d12, d13, d14))  # Noisy measurement matrix D

c = c_t + c_e


#####################   4. Initializing EGLE Variables   ##########################   
x_IG_hat = x_t*0.90## Initial guess for the parameter estimation
x_curr_hat = x_IG_hat # Initializing the x_curr
Max_iter =20 # Setting maximum number of iterations
secondary_loop_max_iter = 5 # maximum number of  iterations in the inner loop (in Nnewton's method)
c_hat = c # the noisy measurements are used for estimation
D_hat = D
global p
p = D.shape[1] # number of parameters for current estimation problem


m = len(Mu_vector) # Decide the number of Gaussian components here
nGC = m


mu_cGj_hat_list =[]
mu_DGj_hat_list =[]

Sigma_cGj_hat_list =[]
Sigma_DGj_hat_list =[]

Mu_vector_init = []
Sigma_vector_init = []


[mu_cGj_hat_list,mu_DGj_hat_list, Sigma_cGj_hat_list, Sigma_DGj_hat_list, c_j_hat_list, D_1j_hat_list] = EGLE_Initialization_PS(Mu_vector_init, Sigma_vector_init,mu_cGj_hat_list, mu_DGj_hat_list, Sigma_cGj_hat_list, Sigma_DGj_hat_list, m, x_IG_hat,  c_hat, D_hat)



x_hist_np_hat = np.zeros((Max_iter+1,x_IG_hat.shape[0])) # initializing variable to store the estiamted parameter at every iterations


x_hist_list_hat = []
x_curr_list_hat = []
for m1 in range(p):
    x_curr_list_hat.append(x_curr_hat[m1][0])
x_hist_list_hat.append(x_curr_list_hat)   


for i in range(Max_iter):
    print(i)
    x_prev_hat = x_curr_hat
    c_hat = c
    D_hat = D
    x = x_curr_hat
    
    
    [x, mu_cGj_hat_list, mu_DGj_hat_list, Sigma_cGj_hat_list, Sigma_DGj_hat_list, c_j_hat_list, D_1j_hat_list] = EGLE_Noise_char_Estimation(x, mu_cGj_hat_list, mu_DGj_hat_list, Sigma_cGj_hat_list, Sigma_DGj_hat_list, c_j_hat_list, D_1j_hat_list,  c_hat, D_hat)
        

    x_curr_hat = EGLE_Parameter_Estimation_Step(x_curr_hat, secondary_loop_max_iter)   
   
    print(x_curr_hat)
    
    
    x_curr_list_hat = []
    for m1 in range(p):
        x_curr_list_hat.append(x_curr_hat[m1][0])
    x_hist_list_hat.append(x_curr_list_hat)
    
    if(np.linalg.norm(np.asarray(x_hist_list_hat)[i+1,:]-np.asarray(x_hist_list_hat)[i,:])<0.000001):
        break
    
    
    
x_hist_np_hat = np.asarray(x_hist_list_hat)



print('Comparing the performance of EGLE estimation with least sqaures and total least squares: ')
print('\n')


x_TLS = Total_Least_Squares_SVD(D,c)
x_LS = Least_Squares(D,c)


## ARE - output Table
Header_out_list = ['Parameter', 'ARE EGLE (%)', 'ARE LS(%)', 'ARE TLS(%)']
ARE_output_data = np.hstack((ARE(x_hist_np_hat[i+1,:].reshape(-1,1), x_t)*100,ARE(x_LS, x_t)*100, ARE(x_TLS, x_t)*100))
ARE_output_data_rounded = np.round(ARE_output_data, 6)
ARE_output_data_rounded = np.hstack((np.asarray([['x1', 'x2', 'x3', 'x4']]).T, ARE_output_data_rounded))


# Print headers
print('{:<10} {:<18} {:<18}  {:<18}'.format(*Header_out_list))
# Print rows of data
for row in ARE_output_data_rounded:
    print('{:<10} {:<18} {:<18}  {:<18}'.format(*row))

print('\n')

print('The net ARE of least squares method is:       ' + str(np.round(np.linalg.norm(ARE(x_LS, x_t)*100), 6))+ ' %')
print('The net ARE of total least squares method is: ' + str(np.round(np.linalg.norm(ARE(x_TLS, x_t)*100),6))+ ' %')
print('The net ARE of the EGLE method is:            ' + str(np.round(np.linalg.norm(ARE(x_hist_np_hat[i+1,:].reshape(-1,1), x_t)*100),6))+ ' %')



