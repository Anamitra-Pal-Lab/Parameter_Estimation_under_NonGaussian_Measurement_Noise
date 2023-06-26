
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


from EGLE_Generate_Dataset import Create_Toy_data_based_on_parameters
from EGLE_Generate_Dataset import Read_PMU_data_true_values
from EGLE_Generate_Dataset import Create_D_matrix_TLPE


from EGLE_Create_Noisy_Measurements import GMM_Noise_vector_Generation
from EGLE_Create_Noisy_Measurements import Create_GMM_noise_vectors_for_D_and_c
from EGLE_Create_Noisy_Measurements import Create_GMM_noise_vectors_for_D_and_c
from EGLE_Create_Noisy_Measurements import Create_TiSy_GMM_noise_for_D_and_c

from EGLE_Estimation import EGLE_Initialization_PS
from EGLE_Estimation import EGLE_Noise_char_Estimation


from EGLE_Performance_comparison import Total_Least_Squares_SVD
from EGLE_Performance_comparison import Least_Squares
from EGLE_Performance_comparison import ARE


# parameter estimation step of EGLE algorithm
def EGLE_Parameter_Estimation_Step(x_curr_hat, secondary_loop_max_iter):
    # EGLE - Parameter Estimation
    g_x_GMM_gen_Jacobi_hat = nd.Jacobian(g_x_GMM_hat_gen)
    Jacobi_at_x_curr = g_x_GMM_gen_Jacobi_hat(x_curr_hat)
    Jacobi_at_x_curr = Jacobi_at_x_curr.reshape(4,4)
    
    for j1 in range(secondary_loop_max_iter):
        x_new_hat = x_curr_hat - np.linalg.pinv(Jacobi_at_x_curr).dot(g_x_GMM_hat_gen(x_curr_hat))
        x_curr_hat = x_new_hat
    
        #print(x_curr_hat)
    # # # print(np.linalg.norm(g_x_GMM_gen_Jacobi_hat(x_new_hat)))
    x_curr_hat = x_new_hat
    return(x_curr_hat)

# the loss function for EGLE
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






# Conversion of Y parameters to rxb values    
def rxb_to_Y1234(r,x,b): # Transformation of actual line parameter values (r,x,b) to the Y parameters of our equations
    z = r+1j*x
    y = (1/z)
    Y1 = y.real
    Y2 = -1*(b+y.imag)
    Y3 = -1*y.real
    Y4 = y.imag
    return(Y1, Y2, Y3, Y4)
    
# Conversion of rxb values to Y parameters 
def rxb_to_Y1234_explicit(r,x,b):
    Y1 = r/(r**2 + x**2)
    Y2 = (x-b*(r**2 + x**2))/(r**2 + x**2)
    Y3 = -1*Y1
    Y4 = -1*x/(r**2 + x**2)
    return(Y1, Y2, Y3, Y4)
    
# Alternative function for Y to rxb transformation
def Y1234_to_rxb(Y):
    Y1 = Y[0]
    Y2 = Y[1]
    Y3 = Y[2]
    Y4 = Y[3]
    
    r =  Y1/(Y1**2+ Y4**2)
    x = -Y4/(Y1**2+ Y4**2)
    b = -Y2-Y4
    
    return(r,x,b)


# Function which actually creates a GMM vector
def Varying_GMM_Error_vector_Creation(sample_size, Mu_vector, Sigma_vector, weight_vector, seed_number): ## The GMM error 
    error = sample_size
    nsamp = sample_size
    #np.random.seed(seed=seed_number)
    data1 = stats.norm.rvs(loc=Mu_vector[0], scale=Sigma_vector[0], size=round(weight_vector[0]*sample_size)) 
    data1 = data1.reshape(-1, 1)
    data2 = stats.norm.rvs(loc=Mu_vector[1], scale=Sigma_vector[1], size=round(weight_vector[1]*sample_size)) 
    data2 = data2.reshape(-1, 1)
#    data3 = stats.norm.rvs(loc=Mu_vector[2], scale=Sigma_vector[2], size=round(weight_vector[2]*sample_size)) 
#    data3 = data3.reshape(-1, 1)
#    GMM_vector = np.vstack((data1, data2, data3)) 
    GMM_vector = np.vstack((data1, data2)) 
    error = GMM_vector
    np.random.shuffle(error)
    #error = np.random.shuffle(error)
    return(error)  

#%%


##################################################################################################
## Seeding to obtain replicable results
seed_value= 350
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


###############################################################################
#####################   Main Starts here   ##########################   
###############################################################################
    
print('starting')
#####################   1. Loading the true power system measurements     ##########################  
    



file_name = r'118bus_Data_VI_ri_eg1.xlsx'


[np_Ipr, np_Ipi, np_Iqr, np_Iqi, np_Vpr, np_Vpi, np_Vqr, np_Vqi] = Read_PMU_data_true_values(file_name)

#%%
## Choose line segment for transmission line parameter estimation
##########################################################################################
branch = 1  ### Number of the branch whose line parameters has to be estimated
bp = branch-1 ### bp = branch number in python, since in python the numbering starts from 0
##########################################################################################

## 
# Calculating the true D measurement matrix of I=VY formulation (D=V when compared with c=Dx)
        
# Loading rxb value for that branch
file_name2 = r'Line_parameter_true_values_eg1.xlsx'

df_LP_118 = pd.read_excel(file_name2, sheet_name = 'Sheet1', header=None)
np_LP_118 = df_LP_118.values

R = np_LP_118[:,2]
X = np_LP_118[:,3]
B = np_LP_118[:,4]

r = R[bp]
x = X[bp]
b = B[bp]/2   



D_t = Create_D_matrix_TLPE(np_Vpr, np_Vpi, np_Vqr, np_Vqi, bp)



# Calcualting Y parameters from rxb
[Y1, Y2, Y3, Y4] = rxb_to_Y1234(r,x,b)

x1_t = Y1
x2_t = Y2
x3_t = Y3
x4_t = Y4
    
x_t = np.vstack((x1_t, x2_t, x3_t, x4_t))  ####### Actual value of the two parameter to be estimated

# True c measurement are calculated from true D measurements and true line parameters
c_t = D_t.dot(x_t)




#####################   2. Noise generation   ##########################  

# Creating the GMM noise vector to be added in the c variable

# The noise characteristics
mu_c1 = 0.00
mu_c2 = 0.01

sigma_c1 = 0.03*0.5
sigma_c2 = 0.03*0.5




weight_c1 = 0.3
weight_c2 = 0.7


Mu_vector = [mu_c1, mu_c2]
Sigma_vector = [sigma_c1, sigma_c2]
weight_vector = [weight_c1, weight_c2]

sample_size = len(c_t)
seed_number = 10



zeros = [0] * round(weight_vector[0]*int(sample_size))
ones = [1] * round(weight_vector[1]*int(sample_size))
# concatenate the two lists
order_comp = zeros + ones
# shuffle the list randomly
random.shuffle(order_comp)
[c_e, d11_e, d12_e, d13_e, d14_e] = Create_TiSy_GMM_noise_for_D_and_c(order_comp, Mu_vector, Sigma_vector, weight_vector)



# # GMM noise in c_e
# c_e = Varying_GMM_Error_vector_Creation(sample_size, Mu_vector, Sigma_vector, weight_vector, seed_number) ## The function which creates the GMM error vector using the decided parameters
# c_e = c_e.reshape(1,-1).T


# sample_size = (D_t[:,0].reshape(1,-1).T).shape[0]
# seed_number = 10
# [d11_e, d12_e, d13_e, d14_e, c_e] = Create_GMM_noise_vectors_for_D_and_c(Mu_vector, Sigma_vector, weight_vector, sample_size)


# #####################   3. Noisy data  = True data + Noise   ########################## 
d11 = (D_t[:,0].reshape(1,-1).T) + d11_e
d12 = (D_t[:,1].reshape(1,-1).T) + d12_e
d13 = (D_t[:,2].reshape(1,-1).T) + d13_e
d14 = (D_t[:,3].reshape(1,-1).T) + d14_e

# d11 = (D_t[:,0].reshape(1,-1).T) + c_e
# d12 = (D_t[:,1].reshape(1,-1).T) + c_e
# d13 = (D_t[:,2].reshape(1,-1).T) + c_e
# d14 = (D_t[:,3].reshape(1,-1).T) + c_e


D = np.hstack((d11, d12, d13, d14))  # Noisy measurement matrix D



c = c_t + c_e



#####################   3. Initializing EGLE Variables   ##########################   
x_IG_hat = x_t*0.90## Initial guess for the parameter estimation
x_curr_hat = x_IG_hat # Initializing the x_curr
Max_iter =500 # Setting maximum number of iterations
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
   
    
    # if(np.linalg.norm(np.asarray(x_hist_list_hat)[i+1,:]-np.asarray(x_hist_list_hat)[i,:])<0.0001):
    #     break
    
    if(i>1):
        x_curr_hat[0] = (x_curr_hat[0]-x_curr_hat[2])/2
        x_curr_hat[2] = -1*x_curr_hat[0]
    
    print(x_curr_hat)
    
    x_curr_list_hat = []
    for m1 in range(p):
        x_curr_list_hat.append(x_curr_hat[m1][0])
    x_hist_list_hat.append(x_curr_list_hat)
    
x_hist_np_hat = np.asarray(x_hist_list_hat)



print('Comparing the performance of EGLE estimation with least sqaures and total least squares: ')
print('\n')


x_TLS = Total_Least_Squares_SVD(D,c)
x_LS = Least_Squares(D,c)


# ## ARE - output Table
# Header_out_list = ['Parameter', 'ARE EGLE (%)', 'ARE LS(%)', 'ARE TLS(%)']
# ARE_output_data = np.hstack((ARE(x_hist_np_hat[i+1,:].reshape(-1,1), x_t)*100,ARE(x_LS, x_t)*100, ARE(x_TLS, x_t)*100))
# ARE_output_data_rounded = np.round(ARE_output_data, 6)
# ARE_output_data_rounded = np.hstack((np.asarray([['x1', 'x2', 'x3', 'x4']]).T, ARE_output_data_rounded))


# # Print headers
# print('{:<10} {:<18} {:<18}  {:<18}'.format(*Header_out_list))
# # Print rows of data
# for row in ARE_output_data_rounded:
#     print('{:<10} {:<18} {:<18}  {:<18}'.format(*row))

# print('\n')

# print('The net ARE of least squares method is:       ' + str(np.round(np.linalg.norm(ARE(x_LS, x_t)*100), 6))+ ' %')
# print('The net ARE of total least squares method is: ' + str(np.round(np.linalg.norm(ARE(x_TLS, x_t)*100),6))+ ' %')
# print('The net ARE of the EGLE method is:            ' + str(np.round(np.linalg.norm(ARE(x_hist_np_hat[i+1,:].reshape(-1,1), x_t)*100),6))+ ' %')




#x_EGLE = x_hist_np_hat[i+1,:].reshape(-1,1)
x_EGLE = np.mean(x_hist_np_hat[i-4:i+1,:], axis=0).reshape(-1,1)


[r_LS, x_reac_LS, b_LS] = Y1234_to_rxb(x_LS) # back calculating the physical line parameter estiamtes from Y estimates for LS

[r_TLS, x_reac_TLS, b_TLS] = Y1234_to_rxb(x_TLS) # back calculating the physical line parameter estiamtes from Y estimates for LS

[r_EGLE, x_reac_EGLE, b_EGLE] = Y1234_to_rxb(x_EGLE) # back calculating the physical line parameter estiamtes from Y estimates for LS

[r_t, x_reac_t, b_t] = Y1234_to_rxb(x_t) # back calculating the physical line parameter estiamtes from Y estimates for true value - for estimation performance analysis




## ARE - line parameters - Table
Header_out_list1 = ['Parameter', 'ARE EGLE (%)', 'ARE LS(%)', 'ARE TLS(%)']

r_AREs = np.hstack(( ARE(r_EGLE, r_t),ARE(r_LS, r_t), ARE(r_TLS, r_t) ))
x_reac_AREs = np.hstack(( ARE(x_reac_EGLE, x_reac_t),ARE(x_reac_LS, x_reac_t), ARE(x_reac_TLS, x_reac_t) ))
b_AREs = np.hstack(( ARE(b_EGLE, b_t),ARE(b_LS, b_t), ARE(b_TLS, b_t) ))

ARE_matrix = np.vstack((r_AREs, x_reac_AREs, b_AREs))
ARE_perc_matrix_rounded = np.round(ARE_matrix*100,6)
ARE_perc_matrix_rounded = np.hstack((np.asarray([['r', 'x', 'b']]).T, ARE_perc_matrix_rounded))

# Print headers
print('{:<10} {:<18} {:<18}  {:<18}'.format(*Header_out_list1))
# Print rows of data
for row in ARE_perc_matrix_rounded:
    print('{:<10} {:<18} {:<18}  {:<18}'.format(*row))

print('\n')

print('The net ARE of least squares method is:       ' + str(np.round(np.linalg.norm(ARE_perc_matrix_rounded[:,2]), 6))+ ' %')
print('The net ARE of total least squares method is: ' + str(np.round(np.linalg.norm(ARE_perc_matrix_rounded[:,3]),6))+ ' %')
print('The net ARE of the EGLE method is:            ' + str(np.round(np.linalg.norm(ARE_perc_matrix_rounded[:,1]),6))+ ' %')






