

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



## The total least square function : Using SVD and LRA   
def Total_Least_Squares_SVD(D,c):
    if len(D.shape) is 1:
        n = 1
        D = D.reshape(len(D),1)
    else:
        n = np.array(D).shape[1] # the number of variable of D
    
    Z = np.vstack((D.T,c.T)).T
    U, s, Vt = la.svd(Z, full_matrices=True)

    V = Vt.T
    VDc = V[:n, n:]
    Vcc = V[n:, n:]
    x_tls = - VDc  / Vcc # total least squares soln
    
    Dtct = - Z.dot(V[:,n:]).dot(V[:,n:].T)
    Dt = Dtct[:,:n] # D error
    c_tls = (D+Dt).dot(x_tls)

    fro_norm = la.norm(Dtct, 'fro')#Frobenius norm
    ### c_tls, D + Dt, x_tls, fro_norm ### Optional return
    return x_tls  
    

def Least_Squares(D,c):
    x_LS = np.linalg.inv(D.T.dot(D)).dot(D.T.dot(c))
    return(x_LS)


def g_x_GMM_hat_gen(x):
    
    g_hat = np.zeros((4,1))
    p=4
    
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



## Function to validate the error
def Net_Abs_Rel_Err_Index(x_Est, x_t):
    Net_err_Index = np.linalg.norm((x_Est-x_t)/(x_t))
    return(Net_err_Index)

    
def noise_function_check(sample_size, Mu_vector, Sigma_vector, weight_vector):
    print(sample_size)
    print(Mu_vector)
    print(Sigma_vector)
    print(weight_vector)
    
    
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



###############################################################################
#####################   Main Starts here   ##########################   
###############################################################################
    


###############################################################################
#####################   1. True data generation   ##########################   
###############################################################################


# Generating the true D values (between -1 and 1)
min_range = -1
max_range= 1

nsamp  =10000

d11_t = np.random.uniform(min_range,max_range,size=nsamp) 
d11_t=d11_t.reshape(1,-1).T
d12_t = np.random.uniform(min_range,max_range,size=nsamp) 
d12_t=d12_t.reshape(1,-1).T
d13_t = np.random.uniform(min_range,max_range,size=nsamp) 
d13_t=d13_t.reshape(1,-1).T
d14_t = np.random.uniform(min_range,max_range,size=nsamp) 
d14_t=d14_t.reshape(1,-1).T

D_t = np.hstack((d11_t, d12_t, d13_t, d14_t)) # Entire D true matrix

## True parameter values
x_t = [1,2,-1,4]
x_t=np.asarray(x_t)
x_t=x_t.reshape(1,-1).T   
x = x_t
c_t = D_t.dot(x_t) ## Calculating the true c parameter values from D and x true values

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

Mu_vector = [mu_c1, mu_c2, mu_c3]
Sigma_vector = [sigma_c1, sigma_c2, sigma_c3]
weight_vector = [weight_c1, weight_c2, weight_c3]

sample_size = len(d11_t)
seed_number = 10

# GMM noise in c_e

#noise_function_check(sample_size, Mu_vector, Sigma_vector, weight_vector)
#%%


###############################################################################
#####################   2. Noisy generation   ##########################   
###############################################################################



c_e = GMM_Noise_vector_Generation(sample_size, Mu_vector, Sigma_vector, weight_vector) ## The function which creates the GMM error vector using the decided parameters
c_e = c_e.reshape(1,-1).T



# # The noise characteristics
# mu_D1 = 0.00
# mu_D2 = 0.05
# #mu_D3 = mu_D1
# #mu_D4 = mu_D1

# sigma_D1 = 0.03*0.5
# sigma_D2 = sigma_D1
# #sigma_D3 = sigma_D1
# #sigma_D4 = sigma_D1

# weight_D1 = 0.3
# weight_D2 = 0.7


# Mu_vector = [mu_D1, mu_D2]
# Sigma_vector = [sigma_D1, sigma_D2]
# weight_vector = [weight_D1, weight_D2]


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



###############################################################################
#####################   3. Noisy data  = True data + Noise   ##########################   
###############################################################################


### Measurement = True + Error
d11 = d11_t + d11_e
d12 = d12_t + d12_e 
d13 = d13_t + d13_e
d14 = d14_t + d14_e 

D = np.hstack((d11, d12, d13, d14))  # Noisy measurement matrix D

c = c_t + c_e



mu_cGj_hat_list =[]
mu_DGj_hat_list =[]

Sigma_cGj_hat_list =[]
Sigma_DGj_hat_list =[]

c_hat = c
D_hat = D





    
###############################################################################
#####################   4a1. Initializing the parmaeter (x)   ##################   
###############################################################################

Max_iter =10  # Setting maximum number of iterations
x_IG_hat = x_t*0.90## Initial guess for the parameter estimation


###############################################################################
#####################   4a2. Decide the number of Gaussian components here ####
###############################################################################
m = len(Mu_vector)
m=3

nGC = m


p = D.shape[1]


# Mu_vector_init = [-0.01,0,0.01]
# Sigma_vector_init = [0.05, 0.05, 0.05]

#%%



Mu_vector_init = []
Sigma_vector_init = []


Mu_vector_init.append(0.0)
Sigma_vector_init.append(0.05) 

for i in range(2,m+1):
    # Mu_vector_init.append(0.01*(-1)**(m)*(m-m%2)/2)
    Mu_vector_init.append(0.01*(-1)**(i)*(i-i%2)/2)
    Sigma_vector_init.append(0.05)
    
Mu_vector_init = np.asarray(Mu_vector_init)
Sigma_vector_init = np.asarray(Sigma_vector_init)
#%%

## Replacing the mean variables calculated from EM results by exact values added

MFac = 1.2 # Initial noise vector control parmaeter - EGLE final results are not impacted by this, the speed can be improved

for j in range(m):
    mu_cGj_hat_list.append(Mu_vector_init[j]*MFac) 
    mu_DGj_hat_list.append(Mu_vector_init[j]*MFac)
    Sigma_cGj_hat_list.append(Sigma_vector_init[j]*MFac) 
    Sigma_DGj_hat_list.append(Sigma_vector_init[j]*MFac) 
    
print(mu_cGj_hat_list) 
  
# for j in range(m):
#     Sigma_cGj_hat_list.append(sigma_c1*MFac)

# for j in range(m):
#     Sigma_DGj_hat_list.append(sigma_D1*MFac)
    
    
k_mu_net_Gaussian_gen = mu_cGj_hat_list[0] - mu_DGj_hat_list[0]*sum(x_IG_hat)
k_sigma_net_Gaussian_gen = Sigma_cGj_hat_list[0] + Sigma_DGj_hat_list[0]*sum(x_IG_hat**2)


Lambda_gen = (1/k_sigma_net_Gaussian_gen)*(D_hat.dot(x_IG_hat)-c_hat - k_mu_net_Gaussian_gen)
c_e_hat_gen = Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0]
D_e_hat_gen = np.hstack(((-x[0]*Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0]),(-x[1]*Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0]),(-x[2]*Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0]),(-x[3]*Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0])))
                     


clf = mixture.GaussianMixture(n_components=nGC)
clf.fit(c_e_hat_gen) 
mu_c = clf.means_
covar_c = clf.covariances_
weight_c = clf.weights_  
sigma_c = np.sqrt(covar_c)
MF1 = clf.predict(c_e_hat_gen) 


c_j_hat_list=[]
D_1j_hat_list=[]
for j in range(m):
    c_j_hat_list.append(c_hat[np.where(MF1==j)])
    D_1j_hat_list.append(D_hat[np.where(MF1==j)])
    
    
mu_cGj_hat_list=[]
mu_DGj_hat_list=[]
Sigma_cGj_hat_list=[]
Sigma_DGj_hat_list =[]
for q1 in range(m):
    mu_cGj_hat_list.append(mu_c[q1])
    mu_DGj_hat_list.append(mu_c[q1])
    
    Sigma_cGj_hat_list.append(sigma_c[q1])
    Sigma_DGj_hat_list.append(sigma_c[q1])
    
    

x_curr_hat = x_IG_hat # Initializing the x_curr




x_hist_np_hat = np.zeros((Max_iter+1,x_IG_hat.shape[0]))
i=0

x_hist_np_hat[0,0] = x_curr_hat[0]
x_hist_np_hat[0,1] = x_curr_hat[1]
x_hist_np_hat[0,2] = x_curr_hat[2]
x_hist_np_hat[0,3] = x_curr_hat[3]    


for i in range(Max_iter):
    print(i)
    x_prev_hat = x_curr_hat
    c_hat = c
    D_hat = D
    x = x_curr_hat
    
    
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
        Lambda_G_curr_hat_gen = (1/k_sigma_net_Gj_hat_gen[q1])*(-D_1j_hat_list[q1].dot(x)+c_j_hat_list[q1]-k_mu_net_Gj_hat_gen[q1])
        Lambda_Gj_hat_gen.append(Lambda_G_curr_hat_gen)
        
        
    c_eGj_hat = []
    D1_eGj_hat = []
    D2_eGj_hat = []
    D3_eGj_hat = []
    D4_eGj_hat = []
    
    
    
    for q1 in range(m):
        c_eGj_hat.append((Sigma_cGj_hat_list[q1]**2)*Lambda_Gj_hat_gen[q1]+mu_cGj_hat_list[q1])
        
    for q1 in range(m):    
        D1_eGj_hat.append(-x[0]*Sigma_DGj_hat_list[q1]**2*Lambda_Gj_hat_gen[q1]+mu_DGj_hat_list[q1])
        D2_eGj_hat.append(-x[1]*Sigma_DGj_hat_list[q1]**2*Lambda_Gj_hat_gen[q1]+mu_DGj_hat_list[q1])
        D3_eGj_hat.append(-x[2]*Sigma_DGj_hat_list[q1]**2*Lambda_Gj_hat_gen[q1]+mu_DGj_hat_list[q1])
        D4_eGj_hat.append(-x[3]*Sigma_DGj_hat_list[q1]**2*Lambda_Gj_hat_gen[q1]+mu_DGj_hat_list[q1])

    c_e_hat_gen_f2 = np.vstack((c_eGj_hat[0], c_eGj_hat[1]))
    c_e_hat_gen = c_e_hat_gen_f2
    for q1 in range(2,m):
        c_e_hat_gen = np.vstack((c_e_hat_gen_f2, c_eGj_hat[q1]))
        c_e_hat_gen_f2 = c_e_hat_gen



    D1_e_hat_gen_f2 = np.vstack((D1_eGj_hat[0], D1_eGj_hat[1]))
    D2_e_hat_gen_f2 = np.vstack((D2_eGj_hat[0], D2_eGj_hat[1]))
    D3_e_hat_gen_f2 = np.vstack((D3_eGj_hat[0], D3_eGj_hat[1]))
    D4_e_hat_gen_f2 = np.vstack((D4_eGj_hat[0], D4_eGj_hat[1]))
    
    D1_e_hat_gen = D1_e_hat_gen_f2
    D2_e_hat_gen = D2_e_hat_gen_f2
    D3_e_hat_gen = D3_e_hat_gen_f2
    D4_e_hat_gen = D4_e_hat_gen_f2
    
    for q1 in range(2,m):
        D1_e_hat_gen = np.vstack((D1_e_hat_gen_f2, D1_eGj_hat[q1]))
        D2_e_hat_gen = np.vstack((D2_e_hat_gen_f2, D2_eGj_hat[q1]))
        D3_e_hat_gen = np.vstack((D3_e_hat_gen_f2, D3_eGj_hat[q1]))
        D4_e_hat_gen = np.vstack((D4_e_hat_gen_f2, D4_eGj_hat[q1]))
        
        D1_e_hat_gen_f2 = D1_e_hat_gen
        D2_e_hat_gen_f2 = D2_e_hat_gen
        D3_e_hat_gen_f2 = D3_e_hat_gen
        D4_e_hat_gen_f2 = D4_e_hat_gen



    nGC = m
    clf = mixture.GaussianMixture(n_components=nGC)
    clf.fit(c_e_hat_gen) 
    mu_c = clf.means_
    covar_c = clf.covariances_
    weight_c = clf.weights_  
    sigma_c = np.sqrt(covar_c)
    MF1 = clf.predict(c_e_hat_gen)
                      
     
    print(mu_c)                   
     
    nGC = m
    clf = mixture.GaussianMixture(n_components=nGC)
    clf.fit(D1_e_hat_gen) 
    mu_Da = clf.means_
    covar_Da = clf.covariances_
    weight_Da = clf.weights_  
    sigma_Da = np.sqrt(covar_Da)
    
    c_j_hat_list=[]
    D_1j_hat_list=[]
    for q1 in range(m):
        c_j_hat_list.append(c_hat[np.where(MF1==q1)])
        D_1j_hat_list.append(D_hat[np.where(MF1==q1)])
        
        
    mu_cGj_hat_list=[]
    mu_DGj_hat_list=[]
    Sigma_cGj_hat_list=[]
    Sigma_DGj_hat_list =[]
    for q1 in range(m):
        mu_cGj_hat_list.append(mu_c[q1])
        mu_DGj_hat_list.append(mu_Da[q1])
        
        Sigma_cGj_hat_list.append(sigma_c[q1])
        Sigma_DGj_hat_list.append(sigma_Da[q1][0][0])
        
        
        
    # g_x_GMM_hat_gen(x)
    g_x_GMM_gen_Jacobi_hat = nd.Jacobian(g_x_GMM_hat_gen)
    Jacobi_at_x_curr = g_x_GMM_gen_Jacobi_hat(x_curr_hat)
    Jacobi_at_x_curr = Jacobi_at_x_curr.reshape(4,4)
    
    x_new_hat = x_curr_hat - np.linalg.pinv(Jacobi_at_x_curr).dot(g_x_GMM_hat_gen(x_curr_hat))
    
    
    # # # print(np.linalg.norm(g_x_GMM_gen_Jacobi_hat(x_new_hat)))
    x_curr_hat = x_new_hat
    
    print(x_curr_hat)
    
    
    
    x_hist_np_hat[i+1,0] = x_curr_hat[0]
    x_hist_np_hat[i+1,1] = x_curr_hat[1]
    x_hist_np_hat[i+1,2] = x_curr_hat[2]
    x_hist_np_hat[i+1,3] = x_curr_hat[3]
    
    
    
    
    # if(np.linalg.norm(x_hist_np_hat[i+1,:]-x_hist_np_hat[i,:])<0.0001):
    #     break
    
    
    

    
x1 = x_hist_np_hat[:,0]
x2 = x_hist_np_hat[:,1]
x3 = x_hist_np_hat[:,2]
x4 = x_hist_np_hat[:,3]


