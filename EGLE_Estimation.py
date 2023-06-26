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

def EGLE_Initialization(Mu_vector_init, Sigma_vector_init, mu_cGj_hat_list, mu_DGj_hat_list, Sigma_cGj_hat_list, Sigma_DGj_hat_list, m, x_IG_hat,  c_hat, D_hat):
    Mu_vector_init.append(0.0)
    Sigma_vector_init.append(0.05)
    print('Noise initialization starts now')
    print(m)
    for i in range(2,m+1):
        # Mu_vector_init.append(0.01*(-1)**(m)*(m-m%2)/2)
        Mu_vector_init.append(10*(-1)**(i)*(i-i%2)/2)
        Sigma_vector_init.append(0.05)
        
    Mu_vector_init = np.asarray(Mu_vector_init)
    Sigma_vector_init = np.asarray(Sigma_vector_init)
    
    MFac = 1.2 # Initial noise vector control parmaeter - EGLE final results are not impacted by this, the speed can be improved
    
    for j in range(m):
        mu_cGj_hat_list.append(Mu_vector_init[j]*MFac) 
        mu_DGj_hat_list.append(Mu_vector_init[j]*MFac)
        Sigma_cGj_hat_list.append(Sigma_vector_init[j]*MFac) 
        Sigma_DGj_hat_list.append(Sigma_vector_init[j]*MFac) 
        
    print(mu_cGj_hat_list) 
      

    k_mu_net_Gaussian_gen = mu_cGj_hat_list[0] - mu_DGj_hat_list[0]*sum(x_IG_hat)
    k_sigma_net_Gaussian_gen = Sigma_cGj_hat_list[0] + Sigma_DGj_hat_list[0]*sum(x_IG_hat**2)
    
    
    Lambda_gen = (1/k_sigma_net_Gaussian_gen)*(D_hat.dot(x_IG_hat)-c_hat - k_mu_net_Gaussian_gen)
    c_e_hat_gen = Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0]
    D_e_hat_gen = np.hstack(((-x_IG_hat[0]*Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0]),(-x_IG_hat[1]*Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0]),(-x_IG_hat[2]*Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0]),(-x_IG_hat[3]*Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0])))
                         
    
    nGC = m
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
    
    
    return(mu_cGj_hat_list,mu_DGj_hat_list, Sigma_cGj_hat_list, Sigma_DGj_hat_list, c_j_hat_list, D_1j_hat_list )



def EGLE_Initialization_PS(Mu_vector_init, Sigma_vector_init, mu_cGj_hat_list, mu_DGj_hat_list, Sigma_cGj_hat_list, Sigma_DGj_hat_list, m, x_IG_hat,  c_hat, D_hat):
    Mu_vector_init.append(0.0)
    Sigma_vector_init.append(0.005) 
    
    for i in range(2,m+1):
        # Mu_vector_init.append(0.01*(-1)**(m)*(m-m%2)/2)
        Mu_vector_init.append(0.001*(-1)**(i)*(i-i%2)/2)
        Sigma_vector_init.append(0.005)
        
    Mu_vector_init = np.asarray(Mu_vector_init)
    Sigma_vector_init = np.asarray(Sigma_vector_init)
    
    MFac = 1.2 # Initial noise vector control parmaeter - EGLE final results are not impacted by this, the speed can be improved
    
    for j in range(m):
        mu_cGj_hat_list.append(Mu_vector_init[j]*MFac) 
        mu_DGj_hat_list.append(Mu_vector_init[j]*MFac)
        Sigma_cGj_hat_list.append(Sigma_vector_init[j]*MFac) 
        Sigma_DGj_hat_list.append(Sigma_vector_init[j]*MFac) 
        
    print(mu_cGj_hat_list) 
      

    k_mu_net_Gaussian_gen = mu_cGj_hat_list[0] - mu_DGj_hat_list[0]*sum(x_IG_hat)
    k_sigma_net_Gaussian_gen = Sigma_cGj_hat_list[0] + Sigma_DGj_hat_list[0]*sum(x_IG_hat**2)
    
    
    Lambda_gen = (1/k_sigma_net_Gaussian_gen)*(D_hat.dot(x_IG_hat)-c_hat - k_mu_net_Gaussian_gen)
    c_e_hat_gen = Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0]
    D_e_hat_gen = np.hstack(((-x_IG_hat[0]*Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0]),(-x_IG_hat[1]*Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0]),(-x_IG_hat[2]*Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0]),(-x_IG_hat[3]*Sigma_cGj_hat_list[0]*Lambda_gen + mu_cGj_hat_list[0])))
                         
    
    nGC = m
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
    
    
    return(mu_cGj_hat_list,mu_DGj_hat_list, Sigma_cGj_hat_list, Sigma_DGj_hat_list, c_j_hat_list, D_1j_hat_list )

def EGLE_Noise_char_Estimation(x, mu_cGj_hat_list, mu_DGj_hat_list, Sigma_cGj_hat_list, Sigma_DGj_hat_list, c_j_hat_list, D_1j_hat_list,  c_hat, D_hat):
    m = len(mu_cGj_hat_list)
    p = len(x)
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
                      
     
    #print(mu_c)                   
     
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

    #print(mu_c)
    return(x, mu_cGj_hat_list, mu_DGj_hat_list, Sigma_cGj_hat_list, Sigma_DGj_hat_list, c_j_hat_list, D_1j_hat_list)

# def EGLE_Parameter_Estimation_Step(x_curr_hat, secondary_loop_max_iter):
#     # EGLE - Parameter Estimation
#     g_x_GMM_gen_Jacobi_hat = nd.Jacobian(g_x_GMM_hat_gen)
#     Jacobi_at_x_curr = g_x_GMM_gen_Jacobi_hat(x_curr_hat)
#     Jacobi_at_x_curr = Jacobi_at_x_curr.reshape(4,4)
    
#     for j1 in range(secondary_loop_max_iter):
#         x_new_hat = x_curr_hat - np.linalg.pinv(Jacobi_at_x_curr).dot(g_x_GMM_hat_gen(x_curr_hat))
#         x_curr_hat = x_new_hat
    
#     # # # print(np.linalg.norm(g_x_GMM_gen_Jacobi_hat(x_new_hat)))
#     x_curr_hat = x_new_hat
#     return(x_curr_hat)



# def g_x_GMM_hat_gen(x):
    
#     g_hat = np.zeros((p,1))

    
#     k_sigma_net_Gj_hat_gen = []
#     for q1 in range(m):
#         k_sigma_net_G1_hat_gen = 0
#         for q in range(p):
#             k_sigma_net_G1_hat_gen = k_sigma_net_G1_hat_gen + (Sigma_DGj_hat_list[q1]**2)*x[q]**2
#         k_sigma_net_G1_hat_gen = k_sigma_net_G1_hat_gen + (Sigma_cGj_hat_list[q1]**2) 
#         k_sigma_net_Gj_hat_gen.append(k_sigma_net_G1_hat_gen)        

    
#     k_mu_net_Gj_hat_gen = []
#     for q1 in range(m):
#         k_mu_net_G1_hat_gen = 0
#         for q in range(p):
#             k_mu_net_G1_hat_gen = k_mu_net_G1_hat_gen - (mu_DGj_hat_list[q1])*x[q]
#         k_mu_net_G1_hat_gen = k_mu_net_G1_hat_gen + (mu_cGj_hat_list[q1]) 
#         k_mu_net_Gj_hat_gen.append(k_mu_net_G1_hat_gen)
    
#     Lambda_Gj_hat_gen = []
#     Lambda_G_curr_hat_gen = 0
#     for q1 in range(m):
#         #Lambda_G_curr_hat_gen = (1/k_sigma_net_Gj_hat_gen[q1])*(-D_1j_hat_list[q1].reshape(1,-1).T.dot(x)+c_j_hat_list[q1]-k_mu_net_Gj_hat_gen[q1])
#         Lambda_G_curr_hat_gen = (1/k_sigma_net_Gj_hat_gen[q1])*(-D_1j_hat_list[q1].dot(x)+c_j_hat_list[q1]-k_mu_net_Gj_hat_gen[q1])
#         Lambda_Gj_hat_gen.append(Lambda_G_curr_hat_gen)

#     for q1 in range(p):
#         g_hat[q1]=0
#         for j in range(m):
#             g_hat[q1]=g_hat[q1]+(D_1j_hat_list[j][:,q1].reshape(1,-1).T+x[q1]*Sigma_DGj_hat_list[j]**2*Lambda_Gj_hat_gen[j]-mu_DGj_hat_list[j]).T.dot(Lambda_Gj_hat_gen[j])  
            
#     return(g_hat)