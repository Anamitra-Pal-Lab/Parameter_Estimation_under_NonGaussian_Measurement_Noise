
import numpy as np
import matplotlib.pyplot as plt

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


def MTEE_estimation(c1_n, D1_n, x_initial, max_iter, nu_0, tolerance, sigma):
    
    ## c1_n - noisy dependent variable
    ## D1_n - noisy indpendent variable
    ## max_iter - maximum number of iterations
    ## nu_0 - learing rate (base)
    ## tolerance - to exit the loop
    
    epsilon_0 = 1
    nu = nu_0
    
    p = D.shape[1]
    n = D.shape[0]
    Lambda = np.eye(p)
    x_hist = []
    x_curr = x_initial.ravel()
    
    for iter in range(max_iter):
        print('Number of iterations is ' + str(iter))
        print('-------------------------------------')
        
        e_tot = (c1_n - (D1_n.dot(x_curr)).reshape((1,-1)).T)/(np.sqrt(x_curr.T.dot(Lambda).dot(x_curr))+(epsilon_0**(-2)))
        
       
        Jac_init = np.zeros((p,))
        Obj = 0
    
        for i in range(n):
            for j in range(n):
                DeltaEij=e_tot[i]-e_tot[j]
                DeltaXij_Tans=(D1_n[i,:]-D1_n[j,:]).T
                G_Sigma=(1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(DeltaEij*DeltaEij)/(2*sigma*sigma))
               
                T1 = ((DeltaEij*DeltaEij*x_curr)/(((x_curr.T).dot(x_curr))+(epsilon_0^(-2))))
                 
                Jac_init=Jac_init+(G_Sigma*(T1+(DeltaEij*DeltaXij_Tans/(((x_curr.T).dot(x_curr))+(epsilon_0^(-2))))))
                
            
                DeltaEij=e_tot[j]-e_tot[i]
                Obj = Obj + (1/(np.sqrt(2*np.pi)*sigma*np.sqrt(2)))*np.exp(-(DeltaEij*DeltaEij)/(2*sigma*sigma*np.sqrt(2)*np.sqrt(2)))
    
      
        Jac_final= (1/((sigma*sigma)*(n*n)))*Jac_init
    
        Obj_final = (1/(n*n))*Obj
        
        x_new = x_curr+ nu*Jac_final
        

        if((np.linalg.norm(x_new-x_curr)<tolerance) and (iter>5)):
            break
        x_curr = x_new
        x_hist.append(x_curr)
        
        
     
        
       
        sigma = sigma - ((1-0.2)/max_iter)
        #nu = nu - ((0.05-0.005)/max_iter)
        #nu = 752.5
        nu = nu_0
        
        
        
        print(x_curr)
        print("--- %s seconds ---" % (time.time() - start_time))
        
    return(x_hist) ## the array of estimates in each iteration, pick the final element for final estimate
    
# D1_t = np.asarray([[3, 6, 2],[7, 5, 4], [8, 5, 3], [8, 6, 4], [9, 6, 3]])
# x1_t = np.asarray([[4, 3, 6]]).T

# n = D1_t.shape[0]
# p = D1_t.shape[1]

# #c1_e = np.asarray([0.1, 0.1, 0.1, 0.1, 0.1]).T
# c1_e = np.ones((n,1))*0.1
# D1_e = np.ones((n,p))*0.1

# #D1_e = np.asarray([[0.1, 0.1, 0.1],[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

# #Entire_buses_int = Entire_buses.astype(np.int64)

# D1_n = D1_t + D1_e

# c1_t = (D1_n).dot(x1_t)

# c1_n = c1_t+ c1_e



###############################################################################
#####################   Main Starts here   ##########################   
###############################################################################
    
# Generating the true D values (between -1 and 1)
min_range = -1
max_range= 1

nsamp  =100

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
x_t = [1,2,3,4]
x_t=np.asarray(x_t)
x_t=x_t.reshape(1,-1).T   
x = x_t
c_t = D_t.dot(x_t) ## Calculating the true c parameter values from D and x true values

sample_size = c_t.shape[0]

Mu_vector = [0, 0.01]
Sigma_vector = [0.003, 0.003]
weight_vector = [0.3, 0.7]


c_e = GMM_Noise_vector_Generation(sample_size, Mu_vector, Sigma_vector, weight_vector)

D_e_list = []
for j in range(D_t.shape[1]):
    D_e_col =  GMM_Noise_vector_Generation(sample_size, Mu_vector, Sigma_vector, weight_vector)
    D_e_list.append(D_e_col.flatten())
    
D_e = np.asarray(D_e_list).T  
  
D= D_t + D_e
c =c_t + c_e

c1_n = c
D1_n = D
{}


#%%
import time
start_time = time.time()
  
sigma = 1
nu = 5
max_iter = 1000000  
tolerance = 0.0000001    
nu_0= 10
x_initial = 0.95*x_t
 
x_hist = MTEE_estimation(c1_n, D1_n,  x_initial, max_iter, nu_0, tolerance, sigma)

print("--- %s seconds ---" % (time.time() - start_time))

    #%%
    
x_hist_np = np.asarray(x_hist)

end_sample = 90000


plt.plot(x_hist_np[:end_sample,0],'r-', markersize=1)
#plt.legend(loc=4, prop={'size': 10})
plt.xlabel('Iterations', size = 18)
plt.ylabel('$Estimated \:\:\:  Y_1$', size = 18)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.savefig('MTEE-Toy_Y1.png',bbox_inches='tight', dpi=200)
plt.show()


plt.plot(x_hist_np[:end_sample,1],'r-', markersize=1)
#plt.legend(loc=4, prop={'size': 10})
plt.xlabel('Iterations', size = 18)
plt.ylabel('$Estimated \:\:\:  Y_2$', size = 18)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.savefig('MTEE-Toy_Y2.png',bbox_inches='tight', dpi=200)
plt.show()

plt.plot(x_hist_np[:end_sample,2],'r-', markersize=1)
#plt.legend(loc=4, prop={'size': 10})
plt.xlabel('Iterations', size = 18)
plt.ylabel('$Estimated \:\:\:  Y_3$', size = 18)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.savefig('MTEE-Toy_Y3.png',bbox_inches='tight', dpi=200)
plt.show()


plt.plot(x_hist_np[:end_sample,3],'r-', markersize=1)
#plt.legend(loc=4, prop={'size': 10})
plt.xlabel('Iterations', size = 18)
plt.ylabel('$Estimated \:\:\:  Y_4$', size = 18)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.savefig('MTEE-Toy_Y4.png',bbox_inches='tight', dpi=200)
plt.show()


#%%
# plt.plot()
# plt.show()
# plt.plot(x_hist_np[:end_sample,1])
# plt.show()
# plt.plot(x_hist_np[:end_sample,2])
# plt.show()
# plt.plot(x_hist_np[:end_sample,3])
# plt.show()

    
    
    
    
    
    
    #%%
    