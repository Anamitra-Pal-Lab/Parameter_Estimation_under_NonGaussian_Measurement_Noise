

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
#import numdifftools as nd


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



mu_c1 = -0.004
mu_c2 = 0.006
sigma_c1 = 0.05*0.05
sigma_c2 = 0.05*0.05
weight_c1 = 0.4
weight_c2 = 0.6


# mu_c1 = -0.2
# mu_c2 = 0.2
# sigma_c1 = 0.12    
# sigma_c2 = 0.12
# weight_c1 = 0.4
# weight_c2 = 0.6


# mu_c1 = 0
# mu_c2 = 0.005
# sigma_c1 = 0.0015    
# sigma_c2 = 0.0015
# weight_c1 = 0.3
# weight_c2 = 0.7


# mu_c1 = -0.002
# mu_c2 = 0
# mu_c3 = 0.005
# mu_c4 = 0.008

# sigma_c1 = 0.0015   
# sigma_c2 = 0.0015
# sigma_c3 = 0.0015    
# sigma_c4 = 0.0015

# weight_c1 = 0.1
# weight_c2 = 0.2
# weight_c3 = 0.5
# weight_c4 = 0.2

# Mu_vector = [mu_c1, mu_c2,mu_c3, mu_c4]
# Sigma_vector = [sigma_c1, sigma_c2, sigma_c3, sigma_c4]
# weight_vector = [weight_c1, weight_c2, weight_c3, weight_c4]

Mu_vector = [mu_c1, mu_c2]
Sigma_vector = [sigma_c1, sigma_c2]
weight_vector = [weight_c1, weight_c2]

sample_size = 10000
seed_number = 100

# GMM noise in c_e
c_e = GMM_Noise_vector_Generation(sample_size, Mu_vector, Sigma_vector, weight_vector)


plt.hist(c_e, 100)
plt.ylabel('Probability density')
plt.xlabel('Noise')
plt.savefig('GMM_noise.png',bbox_inches='tight', dpi=300)
plt.show()