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


def ARE(x1, xt):
    ARE_out = abs((x1-xt)/xt)
    return(ARE_out)

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
    ### c_tls, D + Dt, x_tls, fro_norm 
    return x_tls  
    

def Least_Squares(D,c):# Least sqaures estimation of the linear system Dx =c
    x_LS = np.linalg.inv(D.T.dot(D)).dot(D.T.dot(c))
    return(x_LS)