######################################################
#######################################################
# 
# functions and simulations for returning samples from
# dwell time distributions that represent single molecule
# reactions.  
#
########################################################
#########################################################

import numpy as np
import random as rd

############################################################
############################################################
# multi_poisson(tau):
# returns a sample wait time for N step process
# 
#   1    2
# A -> B -> C ...
#
# where each step is irreversible and has an exponential 
# distribution of single step wait times with mean wait times tau1 and tau2
#
# input:
# tau = [tau1, tau2, ...]
#
# output:
# wait_time = sample wait time
# 
def multi_poisson(tau):
    "multi_poisson([tau1,tau2...]): returns a sample from an n step process with mean wait times [tau1,tau2,...]"
    
    tau = np.array(tau)
    wait_time = 0.0
    for t in tau:
        wait_time = wait_time - np.log(1.0-rd.random())*t
     
    return wait_time

############################################################
############################################################
# multi_poissonN(tau,N):
# returns N sample wait times for n step process
# 
#   1    2     3
# A -> B -> C - > ...
#
# where each step is irreversible and has an exponential 
# distribution of wait times with mean wait times tau[i]
#
# input:
# tau	[tau1, tau2, ...]
# N 	number of samples
#
# output:
# wait_time = array of N samples wait time
#
def multi_poissonN(tau,N):
    "multi_poissonN([tau1,tau2,...],N): returns N samples from an n step process with mean wait times tau1 and tau2"
  
    tau = np.array(tau)
    wait_time = np.zeros(N)
    for j in range(N):
        for t in tau:
            wait_time[j] = wait_time[j] - np.log(1.0-rd.random())*t
    
    return wait_time





