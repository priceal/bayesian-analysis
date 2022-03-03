"""
Functions for performing Bayesian analysis of dwell time samples

v 2022 03 03

"""

import numpy as np
import pylab as plt
import math

##############################################################################
# iterating bayesian updating
##############################################################################
def update(prior,likelihood,evidence=1.0,norm=True):

    """
    performs simple Bayesian updating of prior using liklihood
    If evidence is given, divides by evidence to normalize.
    If not, calculates sum of product to normalize.
    input arrays can have any dims, but must be same.

    Args:
        prior (array): contains prior probabilities. can be any dims.
        liklihood (array): must be same dims as prior
        evidence: optional

    Returns:
        array: the posterior, same dims as prior

    """

    p = np.array(prior)
    l = np.array(likelihood)

    posterior = p * l / evidence
    if norm == True:
        posterior = posterior/posterior.sum()

    return posterior
    
##############################################################################
def multiUpdate(prior, observations, params, prob, 
                T=np.Inf, evidence=1.0, norm=True ):
    """
    performs multiple rounds of updating using an array or list of observations.
    Liklihood is calculated each round using passed conditional probability
    function 'prob'. Prior must be . xlist i prob(x,p) should be 

    Parameters
    ----------
    prior : array
        array of probs for each parameter value listed in params
    observations : array or list
        array or list of observations
    params : list of arrays
        meshdgrid description of parameter space
    prob : function
        a function which returns the probability of getting x given a 
        specific value of p.
    T : TYPE, optional
        time window. The default is np.Inf.
    evidence : TYPE, optional
        DESCRIPTION. The default is 1.0.
    norm : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    array: posterior

    """
    
    current = prior
    for x in observations:
        lklhd = likelihood(x, params, prob, T = T ) 
        current = update(current, lklhd, evidence=evidence, norm=norm)

    return current

##############################################################################
def likelihood( obs, params, prob, T = np.Inf ):
    """
    returns array of likelihoods conditioned on a single observation.

    Args:
        obs (float): single real value of observed data
        params (list or array): meshgrid describing parameter space
        prob (function): conditional probability function

    Returns:
        array: liklihood, same dims as parameter space

    """
    returnValue = np.zeros( params[0].shape )
    
    pList = []
    for p in params:
        pList.append( np.ravel(p) )
    
    pArray = np.array(pList).T    
    returnList = []
#    print(pArray)
    for p in pArray:
#        print('prob( {}, {} ) = {}'.format(obs,p, prob( obs, p, T=T)))
        returnList.append( prob(obs, p, T=T ) )

    returnArray = np.array( returnList ).T
    return np.reshape( returnArray, params[0].shape )

##############################################################################


##############################################################################
# analyzing data and distributions
##############################################################################
def maximum(params, current):
    """
    returns parameter value with maximum probability

    Args:
        params (list or array): DESCRIPTION.
        current (list or array): DESCRIPTION.

    Returns:
        float: DESCRIPTION.

    """
    
    maxloc = np.unravel_index( current.argmax(), current.shape )
    kmax = []
    for k in params:
        kmax.append( k[maxloc] )
    return kmax

##############################################################################
def std(params, current):
    """
    returns standard deviation of params weighted by probability

    Args:
        params (list or array): DESCRIPTION.
        prob (list or array): DESCRIPTION.

    Returns:
        float: DESCRIPTION.

    """
    
    return math.sqrt(np.cov(params, aweights=current))

##############################################################################
def uniformProb( params ):
    """
    a uniform probability density

    Args:
        paramList (list or array): DESCRIPTION.

    Returns:
        array: DESCRIPTION.

    """
    
    dims = params[0].shape
    size = np.size( params[0] )
    return np.ones( dims )/size


##############################################################################
def createParams( *p ):
    """
    creates a list of parameters arrays for use in various functions
    in each array, fastest (rightmost) varying index corresponds to first 
    parameter, next fastest to second, etc.

    Parameters
    ----------
    *p : lists or arrays
        3-tuples containing lower limit, upper limit, number of points

    Returns
    -------
    list of arrays
        meshgrid of parameter values. First parameter range is most slowly
        varying, and last given is fastest varying

    """
   
    klist = []
    for pi in p:
        klist.append( np.linspace(pi[0],pi[1],pi[2])  )
    k = tuple(klist)
    
    return np.meshgrid(*k,indexing='ij')

##############################################################################
# PDFs
##############################################################################
def singleExp(t, tau, T=np.Inf):
    """
     returns value of single exponential distribution, properly weighted using
     the a time window.

    Parameters
    ----------
    t : array with a single value
        time point
    tau : TYPE
        decay time constant
    T : TYPE, optional
        time window. The default is np.Inf.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    tau = tau[0]
    return np.exp(-t/tau)/tau/(1.0-np.exp(-T/tau))

##############################################################################
def doubleExp(t, tau, T=np.Inf):
    """
    returns value of single exponential distribution

    Args:
        t (float): DESCRIPTION.
        tau (float): DESCRIPTION.

    Returns:
        float: DESCRIPTION.

    """
    
    if tau[0] == tau[1]:
        norm = 1.0
        if T < np.Inf:
            norm -= np.exp(-T/tau[0])*(1.0+T/tau[0])
        result = t*np.exp(-t/tau[0])/norm/tau[0]/tau[0]
    else:
        norm = tau[0]*(1.0-np.exp(-T/tau[0]))-tau[1]*(1.0-np.exp(-T/tau[1]))
        result = (np.exp(-t/tau[0]) - np.exp(-t/tau[1]))/norm
  
    return  result

##############################################################################
def baseExp(t, params, T = np.Inf ):
    """
    returns value of single exponential distribution with a slow baseline.
    params = (tau, s, p)
    tau = decay time
    s = baseline decay >> tau
    p = amplitude of baseline << 1

    Args:
        t (float): time point
        params (iterable): tau, s, p
        T (): the time window

    Returns:
        float: DESCRIPTION.

    """
    tau, s, p = params
    windowFactor = 1.0-(1.0-p)*np.exp(-T/tau)-p*np.exp(-T/s)

    result = (1.0-p)*np.exp(-t/tau)/tau + p*np.exp(-t/s)/s
    
    return  result/windowFactor

##############################################################################
def plot(obs,params,prob,T=np.Inf,bins=100):
    """
    plots a PDF

    Args:
        paramList (list or array): DESCRIPTION.
        probs (list or array): DESCRIPTION.

    Returns:
        None.

    """
    N = len(obs)
    counts, tb = np.histogram(obs,bins=bins)
    t = 0.5*(tb[:-1]+tb[1:])
    dt = t[1]-t[0]
    
    y = prob( t, params, T ) * N * dt
    survival = N - np.cumsum(counts)
    y_int = N - np.cumsum(y)
    
    fig, ax = plt.subplots(2,1,sharex='col')
    
    ax[0].hist(obs,bins=bins)
    ax[0].plot(t,y, '-')
    
    ax[1].plot(t, survival, 'o')
    ax[1].plot(t, y_int, '-')
    
    return

##############################################################################
def plot2D( p, k, label = ['1','2'] ):
    
    vmin = k[0].min()
    vmax = k[0].max()
    hmin = k[1].min()
    hmax = k[1].max()
    aspect = (hmax-hmin)/(vmax-vmin)
    
    plt.figure()
    plt.imshow(p, origin = 'lower', aspect = aspect, \
               extent=[hmin,hmax,vmin,vmax])
    plt.xlabel(label[1])
    plt.ylabel(label[0])

##############################################################################    
def plot3D( p, k, label = ['1','2','3'], mode = 'slice'):
    
    vmin = k[0].min()
    vmax = k[0].max()
    hmin = k[1].min()
    hmax = k[1].max()
    dmin = k[2].min()
    dmax = k[2].max()
    
    if mode == 'slice':
        maxloc = np.unravel_index( p.argmax(), p.shape )
        p0 = p[maxloc[0],:,:]
        p1 = p[:,maxloc[1],:]
        p2 = p[:,:,maxloc[2]]
    elif mode == 'project':
        p0 = p.sum(axis=0) # for 1,0
        p1 = p.sum(axis=1) # for 0,1
        p2 = p.sum(axis=2) # for 0,0
    
    aspect01 = (hmax-hmin)/(vmax-vmin)
    aspect11 = (dmax-dmin)/(vmax-vmin)
    aspect00 = (hmax-hmin)/(dmax-dmin)
    
    fig, ax = plt.subplots(2,2,sharex='col',sharey='row')
    fig.suptitle(mode)
    
    ax[1,0].imshow(p2, origin = 'lower', aspect = aspect01, \
               extent=[hmin,hmax,vmin,vmax])
    
    ax[1,1].imshow(p1, origin = 'lower', aspect = aspect11, \
               extent=[dmin,dmax,vmin,vmax])
    
    ax[0,0].imshow(p0.T, origin = 'lower', aspect = aspect00, \
               extent=[hmin,hmax,dmin,dmax])
    
    ax[1,0].set_ylabel(label[0])
    ax[1,0].set_xlabel(label[1])
    ax[0,0].set_ylabel(label[2])
    ax[1,1].set_xlabel(label[2])
    
