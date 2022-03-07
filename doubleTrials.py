"""
creates arrays of simulated samples of dwell times from multi-step
analyze a dwell distribution using the double step probability
processes
"""

# model parameters
tau = [20,50] 
samples = 100
maxT = np.Inf      # time window

# bayesian analysis
tau1Limits = [ 1, 45, 20]  
tau2Limits = [ 20, 90, 20]  
trials = 100

#########################################################
print("tau1 resolution = {:.3f}"
      .format((tau1Limits[1]-tau1Limits[0])/(tau1Limits[2]-1)))
print("tau2 resolution = {:.3f}"
      .format((tau2Limits[1]-tau2Limits[0])/(tau2Limits[2]-1)))

params = ba.createParams( tau1Limits, tau2Limits )
p0 = ba.uniformProb( params )

results = []
for n in range(trials):
    
    real = sim.multi_poissonN( tau, samples ) 
    total = real[ real < maxT ]
    
    p1 = ba.multiUpdate( p0, total, params, ba.doubleExp, T=maxT )
    t1,t2 = ba.maximum(params,p1)
    results.append( [min(t1,t2), max(t1,t2)] )

results = np.array(results)
print("mean tau1 +/- std = {:.3f} + {:.3f}".format(results[:,0].mean(),results[:,0].std()))
print("mean tau2 +/- std = {:.3f} + {:.3f}".format(results[:,1].mean(),results[:,1].std()))

plt.plot(results[:,0],results[:,1],'.')
plt.xlim(tau1Limits[0],tau1Limits[1])
plt.ylim(tau2Limits[0],tau2Limits[1])
maxLimit = max( tau1Limits[1], tau2Limits[1] )
plt.plot( [0,maxLimit], [0, maxLimit] )
