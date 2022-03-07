"""
creates arrays of simulated samples of dwell times from multi-step
analyze a dwell distribution using the single step probability
processes
"""

# model parameters
tau = [10] 
samples = 50
maxT = np.Inf      # time window

# bayesian analysis
tau1Limits = [ 1, 20, 50]  
trials = 200

# plotting
bins = 10

#########################################################
print("tau1 resolution = {:.3f}"
      .format((tau1Limits[1]-tau1Limits[0])/(tau1Limits[2]-1)))

params = ba.createParams( tau1Limits )
p0 = ba.uniformProb( params )

results = []
for n in range(trials):
    
    real = sim.multi_poissonN( tau, samples ) 
    total = real[ real < maxT ]
    
    p1 = ba.multiUpdate( p0, total, params, ba.singleExp, T=maxT )
    results.append( ba.maximum(params,p1)[0] )

results = np.array(results)
print("mean +/- std = {:.3f} + {:.3f}".format(results.mean(),results.std()))
plt.hist( results, bins=bins )
plt.xlim(tau1Limits[0],tau1Limits[1])
