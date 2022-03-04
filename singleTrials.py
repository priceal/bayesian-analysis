


"""
creates arrays of simulated samples of dwell times from multi-step

analyze a dwell distribution using the single step probability

processes
"""

tau = [50] 

samples = 200
maxT = np.Inf


tau1Limits = [ 25, 75, 51]  
trials = 100

#########################################################
print("tau1 resolution = ", (tau1Limits[1]-tau1Limits[0])/(tau1Limits[2]-1))


#########################################################

params = ba.createParams( tau1Limits )
p0 = ba.uniformProb( params )

results = []
for n in range(trials):
    
    real = sim.multi_poissonN( tau, samples ) 
    total = real[ real < maxT ]
    
    p1 = ba.multiUpdate( p0, total, params, ba.singleExp, T=maxT )
    results.append( ba.maximum(params,p1)[0] )

results = np.array(results)
print(results.mean(),results.std())
plt.hist( results, bins=20 )
