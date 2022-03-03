"""
analyze a dwell distribution using the single step probability
"""

tau1Limits = [ 1, 200, 100]  

observations = total

# time window
T = np.Inf

#########################################################
print("tau1 resolution = ", (tau1Limits[1]-tau1Limits[0])/(tau1Limits[2]-1))

params = ba.createParams( tau1Limits )
p0 = ba.uniformProb( params )
p1 = ba.multiUpdate( p0, observations, params, ba.singleExp, T=T )
bestParams= ba.maximum(params,p1)

ba.plot(observations, bestParams, ba.singleExp, T=T, bins=100 )
print("maxmium located at:", bestParams )
#print("standard deviations:",ba.std(taus,p1))
plt.figure()
plt.plot( params[0],p1 )
