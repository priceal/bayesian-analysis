"""
analyze a dwell distribution using the two step probability
"""

tau1Limits = [ 20, 50, 100]
tau2Limits = [ 1, 20, 100]

observations = total

# time window
T = np.Inf

#########################################################
print("tau1 resolution = ", (tau1Limits[1]-tau1Limits[0])/(tau1Limits[2]-1))
print("tau2 resolution = ", (tau2Limits[1]-tau2Limits[0])/(tau2Limits[2]-1))

params = ba.createParams( tau1Limits, tau2Limits )
p0 = ba.uniformProb( params )
p1 = ba.multiUpdate( p0, observations[:100], params, ba.doubleExp )
bestParams= ba.maximum(params,p1)

ba.plot(observations, bestParams, ba.doubleExp )
print("maxmium located at:", bestParams )
#print("standard deviations:",ba.std(taus,p1))
ba.plot2D(p1,params,label=['tau1','tau2'] )


