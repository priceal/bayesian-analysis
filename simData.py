"""
creates arrays of simulated samples of dwell times from multi-step
processes
"""

tau = [30]

samples = 500
maxT = np.Inf

#t0,t1,nt = 0,4000,200

#########################################################
real = sim.multi_poissonN( tau, samples ) 
total = real[ real < maxT ]

plt.figure()
counts, tt, fig = plt.hist( total, bins=100 )
