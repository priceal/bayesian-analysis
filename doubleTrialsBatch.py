"""
creates arrays of simulated samples of dwell times from multi-step
analyze a dwell distribution using the double step probability
processes

does a whole batch, add lines to add jobs to batch
"""

batch = []

batch.append(
        { 'tau': [10,50],
         'samples': 100,
         'tau1Limits': [1, 40, 10],
         'tau2Limits': [20, 80, 10],
         'trials': 10,
         'fileName': 'double_10_50_100_10.pkl'} )

batch.append(
        { 'tau': [20,50],
         'samples': 100,
         'tau1Limits': [10, 50, 10],
         'tau2Limits': [20, 80, 10],
         'trials': 20,
         'fileName': 'double_20_50_100_20.pkl'} )

batch.append(
        { 'tau': [50,50],
         'samples': 200,
         'tau1Limits': [30, 80, 10],
         'tau2Limits': [30, 80, 10],
         'trials': 40,
         'fileName': 'double_50_50_200_40.pkl'} )


# universal parameters
maxT = np.Inf      # time window

#########################################################
for batchItem in batch:

    print('now doing batch...\n',batchItem)
    tau = batchItem['tau']
    samples = batchItem['samples']
    tau1Limits = batchItem['tau1Limits']
    tau2Limits = batchItem['tau2Limits']
    trials = batchItem['trials']
    saveFile = batchItem['fileName']

    print("   tau1 resolution = {:.3f}"
      .format((tau1Limits[1]-tau1Limits[0])/(tau1Limits[2]-1)))
    print("   tau2 resolution = {:.3f}"
      .format((tau2Limits[1]-tau2Limits[0])/(tau2Limits[2]-1)))

    params = ba.createParams( tau1Limits, tau2Limits )
    p0 = ba.uniformProb( params )

    startTime = timeit.default_timer()
    results = []
    for n in range(trials):
    
        real = sim.multi_poissonN( tau, samples ) 
        total = real[ real < maxT ]
    
        p1 = ba.multiUpdate( p0, total, params, ba.doubleExp, T=maxT )
        t1,t2 = ba.maximum(params,p1)
        results.append( [min(t1,t2), max(t1,t2)] )

    endTime = timeit.default_timer()
    print("   loop took {:.3f} seconds to execute.".format(endTime-startTime))

    results = np.array(results)
    print("   mean tau1 +/- std = {:.3f} + {:.3f}".format(results[:,0].mean(),results[:,0].std()))
    print("   mean tau2 +/- std = {:.3f} + {:.3f}".format(results[:,1].mean(),results[:,1].std()))

    plt.plot(results[:,0],results[:,1],'.')
    plt.xlim(tau1Limits[0],tau1Limits[1])
    plt.ylim(tau2Limits[0],tau2Limits[1])
    maxLimit = max( tau1Limits[1], tau2Limits[1] )
    plt.plot( [0,maxLimit], [0, maxLimit] )
    with open( saveFile, 'wb' ) as filePointer:
        pickle.dump( (batchItem,results), filePointer )
