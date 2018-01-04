"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import RTLearner as rtl
import BagLearner as bl
import sys

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip()[10:].split(',')) for s in inf.readlines()])

#    inf = open(sys.argv[1])
#    data = np.loadtxt(inf) #np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    np.random.shuffle(data)
    print data.shape

    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows
    
    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print trainX
    print trainY

    # create a learner and train it

    #learner = rtl.myRTLearner()
    learner = bl.BagLearner(learner = rtl.RTLearner, kwargs = {"leaf_size":1, "verbose":False}, bags = 1, boost = False, verbose = False)
    learner.addEvidence(trainX, trainY)

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]
