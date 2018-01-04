"""
An implementation of Regression Trees.  (c) 2017 Emmanuel Naziga
"""

import numpy as np
import random
from scipy import stats

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose):
	self.learner = learner
	self.bags = bags
	self.kwargs = kwargs
	self.boost = boost
	self.verbose = verbose

	self.learners = []
	
	for i in range(0,self.bags):
    		self.learners.append(self.learner(**kwargs))

    def author(self):
        return 'enaziga3' 

    def addEvidence(self,trainX,trainY):
        all_trees = []

	for i in range(len(self.learners)):
                rdms = np.random.randint(0,trainX.shape[0],size=trainX.shape[0])
                x_bag = trainX[rdms,:]
                y_bag = trainY[rdms]
		new_tree = self.learners[i].build_random_tree(x_bag,y_bag) #build_random_tree(trainX,trainY)
                #new_tree = self.learners[i].build_random_tree(trainX,trainY)
		all_trees.append(new_tree)

        self.trees = all_trees

    def query(self,points):
        """
        @summary: Estimate a set of test points given the tree we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved tree.
        """
        bagged_trees = self.trees
        bagged_predY = [] 
        for decision_tree in bagged_trees:
                predicted_y = []
                for point in points:
                        position = 0
                        is_leaf = False
                        while not is_leaf:
                                node = decision_tree[float(position)]
                                node_or_leaf = node[0]
                                splitVal = node[1]
                                lefttree = node[2]
                                righttree = node[3]
                                if node_or_leaf == -1.0:
                                        y_value = splitVal
                                        is_leaf = True
                                else:
                                        if point[node_or_leaf] <= float(splitVal):
                                                position = position + float(lefttree) 
                                        else:
                                                position = position + float(righttree)
                        predicted_y.append(y_value)
                bagged_predY.append(predicted_y)  #= bagged_predY + np.array(predicted_y)

        matrix_predY = np.array(bagged_predY)
        
        #print len(stats.mode(matrix_predY)[0][0])     
        return stats.mode(matrix_predY)[0][0]

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
