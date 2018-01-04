"""
An implementation of Regression Trees.  (c) 2017 Emmanuel Naziga
"""

import numpy as np
import random
from scipy import stats

class RTLearner(object):

    def __init__(self, leaf_size, verbose = False):
	self.leaf_size = leaf_size

    def author(self):
        return 'enaziga3' 

    def build_random_tree(self,trainX,trainY):

	leaf = -1.0
	ndata, nfeatures = trainX.shape
        
	if ndata <= self.leaf_size:
		return np.array([[leaf,stats.mode(trainY)[0][0],np.nan,np.nan]])
        elif (trainY[1:] == trainY[:-1]).all():
		return np.array([[leaf,stats.mode(trainY)[0][0],np.nan,np.nan]])
        #elif ndata == 0:
        #        return np.array([[leaf,-1,np.nan,np.nan]])
	else:
		split_feature = random.randint(0,nfeatures-1)
		split_indx1 = random.randint(0,ndata-1)
		split_indx2 = random.randint(0,ndata-1)
		
		split_val = (trainX[split_indx1,split_feature] + trainX[split_indx2,split_feature])/2.0
		count = 0
		while count < 10:
		   if (trainX[trainX[:,split_feature] <= split_val]).shape[0] == 0 or (trainX[trainX[:,split_feature] > split_val]).shape[0] == 0:
			split_indx1 = random.randint(0,ndata-1)
			split_indx2 = random.randint(0,ndata-1)
			split_val = (trainX[split_indx1,split_feature] + trainX[split_indx2,split_feature])/2.0
			count = count + 1
		   else:
			break
		
		# Build left part of the tree
		if (trainX[trainX[:,split_feature] <= split_val]).shape[0] == 0:
			return np.array([[leaf,stats.mode(trainY)[0][0],np.nan,np.nan]])
		else:
			new_trainX = trainX[trainX[:,split_feature] <= split_val]
			new_trainY = trainY[trainX[:,split_feature] <= split_val]
			left_tree = self.build_random_tree(new_trainX,new_trainY)

		# Build right part of the tree
		if (trainX[trainX[:,split_feature] > split_val]).shape[0] == 0:
			return np.array([[leaf,stats.mode(trainY)[0][0],np.nan,np.nan]])
		else:
			right_trainX = trainX[trainX[:,split_feature] > split_val]
                	right_trainY = trainY[trainX[:,split_feature] > split_val]
			right_tree = self.build_random_tree(right_trainX,right_trainY)					
	
		# Add root to the tree
		root = np.array([[split_feature,split_val,1,left_tree.shape[0]+1]])
		return (np.concatenate((root,left_tree,right_tree),axis=0))

    def addEvidence(self,trainX,trainY):
        """
        @summary: Add training data to learner
        @param X: X values of data to add
        @param dataY: the Y training values
        """
        # build and save the model
        self.tree = self.build_random_tree(trainX,trainY)

    def query(self,points):
        """
        @summary: Estimate a set of test points given the tree we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved tree.
        """

	decision_tree = self.tree
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
                print y_value
        return np.array(predicted_y)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
