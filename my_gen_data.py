"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regresstion than random trees
def best4LinReg():
    X = np.random.normal(size = (100))
    X1 = 3*X
    X2 = 3*X + 4*X1
    X3 = X-X2+X1*X2
    X4 = X3**3 + 2*X
    newX = np.array([X,X1,X2,X3,X4])
     
    Y = np.sin(newX[:,1])*np.cos(1./(0.0001+newX[:,0]**2)) 
    return newX, Y

def best4RT():
    #X = np.random.normal(size = (50, 2))
    X = np.random.normal(size = (50)) 
    X1 = 3*X
    X2 = 3*X + 4*X1
    X3 = X-X2+X1*X2
    X4 = X3**3 + 2*X
    newX = np.array([X,X1,X2,X3,X4])
    Y = 0.8 * newX[:,0] + 5.0 * newX[:,1] 
    return newX, Y

if __name__=="__main__":
    print "they call me Tim."
