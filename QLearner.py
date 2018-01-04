"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

	# Set class vaiables
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
	self.dyna = dyna
	
	# Initialize Q-table
        self.q = np.random.uniform(-1.0,1.0,size=(num_states,num_actions))
	
	# Initialize memory
        self.memory = []

    def author(self):
        return 'enaziga3'

    def querysetstate(self, s):

	# Set state to desired input value
        self.s = s

	# Decide if to choose random or optimal action to be returned
	random_or_optimal = rand.random()
        if random_or_optimal < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.q[self.s])

        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):

	# Update Q-table and memory using current <s,a,s_prime,r> tuple
        self.q[self.s,self.a] = (1.0-self.alpha)*self.q[self.s,self.a] + \
                                    self.alpha*(r + self.gamma*self.q[s_prime,np.argmax(self.q[s_prime])])
        self.memory.append([self.s, self.a, s_prime, r])
	
	# Decide if to choose random or optimal action to be returned
        random_or_optimal = rand.random()
        if random_or_optimal < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.q[s_prime])

	# If DynaQ is requested, use memory to update Q-table for requested number of times
	if self.dyna != 0:
	    
	    memory_indexes = np.random.choice(len(self.memory), size=self.dyna, replace=True)

	    for memory_index in memory_indexes:

		dyna_state, dyna_action, state_p, R = self.memory[memory_index]

		self.q[dyna_state,dyna_action] = (1.0-self.alpha)*self.q[dyna_state,dyna_action] + \
                                    self.alpha*(R + self.gamma*self.q[state_p,np.argmax(self.q[state_p])])

	if self.verbose: print "s =", s_prime,"a =",action,"r =",r

	# Update learning rate, action and set state to new state (stored in class variables)
	self.rar = self.rar*self.radr
        self.a = action
        self.s = s_prime

	return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
