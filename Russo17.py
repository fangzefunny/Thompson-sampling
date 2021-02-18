import os 
import numpy as np 
import matplotlib.pyplot as plt

'''
Paper link:   https://arxiv.org/abs/1707.02038
'''
########################################
##   Tutorial on Thompson sampling    ##
########################################

'''
Part1: Environment

This is simple bandit environment,
'''
class Bandit:

    def __init__( self, num_arms, args):
        '''Multi-arm bandit with bernoulli bandit

        A initialization, multiple arms are created.
        The probability of each arm that turns reward 1
        if pulled is sampled from Ber(p), where p randomly
        chosen from uniform(0,1) at initialization 
        '''
        self.num_arms = num_arms
        self.reset()
        self.t = 0 
        self.stationary = args.stationary

    def reset( self):
        self.thetas = np.random.uniform( 0, 1, self.num_arms)

    def get_reward_regret( self, act):
        '''Returns random reward for arm action. 
        '''
        self.t += 1
        if (self.stationary==False) and (self.t % 100 == 0):
            self.reset()
        # simulate ber sampling 
        rewards = (np.random.uniform( 0, 1, self.num_arms) 
                            < self.thetas).astype(int)
        reward  = rewards[ act]
        regret  = self.thetas.max() - self.thetas[ act]

        return reward, regret 

class thompson:

    def __init__( self, env):
        self.env    = env
        self.num_arms = env.num_arms
        self._init_model()
    
    def _init_model( self):
        '''Init the internal model of env

        Assume that the agent knows the bandit are
        all bernoulli bandit. Then we  need to guess
        the theta parameter for a ber distribution, which
        follows:

            theta ~ beta( alpha, beta)
        '''
        alphas = np.ones( [self.num_arms,])
        betas  = np.ones( [self.num_arms,])
        self.params = { 'alpha': alphas,
                        'beta' : betas }
    
    def update_model( self, act, reward):
        '''Update the model 

        Grow a better model by feeding the reward
        '''
        self.params['alpha'][act] += reward 
        self.params['beta'][act]  += 1 - reward  

    def get_action( self):
        '''Decide what arms should be pulled

        Sample possible results from the current internal
        model. And then pick the most rewarding one. 
        '''
        thetas = np.random.beta( self, self.params['alpha'], self.params['beta'])
        return thetas.argmax()

    def get_reward_regret( self, act):
        reward, regret = self.env.get_reward_regret( act)
        self.update_model( act, reward)
        return reward, regret

class e_greedy:

    def __init__( self, bandit):
        global epsilon 
        self.epsilon = epsilon 
        self.bandit  = bandit 
        self.num_arms = bandit.num_arms 
        self.qvalue  = np.zeros([self.num_arms])
        self.count   = np.zeros([self.num_arms])

    @staticmethod
    def name( self):
        return 'e-greedy'

    def get_action( self):
        if np.random.uniform(0,1) > self.epsilon:
            act = self.qvalue.argmax()
        else:
            act = np.random.randint( 0, self.num_arms)
        return act 

    def get_reward_regret( self, act):
        reward, regret = self.bandit.get_reward_regret( act)
        self._update_params( act, reward)
        return reward, regret 

    def _update_params( self, act, reward):
        self.count[ act] += 1 
        self.qvalue[ act] += 1/self.count[act] * ( reward - self.qvalue[act])

class UCB:

    def __init__( self, bandit):
        global c
        self.c = c 
        self.bandit = bandit 
        self.num_arms = bandit.num_arms 
        self.qvalue  = np.zeros([self.num_arms])
        self.count   = np.zeros([self.num_arms]) + .0001
        self.t = 1

    @staticmethod
    def name( self):
        return 'UCB'

    def get_action( self):
        ln_t = np.log( np.ones([self.num_arms]) * self.t)
        conf = np.sqrt( ln_t / self.count)
        act = np.argmax( self.qvalue + self.c * conf)
        self.t += 1
        return act 

    def get_reward_regret( self, act):
        reward, regret = self.bandit.get_reward_regret( act)
        self._update_params( act, reward)
        return reward, regret 

    def _update_params( self, act, reward):
        self.count[ act] += 1 
        self.qvalue[ act] += 1/self.count[act] * ( reward - self.qvalue[act])

def simulate( simulations, T, num_arms, alg):
    sum_regrets = np.zeros([T])
    for sim in range( simulations):
        bandit = Bandit( num_arms)
        algo   = alg(bandit)
        regrets= np.zeros( [T])
        for t in range(T):
            action = algo.get_action()
            reward, regret = algo.get_reward_regret(action)
            regrets[t] = regret
        sum_regrets += regrets
    mean_regrets = sum_regrets / simulations 
    return mean_regrets

def experiment( num_arms, T=1000, simulations=1000):
    algos   = [ e_greedy, UCB, thompson]
    names   = [ 'e-greedy', 'UCB', 'thompson']
    regrets = []
    for algo in algos:
        regrets.append( simulate( simulations, T, num_arms, algo))
    
    x = np.arange( len(regrets[0]))
    for i, y in enumerate( regrets):
        plt.plot( x, y, 'o', markersize=2, label=names[i])
    plt.legend( loc='upper right', prop={'size': 16}, numpoints=5)
    

if __name__ == '__main__':

    num_arms = 10 
    epsilon  = .1
    c        = 2
    stationary = True 
    experiment( num_arms)
    try:
        plt.savefig('figures/Russo17.png')
    except:
        os.mkdir( 'figures')
        plt.savefig('figures/Russo17.png')

