import os 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import logsumexp

########################################
##       Decide what to learn         ##
########################################
'''
Paper link:   https://arxiv.org/abs/2101.06197
'''
class Bandit:

    def __init__( self, num_arms, stationary=True):
        '''Multi-arm bandit with bernoulli bandit

        A initialization, multiple arms are created.
        The probability of each arm that turns reward 1
        if pulled is sampled from Ber(p), where p randomly
        chosen from uniform(0,1) at initialization 
        '''
        self.num_arms = num_arms
        self.reset()
        self.t = 0 
        self.stationary = stationary

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

class Thompson:

    def __init__( self, env, verbose=True):
        self.env    = env
        self.num_arms = env.num_arms
        self.verbose = verbose
        self.name = f'Thompson'
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

    def train( self, T):

        # init storages
        regrets = np.zeros( [T,])
        cum_regret = np.zeros( [T,])

        # iteration to learn 
        for t in range(T):

            # sample environment samples from current belief
            env_sample = self.get_sample()
            # generate action 
            act = self.get_action( env_sample)
            # compute reward regrets 
            reward, regret = self.get_reward_regret( act)
            # update internal model
            self.update_model( act, reward)
            # record regrets 
            regrets[t] = regret
            if t > 1:
                cum_regret[t] = regret + cum_regret[t-1]
            else:
                cum_regret[t] = regret
            # verbose 
            if self.verbose:
                if t % 1000 == 0:
                    print (f'Alg: {self.name}, Epi: {t}, Regret: {regret}')

        return regrets, cum_regret

    def get_sample( self):
        '''Generate a sample from the internal model
        '''
        return np.random.beta( self.params['alpha'], self.params['beta'])

    def update_model( self, act, reward):
        '''Update the model 

        Grow a better model by feeding the reward
        '''
        self.params['alpha'][act] += reward 
        self.params['beta'][act]  += 1 - reward  

    def get_action( self, env_sample):
        '''Decide what arms should be pulled

        Sample possible results from the current internal
        model. And then pick the most rewarding one. 
        '''
        return env_sample.argmax()

    def get_reward_regret( self, act):
        reward, regret = self.env.get_reward_regret( act)
        return reward, regret


class BLASTS:

    def __init__( self, env, beta=1., 
                             max_blahut=100, 
                             num_samples=64,
                             tol=1e-3,
                             verbose=True):
        self.env = env
        self.num_arms = env.num_arms
        self.beta = beta 
        self.max_blahut = max_blahut
        self.num_samples = num_samples
        self.tol = tol
        self.verbose = verbose
        self.name = f'BLASTS beta={self.beta}'
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

    def train( self, T):

        # Init storages
        regrets = np.zeros([T,])
        cum_regret = np.zeros([T,])

        # iteration to learn 
        for t in range(T):

            # sample environment samples from current belief
            env_samples = self.get_samples()
            # calculate distortion
            distortion = self.cal_distortion( env_samples)
            # get policy given env_samples and regret
            pi = self.get_policy( distortion)
            # sample from posterior and do probabilistic matching
            act = self.get_action( pi)
            # compute reward regrets 
            reward, regret = self.get_reward_regret( act)
            # update internal model
            self.update_model( act, reward)
            # record regrets 
            regrets[t] = regret 
            if t > 1:
                cum_regret[t] = regret + cum_regret[t-1]
            else:
                cum_regret[t] = regret
            # verbose 
            if self.verbose:
                if t % 1000 == 0:
                    print (f'Alg: {self.name}, Epi: {t}, Regret: {regret}')

        return regrets, cum_regret

    def get_samples( self):
        env_samples = np.zeros( [ self.num_samples, self.num_arms])
        for i in range(self.num_samples):
            env_samples[ i, :] = np.random.beta( self.params['alpha'], self.params['beta'])
        return env_samples 

    def cal_distortion( self, env_samples):
        return (env_samples.max(axis=1, keepdims=True) - env_samples)**2

    def get_policy( self, distortion):
        '''Find the best policy using Blahut-Arimoto alg
        '''
        pi = np.ones( [ self.num_samples, self.num_arms]) / self.num_arms
        pe = np.ones( [self.num_samples, 1]) / self.num_samples
        done = False 
        niter = 0
        while not done:
            # save the old policy for convergence check 
            pi_old = pi.copy()
            # update marginal policy 
            pa = pe.T @ pi 
            # update policy
            log_pi = np.log( pa + 1e-20) - self.beta * distortion
            log_Z  = logsumexp( log_pi, axis=1, keepdims=True)
            pi     = np.exp( log_pi - log_Z)
            # check convergence 
            delta = np.sum( (pi - pi_old)**2)
            if (delta < self.tol) or (niter > self.max_blahut):
                done = True 
                break
            # record the iteration 
            niter += 1

        return pi  

    def update_model( self, act, reward):
        '''Update the model 

        Grow a better model by feeding the reward
        '''
        self.params['alpha'][act] += reward 
        self.params['beta'][act]  += 1 - reward  

    def get_action( self, pi):
        '''Decide what arms should be pulled
        '''
        pi_sample = pi[ np.random.choice( range(self.num_samples)), :]
        return np.random.choice(range(self.num_arms), p=pi_sample)

    def get_reward_regret( self, act):
        reward, regret = self.env.get_reward_regret( act)
        return reward, regret 

def experiment( args):    
    n_alg = 1 + len(args.betas)
    alg_idx = 0
    sum_regrets = np.zeros([ n_alg, args.T])
    sum_cum_regrets = np.zeros([ n_alg, args.T])
    # generate results for 
    for _ in range( args.n):
        env = Bandit( args.num_arms)
        agent = Thompson( env)
        regrets, cum_regret = agent.train( args.T)
        sum_regrets[ alg_idx, :] += regrets
        sum_cum_regrets[ alg_idx, :] += cum_regret
    alg_idx += 1
    for beta in args.betas:
        for _ in range( args.n):
            env = Bandit( args.num_arms)
            agent = BLASTS( env, beta)
            regrets, cum_regret = agent.train( args.T)
            sum_regrets[ alg_idx, :] += regrets
            sum_cum_regrets[ alg_idx, :] += cum_regret
        alg_idx += 1
    mean_regrets = sum_regrets / args.n
    mean_cum_regrets = sum_cum_regrets / args.n 
    return mean_regrets, mean_cum_regrets

def show_results( regrets, cum_regrets, names):
    # plot the regrets 
    T = regrets.shape[1]
    x = np.arange( T)
    plt.figure( 10, 5)
    plt.subplot( 1, 2, 1)
    colors = [ 'b', 'r', 'g', 'purple', 'orange']
    for i in range( regrets.shape[0]):
        plt.plot( x, regrets[i,:], 'o', markersize=2, label=names[i])
    plt.legend( prop={'size': 16}, numpoints=5)
    plt.ylabel('regrets')
    plt.xlabel('time period')
    plt.subplot( 1, 2, 2)
    for i in range( cum_regrets.shape[0]):
        plt.plot( x, cum_regrets[i,:], 'o-', markersize=2, label=names[i])
    plt.legend( prop={'size': 16}, numpoints=5)
    plt.ylabel('cum regrets')
    plt.xlabel('time period')

class hyperparams:

    def __init__( self):
        self.T = 2000
        self.n = 50
        self.num_arms = 50
        self.betas = [.5, 2.5, 10., 50.]
        self.name  = self.get_names()

    def get_names( self):
        name = ['Thompson']
        for beta in self.betas:
            name.append( 'beta={}'.format(beta))
        return name 

if __name__ == '__main__':

    # load hyperparams
    args = hyperparams()

    # begin simulation
    regrets, cum_regrets = experiment( args)

    # plot
    show_results( regrets, cum_regrets, args.name)
    try:
        plt.savefig( 'BLASTS-50 arms.png')
    except:
        os.mkdir( 'figures')
        plt.savefig( 'BLASTS-50 arms.png')
    