import numpy as np
# import tensorflow as tf
rng = np.random
from scipy.stats import truncnorm
from scipy.stats.distributions import norm

import pdb

def ei_not_best(mu, var, best):
    p_over = norm(mu, var).sf(best)  # prob gonna be better
    unnormed = truncnorm(best, np.inf, loc=mu, scale=var).expect()  # EV | better
    ei = unnormed * p_over  # (EV | better) * p(better)
    return ei


def ei_best(mu, var, second_best):
    p_under = norm(mu, var).cdf(second_best)  # find prob we're gonna be lower
    unnormed = truncnorm(-np.inf, second_best, loc=mu, scale=var).expect()  # EV | worse
    ei = unnormed * p_under  # normalize
    return ei


class Bandit(object):
    def __init__(self, drift_rate, mu=0., var=0.):
        self.mu = mu
        self.var = var
        self.drift_rate = drift_rate

    def drift(self):
        self.mu += rng.randn() * self.drift_rate
    
    def sample(self):
        return self.mu + (rng.randn() * self.var)


class Agent(object):
    def __init__(self, bandits, drift_rate):
        self.n = len(bandits)
        self.i_bandits = np.arange(self.n)
        self.bandits = bandits
        self.drift_rate = drift_rate
        self.mu = np.zeros(self.n)
        self.var = np.ones(self.n) * self.drift_rate
        self.voi = [None] * self.n

        self.update_best()

    def update_best(self):
        self.second_best, self.best = np.sort(self.mu)[-2:]
        self.i_second_best, self.i_best = np.argsort(self.mu)[-2:]

    def sample(self):
        # choose which bandit to pull
        choice = self.choose()
        reward = self.bandits[choice].sample()
        self.mu[choice] = reward  # update mu

        # bandits rove
        [bandit.drift() for bandit in self.bandits]
        self.var[choice] = 0.  # reset uncertainty

        # update uncertainty
        for i_bandit in self.i_bandits:
            self.var[i_bandit] += self.drift_rate

        # update inferred best and second best
        self.update_best()

        return choice, reward

    def VOI(self, i_bandit):
        if i_bandit == self.i_best:
            voi = ei_best(self.mu[i_bandit], self.var[i_bandit], self.second_best)
        else:
            voi = ei_not_best(self.mu[i_bandit], self.var[i_bandit], self.best)
        return voi

    def choose(self):
        for i_bandit in self.i_bandits:
            self.voi[i_bandit] = self.VOI(i_bandit)  # cache voi
        # TODO: should this be self.mu + self.voi or just voi?
        return np.argmax(self.voi + self.mu)


if __name__ == '__main__':
    N_BANDIT = 4
    DRIFT_RATE = 0.1
    bandits = [Bandit(DRIFT_RATE) for _ in xrange(N_BANDIT)]

    agent = Agent(bandits, DRIFT_RATE)

    N_SAMPLES = int(1e2)
    vois = []  # history of voi
    mus = []  # inferred means
    var = []  # agent uncertainty
    real_mus = []  # actual means
    i_bandits = []  # which bandit chosen
    rewards = []  # reward from chosen

    for i_sample in xrange(N_SAMPLES):
        i_bandit, reward = agent.sample()
        i_bandits.append(i_bandit)
        rewards.append(reward)
        vois.append(list(agent.voi))
        mus.append(list(agent.mu))
        var.append(list(agent.var))
        real_mus.append(list([b.mu for b in agent.bandits]))

