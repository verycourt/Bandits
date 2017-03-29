# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:06:31 2016

@author: claire
"""
import numpy as np
from random import random, randint

# from random import betavariate
from math import log


def eGreedy(n_arms, epsilon, rewards, draws):
    if np.sum(draws == 0) > 0:
            c = np.where(draws == 0)[0][0]
    else:
        u = random()
        if u < epsilon:
            c = randint(0, n_arms - 1)
        else:
            indices = rewards / draws
            winners = np.argwhere(indices == np.max(indices))
            c = np.random.choice(winners[0])
    return c


## DRAWS = nombre de tirages par machine
def UCB(t, alpha, rewards, draws):
    if np.sum(draws == 0) > 0:
            c = np.where(draws == 0)[0][0]
    else:
        delta = 1 / t ** alpha
        
        # indices = UCB_k(t)
        indices = rewards / draws + np.sqrt( np.log(1/delta) / (2. * draws) ) # DONE
        
        # winners = I_t+1
        winners = np.argwhere(indices == np.max(indices))
        
        # potentiellement on peut avoir des cas d'égalité de score, donc on prend une de ces valeurs aléatoirement
        c = np.random.choice(winners[0])

    return c


def Thompson(n_arms, rewards, draws):
    indices = np.zeros(n_arms)
    for arm in np.arange(n_arms):
        indices[arm] = rewards / draws  # TODO
    winners = np.argwhere(indices == np.max(indices))
    c = np.random.choice(winners[0])
    return c


def kl(a, b):
    return a * log(a / b) + (1 - a) * log((1 - a) / (1 - b))


def computeLowerBound(n_arms, true_means):

    # DONE : use kl
    LB = 0
    
    for k in range(1, 4):
        LB += (true_means[0] - true_means[k]) / kl(true_means[k], true_means[0])

    return LB
