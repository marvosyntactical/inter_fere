import torch
torch.set_default_dtype(torch.float64) #test with lesser dtype

import collections
import argparse
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from search_inference import factor, HashingMarginal, Search, memoize



#== infer in webppl
def Marginal(func):
    return memoize(lambda *args: HashingMarginal(Search(func).run(*args)))


#construct world

n_states = 4

def state_prior():
    #noteice argumentless, try with mock arguments?
    n =pyro.sample("whyDoINeedToNameThisState", dist.Categorical(probs=torch.ones(n_states+1)/(n_states+1)))
    return n

def utterance_prior():
    ix = pyro.sample("utt", dist.Categorical(probs=torch.ones(3)/3))
    return ["none", "some", "all"][ix]


meanings = {
        "none": lambda N:N==0,
        "some": lambda N:N>0,
        "all": lambda N: N==n_states
        }

def meaning(utterance, state):
    return meanings[utterance](state)

for _ in range(1000):
    utt = utterance_prior()
    print("Utterance: "+ utt)
    state = state_prior()
    print("State: ", state, "out of n=4 objects")
    print(meaning(utterance_prior(), state_prior()))
    print("\n\n")
    
