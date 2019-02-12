import torch
torch.set_default_dtype(torch.float64) #test with lesser dtype

import collections
import argparse
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from search_inference import factor, HashingMarginal, Search, memoize

import sys

#== infer in webppl
def Marginal(func):
    return memoize(lambda *args: HashingMarginal(Search(func).run(*args)))


#construct world

n_states = 4

def state_prior():
    #notice argumentless, try with mock arguments?
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
    state = state_prior()
    #if len(sys.argv) == 3:
    #    utt = sys.argv[-2]
    #    myint = int(sys.argv[-1])
    #    state = torch.tensor([myint])
    print("State: ", int(state.item()), "out of n=4 objects")
    print("Utterance: "+ utt)
    print("Truth Value: ",int(meaning(utt, state).item()))
    print("\n\n")


@Marginal
def l0(utterance):
    state = state_prior()
    factor("literal_meaning", 0. if meaning(utterance, state
