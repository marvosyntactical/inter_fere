import torch
torch.set_default_dtype(torch.float64) #test with lesser dtype

import collections
import argparse


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

#for _ in range(1000):
#    utt = utterance_prior() 
#    state = state_prior()
#    #if len(sys.argv) == 3:
#    #    utt = sys.argv[-2]
#    #    myint = int(sys.argv[-1])
#    #    state = torch.tensor([myint])
#    print("State: ", int(state.item()), "out of n=4 objects")
#    print("Utterance: "+ utt)
#    print("Truth Value: ",int(meaning(utt, state).item()))
#    print("\n\n")

@Marginal
def l0(utterance):
    state = state_prior()
    factor("literal_meaning", 0. if meaning(utterance, state) else -9999999.)
    return state

@Marginal
def s1(state):
    alpha = 1
    with poutine.scale(scale=torch.tensor(alpha)):
        utterance = utterance_prior()
        pyro.sample("listener", l0(utterance), obs=state)
    return utterance

@Marginal
def l1(utterance):
    state = state_prior()
    pyro.sample("speaker", s1(state), obs=utterance)
    return state

def viz(agent):
    #TODO with mpl support
    support = agent.enumerate_support()
    data = [agent.log_prob(s).exp().item() for s in support]
    for ix, prob in enumerate(data):
        print("State "+str(ix)+" has probability "+str(prob))
interp_some = l1("some")
viz(interp_some)


#----------------------------HYPERBOLE PRICE MODEL---WHERE IS COST??---
"""
 model cost of utterance:

IDEA A: 
    in utterance_prior() 
"""
"""
c = 1
def utterance_prior():
    ix = pyro.sample("utt", dist.Categorical(probs=torch.ones(3)/3))
    return ["none", "some", "all"][ix]
"""
#TODO associate QUD relevance with subutterance
#####
#for every subutterance, go through prolog machine and check if it conveys the difference
#
#going through all n! combinations of utterance is unnice
#computationally and not cognitively motivated
#
#how to make utteranceprior compositionally without assigning dist to potenzmenge?
#
#look at ccg implementation

def comp_utt_prior():




IDEA B:
    in cost term (see problang.org)

"""


#@Marginal
#def l

















