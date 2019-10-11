import nltk
from nltk.inference import ResolutionProverCommand as rpc

import matplotlib.pyplot as plt
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from phrasal import * 

from search_inference import factor, HashingMarginal, memoize, Search
from functools import wraps


def Marginal(fn): #reduce max tries for debugging
    @wraps(fn)
    def shawarma(*args):
        return HashingMarginal(Search(fn, max_tries=int(1e3)).run(*args))#, fn(*args)
    return memoize(shawarma)
"""
def Marginal(fn):
    #uncomment to disable inference to debug other stuff
    def rap(*args):
        return dist.Categorical(probs=torch.ones(1)), fn(*args)
    return rap
"""

expr = nltk.sem.Expression.fromstring


class L:
    def __init__(self, wk, B, belief):
        self.swk = wk #shared world knowledge, list of expr
        self.B = B#belief set of speaker (for construction of belief prior)
        self.belief = belief#self.belief == last_statement_made 

    @Marginal
    def L0(self, correction):
        print("\\"*20+" literal listener "+"\\"*20+"\n")
        #doesnt take into account own belief or qud, literally infers belief...by replacing subutterance
        print("L0's belief: ", self.belief)
        #print("L0's B set of speaker: ", [b.L() for b in self.B])


        interjector_belief=belief_prior(self.B) #state_prior() in RSAhyperb
        replacement = self.belief.replace_constituents_in_utt(correction)
        print("L0's replacement: ",replacement)

        if replacement == self.belief: #TODO test if this operation works
            evaluation = True
        else: 
            evaluation = rpc(goal=interjector_belief.L(), assumptions=self.swk+[replacement.L()]).prove()

        factor("literal meaning", 0. if evaluation else -99999999.)#condition on s1 belief, correctly infers belief in basic scenario, how do i get blue ambiguity?
        print("+"*20+ "    /listener   "+"\\"*20+"\n")
        return interjector_belief

    @Marginal
    def L1(self, correction, quds):

        """
        #infers belief of s1?
        #only really needs to infer the relevant change as in replacement suggested
        #in correction
        """
        pass

class S:
    def __init__(self, swk, belief, QUDs, B):
        self.swk = swk
        self.belief = belief
        self.QUDs = QUDs
        self.B = B

    #@Marginal
    def utterance_prior(self,given_full_belief):
        #why is my utterance prior a function of state ?????
        both_assigned = set(self.belief.assed.keys()).intersection(set(given_full_belief.assed.keys()))
        diff_roles = [] #differently assigned 
        for role in both_assigned:
            if self.belief.assed[role] != given_full_belief.assed[role]:
                diff_roles.append(role)
        diff = []
        for const in self.belief.elems:
            for r in diff_roles:
                if const.field == r:
                    diff += [const]
        d = phrase(diff) #difference in beliefs phrased fully

        possible_changers = d.sub_utterances()

        ix = pyro.sample("utterance",dist.Categorical(probs=torch.ones(len(possible_changers)) / len(possible_changers)))
        print("possible_changers:")
        print([c.L() for c in possible_changers])
        print(possible_changers[ix])
        return possible_changers[ix]

    @Marginal
    def interject(self, given_full_belief, qud):
        print("~"*20+"Speaker interject"+"~"*20+"\n")
        print("\ndebug: speaker initialized")
        print("qud: "+ qud)
        print("meaning: ", self.QUDs[qud])
        print("\ndebug: self.belief.L(): ", self.belief.L())
        print("\ndebug: given_full_belief: ", given_full_belief.L())
        print("\ndebug: self.swk: ", self.swk)

        qudSelf = rpc(goal=self.QUDs[qud], assumptions=self.swk+[self.belief.L()]).prove(verbose=False) 
        qudOther = rpc(goal=self.QUDs[qud], assumptions=self.swk+[given_full_belief.L()]).prove(verbose=False)
        print("\ndebug: qudSelf: ", qudSelf, " qudOther: ", qudOther)
        print("\nagreement?",qudSelf == qudOther)

        alpha = 1.0
        with poutine.scale(scale=torch.tensor(alpha)):
            utterance = self.utterance_prior(given_full_belief)
            print("interject utterance prior: ", utterance)

            #Construct Listener in head
            listener = L(self.swk, self.B, given_full_belief)
            literal_marginal = listener.L0(utterance)
            projected_literal = self.project(literal_marginal, qud)
            print("projected_literal: ", projected_literal)
            pyro.sample("listener", projected_literal, obs=qudSelf)
            print("#"*20+" Speaker says: "+"#"*20+"\n")
            print(str(utterance))
            print("#"*20+"   /Speaker  "+"#"*20+"\n")
            print("#"*60)
        return utterance

    @Marginal
    def project(self,dist,qud):
        #projection helper function so a hashingmarginal distribution can be used in interjection inference
        v = pyro.sample("proj",dist) #im not even using this var yet
        print("sampled v : ", v, type(v))
        added_expression = [] if str(v) == NULL else [v.L()]
        for elem in added_expression:
            print(elem, type(elem))
        return rpc(goal=self.QUDs[qud],assumptions=self.swk+added_expression).prove()


    @Marginal
    def project_old(self,dist,qud,expr):
        #projection helper function so a hashingmarginal distribution can be used in interjection inference
        v = pyro.sample("proj",dist) #im not even using this var yet
        print("sampled v : ", v, type(v))
        added_expression = [] if str(expr) == NULL else [expr]
        return rpc(goal=self.QUDs[qud],assumptions=self.swk+added_expression).prove()


def belief_prior(B): #over list
    ix = pyro.sample("belief", dist.Categorical(probs=torch.ones(len(B))/len(B)))
    return B[ix]


def qud_prior(quds): #over dict
    values = list(quds.keys())
    ix = pyro.sample("qud", dist.Categorical(probs=torch.ones(len(values))/len(values)))
    qud = values[ix.item()]
    return qud

def plot_dist(d, output="output/distplot.png"):
    support = d.enumerate_support()
    data = [d.log_prob(s).exp().item() for s in d.enumerate_support()]
    names = support

    ax = plt.subplot(111)
    width=0.3
    bins = list(map(lambda x: x-width/2,range(1,len(data)+1)))
    ax.bar(bins,data,width=width)
    ax.set_xticks(list(map(lambda x: x, range(1,len(data)+1))))
    ax.set_xticklabels(names,rotation=45, rotation_mode="anchor", ha="right")

    plt.tight_layout()
    plt.savefig(output)


if __name__ == "__main__":
    pass

