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


def Marginal(fn):
    @wraps(fn)
    def shawarma(*args):
        return HashingMarginal(Search(fn, max_tries=int(1e3)).run(*args))
    return memoize(shawarma)


expr = nltk.sem.Expression.fromstring


class L:
    def __init__(self,s1_alpha, swk, B, belief, QUDs):
        """
        Args:
            s1_alpha: speaker optimality to be assumed by L1. Replace with prior over possible
            optimalities once inference works
            wk: list of nltk.sem.Expression FOL predicates. This is the common world knowledge
            of speaker and listener that both take the other to know
            B: list of phrasal.phrase phrases. this list is the list of possible beliefs the
            speaker might have about how the event under discussion went down.
            belief: phrasal.phrase object indicating listener belief. In the example discussed, this
            is equal to the last statement made upon which the speaker's correction calculation is
            done
            QUDs: dictionary with string keys and nltk.sem.Expression object values, holds questions
            under discussion the speaker may be referring to
        Returns:
            self
        """
        self.s1_alpha = s1_alpha
        self.swk = swk #shared world knowledge, list of expr
        self.B = B#belief set of speaker (for construction of belief prior)
        self.belief = belief#self.belief == last_statement_made 
        self.QUDs = QUDs

    @Marginal
    def L0(self, correction):
        """

        """
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
    def L1(self, correction):

        """
        #infers belief of s1
        #only really needs to infer the relevant change as in replacement suggested
        #in correction
        """
        print("µ"*20, " pragmatic listener calculation ", "µ"*20)
        interjector_belief=belief_prior(self.B)
        #^ convert back into phrase?
        print("duck : ", interjector_belief, type(interjector_belief))
        qud = qud_prior(self.QUDs)
        speaker = S(self.s1_alpha, self.swk, self.B, interjector_belief, self.QUDs)

        #v below self belief == last_statement_made, once again
        speaker_marginal = speaker.interject(self.belief, qud)

        pyro.sample("speaker", speaker_marginal, obs=correction)
        print("µ"*20, " pragmatic listener infers: ", interjector_belief, " ", "µ"*20)
        return interjector_belief


class S:
    def __init__(self, alpha, swk, B, belief, QUDs):
        self.alpha = alpha
        self.swk = swk
        self.B = B
        self.belief = belief
        self.QUDs = QUDs

    def utterance_prior(self,given_full_belief):
        #My utterance prior is a function of speaker state
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
        changerLogits = -torch.tensor([phr.cost for phr in possible_changers], dtype=torch.float64) 
        ix = pyro.sample("utterance",dist.Categorical(logits=changerLogits))
        print("changer logits: ", changerLogits)
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

        with poutine.scale(scale=torch.tensor(float(self.alpha))):
            utterance = self.utterance_prior(given_full_belief)
            print("interject utterance prior: ", utterance)

            #Construct Listener in head
            listener = L(self.alpha, self.swk, self.B, given_full_belief, self.QUDs)
            literal_marginal = listener.L0(utterance)
            projected_literal = self.project(literal_marginal, qud)
            print("projected_literal: ", projected_literal)
            pyro.sample("listener", projected_literal, obs=qudSelf)
            print("#"*20+" Speaker says: "+"#"*20+"\n")
            print(str(utterance), type(utterance))
            print("#"*20+"   /Speaker  "+"#"*20+"\n")
            print("~"*60)
        return utterance

    @Marginal
    def project(self,dist,qud):
        #projection helper function so a hashingmarginal distribution can be used in interjection inference
        v = pyro.sample("proj",dist)
        print("sampled v : ", v, type(v))
        added_expression = [] if str(v) == NULL else [v.L()]
        for elem in added_expression:
            print(elem, type(elem))
        return rpc(goal=self.QUDs[qud],assumptions=self.swk+added_expression).prove()


def belief_prior(B): #over list
    ix = pyro.sample("belief", dist.Categorical(probs=torch.ones(len(B))/len(B)))
    return B[ix]


def qud_prior(quds): #over dict
    keys = list(quds.keys())
    ix = pyro.sample("qud", dist.Categorical(probs=torch.ones(len(keys))/len(keys)))
    qud = keys[ix.item()]
    return qud

def plot_dist(d, output="plots/distplot.png", addinfo=None):
    support = d.enumerate_support()
    data = [d.log_prob(s).exp().item() for s in d.enumerate_support()]
    names = support

    ax = plt.subplot(111)
    width=0.3
    bins = list(map(lambda x: x-width/2,range(1,len(data)+1)))
    ax.bar(bins,data,width=width)
    ax.set_xticks(list(map(lambda x: x, range(1,len(data)+1))))
    ax.set_xticklabels(names,rotation=20, rotation_mode="anchor", ha="right")

    if type(addinfo)==str:
        plt.figtext(.1,.1, addinfo)

    plt.tight_layout()

    plt.savefig(output)


if __name__ == "__main__":
    pass

