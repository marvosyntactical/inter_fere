import nltk
from nltk.inference import TableauProverCommand
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from phrasal import Utt, event, role, phrase 

from search_inference import factor, HashingMarginal, memoize, Search
from functools import wraps

def Marginal(fn):
    @wraps(fn)
    def shawarma(*args):
        return HashingMarginal(Search(fn).run(*args)), fn(*args)
    return memoize(shawarma)

expr = nltk.sem.Expression.fromstring
tpc = TableauProverCommand

class L:
    def __init__(self, wk, B, belief):
        self.swk = wk #shared world knowledge, list of expr
        self.B = B#belief set of speaker (for construction of belief prior)
        self.belief = belief#last statement made


    @Marginal
    def L0(self, correction):
        print("\\"*20+" literal listener "+"\\"*20)
        #doesnt take into account own belief or qud, literally infers belief...by replacing subutterance
        interjector_belief=belief_prior(self.B)
        replacement = self.belief.replace_constituents_in_utt(correction)
        print("L0's replacement: ",replacement)
        evaluation = tpc(goal=interjector_belief.L(), assumptions=self.swk+[replacement.L()]).prove()
        factor("literal meaning", 0. if evaluation else -99999999.)#condition on s1 belief, correctly infers belief in basic scenario, how do i get blue ambiguity?
        print("+"*20+ "    /listener   "+"\\"*20)
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

    @Marginal
    def utterance_prior(self,given_full_belief):
        #why is my utterance prior a function of state ?????
        both_assigned = list(set(self.belief.assed.keys()).intersection(set(given_full_belief.assed.keys())))
        diff_roles = [] #differently assigned 
        for role in both_assigned:
            if self.belief.assed[role] != given_full_belief.assed[role]:
                diff_roles.append(role)
        diff = []
        for const in given_full_belief.elems:
            for r in diff_roles:
                if const.field == r:
                    diff += [const]
        d = phrase(diff) #difference in beliefs phrased fully
        
        possible_changers = d.sub_utterances()
        
        ix = pyro.sample("utterance",dist.Categorical(probs=torch.ones(len(possible_changers)) / len(possible_changers)))
        print("possible_changers, selection:")
        print(possible_changers)
        print(possible_changers[ix])
        return possible_changers[ix]
        
    @Marginal
    def interject(self, given_full_belief, qud):
        #calc difference in roles
        #now: Which are relevant to qud change -> evaluate
        print("~"*20+"Speaker interject"+"~"*20)
        print("\ndebug: speaker initialized,\n qud: "+ qud)
        print("\ndebug: self.belief.L(): ", self.belief.L())
        print("\ndebug: given_full_belief: ", given_full_belief.L())
        print("\ndebug: self.swk: ", self.swk)
        
        qudSelf = tpc(goal=self.QUDs[qud], assumptions=self.swk+[self.belief.L()]).prove(verbose=False) 
        qudOther = tpc(goal=self.QUDs[qud], assumptions=self.swk+[given_full_belief.L()]).prove(verbose=False)
        print("\ndebug: qudSelf: ", qudSelf, " qudOther: ", qudOther)
        print("\nagreement?",qudSelf == qudOther)
                
        alpha = 1.0
        with poutine.scale(scale=torch.tensor(alpha)):
            utterance = self.utterance_prior(given_full_belief)
            #print(utterance)
            utt_no_infer = utterance[1]
            #Construct Listener in head
            listener = L(self.swk, self.B, given_full_belief)
            literal_marginal = listener.L0(utt_no_infer)
            
            projected_literal = tpc(goal=self.QUDs[qud], assumptions=self.swk+[utt_no_infer]).prove()
            pyro.sample("listener", projected_literal, obs=qudSelf)
            
            print("utterance")
            print("#"*20+"   /Speaker  "+"#"*20)
        return utterance

def belief_prior(B):
    ix = pyro.sample("belief", dist.Categorical(probs=torch.ones(len(B))/len(B)))
    return B[ix]

def qud_prior(quds):
    values = list(quds.keys())
    ix = pyro.sample("qud", dist.Categorical(probs=torch.ones(len(values))/len(values)))
    qud = values[ix.item()]
    return qud


