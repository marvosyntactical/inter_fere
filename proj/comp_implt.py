#----------------------------HYPERBOLE PRICE MODEL---WHERE IS COST??---
import itertools
import nltk

import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from search_inference import factor, HashingMarginal, memoize, Search

#pyro stuff

def Marginal(fn):
    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))



expr = nltk.sem.Expression.fromstring
p9 = nltk.inference.Prover9


class Utt:
    def __str__(self):
        return self.str
    
    def L(self):
        return expr(self.str)

class event(Utt):
    def __init__(self, e, args, c):
        self.str = e+"(e,"+",".join(args)+")"
        self.cost = c
        self.field = e
        self.elems = {self}
        self.assed = {"pred": e}

class role(Utt):
    def __init__(self, role, arg, c):
        self.str = role+"(e,"+arg+")"
        self.cost = c
        self.field = role
        self.elems = {self}
        self.assed = {role: arg}

class phrase(Utt):
    def __init__(self, utts):
        for utt in utts:
            assert isinstance(utt,Utt) #java
        self.str = " & ".join([utt.str for utt in utts])
        self.cost = sum([utt.cost for utt in utts])
    
        self.populate()
        self.assign()
    
    
    def populate(self):
        self.elems = set()
        for utt in utts:
            self.elems.update(utt.elems)
        
    def assign(self):
        self.assed = dict()
        for utt in self.elems:
            self.assed.update(utt.assed)
        
    def full_event_repr(self):
        r = False
        for u in self.elems:
            if type(u) == event: r = True
        return r

    def replace_constituent_in_utt(self, cons_utterance):
        #works for events and roles.. . . .. .
        #find role elem in elems with role.role == role
        f = cons_utterance.field
        for elem in self.elems:
            if (type(elem) == role and elem.field == f) or (type(elem)==event and elem.field == f:
                self.elems.remove(elem)
                self.elems.add(cons_utterance)
                #ugly
                #repl role in str
                self.str = self.str.replace(elem.str, cons_utterance.str)
                self.cost = sum([elem.cost for elem in self.elems])
        self.assign()
        
               
            
    def sub_utterances(self):
        combs = []
        for m in range(1,len(self.elems)+1):
            combs += list(itertools.combinations(bases, m))
        r = []
        for su in combs:
            r += [phrase(su)]
        return r


class Person():
    def __init__(self, wk, belief):
        self.wk = wk #list of fol expr
        self.belief = belief #fol expr
        self.assigned_roles =  list(self.belief.assed.keys()) #list of role objects (or role strings??

#What are states?

#L0 and L1 have same state set!!! cant suddenly infer qud in L1

#alle unterkombos von diff -> macht gar keinen sinn (basiert auf S1 wissen)
#alle quds um die es gehen könnte -> denk dir andere quds aus

#agrees or disagrees?

class PragmaticListener():
    @Marginal
    def evaluate_semantics(self, utterance):
        evaluation = p9.Prover9Command(utteranceL(), [ass.L() for ass in self.wk]+[self.belief.L()]).prove()
    
    

class PragmaticSpeaker():
    """
    processes because  we are looking at what the speaker does when he heard a non praggo input
    """
    
    
    def process(self, given_full_belief):
        evaluation = p9.Prover9Command(given_full_belief.L(), [ass.L() for ass in self.wk]+[self.belief.L()]).prove()
        #calc difference in roles
        utt_assigned_roles = list(given_full_belief.assed.keys())
        both_assigned = list(set(self.assigned_roles).intersection(set(utt_assigned_roles)))
        diff_roles = [] #differently assigned 
        for role in both_assigned:
            if self.assigned_roles[role] != utt_assigned_roles[role]:
                diff_roles.append(role)
        #what other person thinks differently phrased fully
        diff = []
        for const in given_full_belief.elems:
            for r in diff_roles:
                if const.field == r:
                    diff += [const]
        d = phrase(diff)
        #calc relevant to qud change
        
        possible_changers = d.sub_utterances()
               
#TODO make role inventory

uni_cost = 1

brutus = role("ag", "brutus", uni_cost)
caesar = role("pat", "caesar", uni_cost)
knife = role("ins", "knife", uni_cost)
stab = event("stab", ["brutus", "caesar"], uni_cost)
forum = role("loc", "forum", uni_cost)



said_by_prosecution = phrase([brutus, stab, caesar, forum])
print(said_by_prosecution)

#swk

#qud possible quds

quds = {
        "hang": expr("should_hang(brutus)")
        "mean": expr("mean(brutus")
        "lashed":expr("should_be_lashed(brutus)")
        "great": expr("great(brutus)")
}

def qud_prior():
    values = quds.keys()
    ix = pyro.sample("qud", dist.Categorical(probs=torch.ones(len(values))/len(values))
    return values[ix]

def utterance_cost():
    




k1 = expr("all x. exists e. kill(e,x,y), loc(e,rome) -> should_hang(x)")#killing in forum very illegal
k2 = expr("all x. exists e. kill(e,x,y) -> mean(x)")#killing is mean
k3 = expr("all x. exists e. ag(e,x), ins(e,knife) -> should_be_lashed(x)")#using knife is slightly bad
k4 = expr("all x. exists e. loc(e,rubicon), ins(e,paddle) -> great(x)")#using paddle at rubicon is awesome






