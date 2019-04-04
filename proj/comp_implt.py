import nltk
from nltk.inference import TableauProverCommand
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine


from search_inference import factor, HashingMarginal, memoize, Search

def Marginal(fn):
    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))

expr = nltk.sem.Expression.fromstring
tpc = TableauProverCommand

class L:
    def __init__(self, wk, B, belief):
        self.wk = wk #shared world knowledge, list of expr
        self.B = B#belief set of speaker (for construction of belief prior)
        self.belief = belief#last statement made


    @Marginal
    def L0(self, correction):
        #doesnt take into account own belief or qud, literally infers belief...by replacing subutterance
        interjector_belief=belief_prior(self.B)
        if correction: #if no correction is made, literal listener is never even called
            replacement = self.belief.replace_constituent_in_utt(correction)
        evaluation = tpc(goal=interjector_belief, assumptions=self.wk+[correction]).prove()
        factor("literal meaning", 0. if evaluation else -99999999.)#condition on s1 belief, correctly infers belief in basic scenario, how do i get blue ambiguity?
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
    def __init__(self, wk, belief, QUDs, B):
        self.wk = wk
        self.belief = belief
        self.QUDs = QUDs
        self.B = B

    @Marginal
    def utterance_prior(self,given_full_belief):
        #why is my utterance prior a function of state ?????
        

        both_assigned = list(set(belief.assed.keys()).intersection(set(given_full_belief.assed.keys())))
        diff_roles = [] #differently assigned 
        for role in both_assigned:
            if belief.assed[role] != given_full_belief.assed[role]:
                diff_roles.append(role)
        diff = []
        for const in given_full_belief.elems:
            for r in diff_roles:
                if const.field == r:
                    diff += [const]
        d = phrase(diff) #difference in beliefs phrased fully
        possible_changers = d.sub_utterances()

        ix = pyro.sample("utterance",dist.Categorical(probs=torch.ones(len(possible_changers)) / len(possible_changers)))
        return possible_changers[ix]
        
    @Marginal
    def interject(self, given_full_belief, qud):
        #calc difference in roles
        #now: Which are relevant to qud change -> evaluate
        print("debug: speaker initialized,\n qud: "+ qud)
        print("debug: self.belief.L(): ", self.belief.L())
        print("debug: given_full_belief: ", given_full_belief.L())
        print("debug: self.wk: ", self.wk)
        
        

        qudSelf = tpc(goal=self.QUDs[qud], assumptions=self.wk+[self.belief.L()]).prove(verbose=False) 
        qudOther = tpc(goal=self.QUDs[qud], assumptions=self.wk+[given_full_belief.L()]).prove(verbose=False)
        print("debug: qudSelf: ", qudSelf, " qudOther: ", qudOther)
        agree = qudSelf == qudOther
        print("agreement?",agree)
        if agree: return ""
        
        alpha = 1.0
        with poutine.scale(scale=torch.tensor(alpha)):
            utterance = self.utterance_prior(given_full_belief)
            #Construct Listener in head
            listener = L(swk, self.B, given_full_belief)

            literal_marginal = L.L0(utterance)
            proj = pyro.sample("proj", literal_marginal)
            projected_literal = tpc(goal=self.QUDs[qud], assumptions=self.wk+[utterance]).prove()
            pyro.sample("listener", projected_literal, obs=qudSelf)
        return utterance
        
def belief_prior(B):
    ix = pyro.sample("belief", dist.Categorical(probs=torch.ones(len(B))/len(B)))
    return B[ix]

def qud_prior(quds):
    values = list(quds.keys())
    ix = pyro.sample("qud", dist.Categorical(probs=torch.ones(len(values))/len(values)))
    qud = values[ix.item()]
    return qud


