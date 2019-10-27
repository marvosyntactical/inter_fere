import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from phrasal import *
from helpers import *
rpc = wrapped_rpc
from search_inference import factor

expr = nltk.sem.Expression.fromstring


class L:
    def __init__(self,s1_alpha, swk, B, belief, QUDs):
        """
        Listener object holding information for both literal (L0) and pragmatic (L1) Listeners to
        use
        Args:
            s1_alpha: speaker optimality to be assumed by L1. Replace with prior over possible
            optimalities once inference works
            wk: list of nltk.sem.Expression FOL predicates. This is the common world knowledge of speaker and listener that both take the other to know
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
        self.belief = belief#self.belief == last_statement_made == given_full_belief 
        self.QUDs = QUDs

    @Marginal
    def L0(self, correction):
        """
        Literal listener only reasoning about literal semantics of replaced correction
        doesnt take into account qud, literally infers belief...by replacing subutterance
        """

        interjector_belief=belief_prior(self.B) #state_prior() in base RSA
        replacement = self.belief.replace_constituents_in_utt(correction)

        if not type(correction) == NULL_Utt:
            added_expr = replacement
        else:
            added_expr = self.belief
        evaluation = rpc(goal=interjector_belief.L(), assumptions=self.swk+[added_expr.L()]).prove()


        factor("literal meaning", 0. if evaluation else -99999999.)
        return interjector_belief

    @Marginal
    def L1(self, correction):

        """
        infers belief of s1
        only really needs to infer the relevant change as in replacement suggested in correction

        important TODO:
            pragmatic listener should select belief from B set

        """

        interjector_belief=belief_prior(self.B)#state_prior() in base RSA
        qud = qud_prior(self.QUDs)
        speaker = S(self.s1_alpha, self.swk, self.B, interjector_belief, self.QUDs)

        #v below self belief == last_statement_made, as always 
        speaker_marginal = speaker.interject(self.belief, qud)
        print("prago listo debuggo")
        print("speaker runs: ", len(speaker_marginal.trace_dist.exec_traces))
        print("inferred speaker dist: ", speaker_marginal._dist_and_values())
        pyro.sample("speaker", speaker_marginal, obs=correction)
        print("µ"*20, " pragmatic listener infers: ", interjector_belief, " ", "µ"*20)
        return interjector_belief


class S:
    def __init__(self, alpha, swk, B, belief, QUDs):
        """
        Listener object holding information for both literal (L0) and pragmatic (L1) Listeners to
        use
        Args:
            alpha: optimality. replace with prior over possible
            optimalities once inference works
            swk: list of nltk.sem.Expression FOL predicates. This is the common world knowledge of speaker and listener that both take the other to know
            B: list of phrasal.phrase phrases. this is the list of possible beliefs the speaker
            thinks the literal listener considers in the interpretation
            belief: phrasal.phrase object indicating listener belief. In the example discussed, this
            is equal to the last statement made upon which the speaker's correction calculation is
            done
            QUDs: dictionary with string keys and nltk.sem.Expression object values, holds questions
            under discussion the speaker may be referring to
        Returns:
            self
        """
        self.alpha = alpha
        self.swk = swk
        self.B = B
        self.belief = belief
        self.QUDs = QUDs
        self.qudSelf = None

    @timid
    @Memo
    def utterance_prior(self,given_full_belief):
        """
        My utterance prior is a function of speaker state/belief and given belief
        This is unlike in other RSA examples and a more general but far costlier to calculate prior
        should be put in place here
        Fixing this:
            utterance prior depends only on belief set; becomes large tho
        #TODO this method is a MESS
        """

        possible_changers = set()
        given_assigned = set(given_full_belief.assed.keys())


        for belief in self.B:
            assigned = set(belief.assed.keys())
            if assigned == {"NULL"}: #first leave out NULL utterances
                possible_changers.add(belief)
                continue
            diff_roles = [] #collect roles differently assigned in opponent
            for role in assigned:
                if role not in given_assigned:
                    diff_roles.append(role)
                elif belief.assed[role] != given_full_belief.assed[role]:
                    diff_roles.append(role)

            diff = [] #collect differing constituents together
            for const in belief.elems:
                for r in diff_roles:
                    if const.field == r:
                        diff += [const]
            if diff:
                d = phrase(diff) #difference in beliefs phrased fully
                combos = set(d.sub_utterances())
                possible_changers |= combos
        possible_changers = list(possible_changers)
        assert None not in possible_changers, "none in utt prior???"

        changerLogits = -torch.tensor([phr.cost for phr in possible_changers], dtype=torch.float64)
        ix = pyro.sample("utterance",dist.Categorical(logits=changerLogits))
        print(ix, len(possible_changers))
        r = possible_changers[ix.item()]
        return r

    @Marginal
    def interject(self, given_full_belief, qud):
        """
        The heart of the project
        Speaker decides wether to make a correction by inferring literal listener interpretation
        based on own question under discussion value
        and the opposing side's statement

        Args:
        """

        print("~"*20+"Speaker interject"+"~"*20+"\n")
        print("\ndebug: speaker initialized")
        print("qud: "+ qud)
        print("meaning: ", self.QUDs[qud])
        print("\ndebug: self.belief.L(): ", self.belief.L())
        print("\ndebug: given_full_belief: ", given_full_belief.L())
        print("\ndebug: self.swk: ", self.swk)

        self.qudSelf = rpc(goal=self.QUDs[qud], assumptions=self.swk+[self.belief.L()]).prove()


        with poutine.scale(scale=torch.tensor(float(self.alpha))):
            utterance = self.utterance_prior(given_full_belief)
            print("interject utterance prior: ", utterance)
            assert utterance != None

            #Construct Listener in head
            listener = L(self.alpha, self.swk, self.B, given_full_belief, self.QUDs)
            literal_marginal = listener.L0(utterance)
            projected_literal = self.project(literal_marginal, qud)
            print("projected_literal: ", projected_literal)
            pyro.sample("listener", projected_literal, obs=self.qudSelf)
            print("#"*20+" Speaker says: "+"#"*20+"\n")
            print(str(utterance), type(utterance))
            print("#"*20+"   /Speaker  "+"#"*20+"\n")
            print("~"*60)
        return utterance

    @Marginal
    def project(self,dist,qud):
        """
        projection helper function so a hashingmarginal distribution can be used in interjection inference
        the speaker uses this projection to consider whether the literal listeners sampled
        interpretation satisfies the speakers QUD
        """
        v = pyro.sample("proj",dist)
        print("sampled v : ", v, type(v))
        qud_interp = rpc(goal=self.QUDs[qud],assumptions=self.swk+[v.L()]).prove()

        return qud_interp


def belief_prior(B): #over list
    ix = pyro.sample("belief", dist.Categorical(probs=torch.ones(len(B))/len(B)))
    return B[ix]


def qud_prior(quds): #over dict
    keys = list(quds.keys())
    ix = pyro.sample("qud", dist.Categorical(probs=torch.ones(len(keys))/len(keys)))
    qud = keys[ix.item()]
    return qud

if __name__ == "__main__":
    pass

