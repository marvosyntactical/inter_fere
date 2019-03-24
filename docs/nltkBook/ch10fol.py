"""
 model cost of utterance:

IDEA A: 
    in utterance_prior() 
"""
c = 1

def utterance_prior():
    ix = pyro.sample("utt", dist.Categorical(probs=torch.ones(3)/3))
    return ["none", "some", "all"][ix]

class Utt:
    def __str__(self):
        return self.str

class event(Utt):
    def __init__(self, e, args):
        self.str = e + "(e,"+ ",".join(args)+")"
        self.cost = c
        self.elems = [self]

class role(Utt):
    def __init__(self, role, arg):
        self.str = role+"(e,"+arg+")"
        self.cost = c
        self.elems = [self]

class phrase(Utt):
    def __init__(self, utt1, utt2):
        assert isinstance(utt1, Utt)
        assert isinstance(utt2, Utt)
        self.str = utt1.str + " /\ " + utt2.str
        self.cost = utt1.cost + utt2.cost
        self.elems = utt1.elems + utt2.elems


brutus = role("agens", "brutus")
caesar = role("patiens", "caesar")
stab = event("stab", ["brutus", "caesar"])
forum = role("location", "forum")


brutus_stab = phrase(brutus, stab)
brutus_stab_caesar = phrase(brutus_stab, caesar)
brutus_stab_caesar_forum = phrase(brutus_stab_caesar, forum)
print(brutus_stab_caesar_forum.cost)
print(brutus_stab_caesar_forum)
print(brutus_stab_caesar_forum.elems)


