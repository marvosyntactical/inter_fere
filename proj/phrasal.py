import itertools
import copy
import nltk

expr = nltk.sem.Expression.fromstring

inventory = {"ins","agn","pat","loc","thm","gol","ass"}

NULL = "NULL"

class Utt:
    def __init__(self):
        self.str = ""
    def __str__(self):
        return self.str

    def L(self):
        return expr(self.str)
    def __hash__(self):
        return hash(self.str)

class NULL_Utt(Utt):
    def __init__(self):
        self.str = NULL
        self.cost = .5#finetune manually? for sensible model, probably 0<c<1
        self.field = "None"
        self.elems = {self}
        self.assed = {"NULL": "NIL"}

class event(Utt):
    def __init__(self, e, args, c):
        self.str = e+"(e,"+",".join(args)+")"
        self.cost = c
        self.field = e
        self.elems = {self}
        self.assed = {"pred": e}

class role(Utt):
    def __init__(self, role, arg, c):
        assert role in inventory
        self.str = role+"(e,"+arg+")"
        self.cost = c
        self.field = role
        self.elems = {self}
        self.assed = {role: arg}

class phrase(Utt):
    def __init__(self, utts):

        for utt in utts:
            assert isinstance(utt,Utt), str(utt)+" "+str(type(utt))
        self.str = "exists e."+" & ".join([utt.str for utt in utts])
        self.cost = sum([utt.cost for utt in utts])

        self.populate(utts)
        self.assign()


    def populate(self, utts):
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


    def replace_constituents_in_utt(self, phrs_utterance):

        if type(phrs_utterance) == NULL_Utt:
            return self

        consts = phrs_utterance.elems
        incr_new = self
        for c in consts:
            incr_new = incr_new.replace_constituent_in_utt(c)
        return incr_new

    def replace_constituent_in_utt(self, cons_utterance):
        """
        find role elem in elems with role.role == role
        works on deepcopy of self and returns it !! ! !! ! 
        """

        f = cons_utterance.field
        proxy = phrase(list[self.elems])
        for elem in proxy.elems:
            if (type(elem) == role and elem.field == f) or (type(elem)==event and elem.field == f):
                proxy.elems.remove(elem)
                proxy.elems.add(cons_utterance)
                proxy.str = proxy.str.replace(elem.str, cons_utterance.str)
                proxy.cost = sum([elem.cost for elem in self.elems])
        proxy.assign()
        return proxy


    def sub_utterances(self):
        combs = []
        for m in range(1,len(self.elems)+1):
            combs += list(itertools.combinations(self.elems, m))
        #empty utterance as option
        return [phrase(su) for su in combs]+[NULL_Utt()]

