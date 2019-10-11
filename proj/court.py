from phrasal import *
import time
import nltk
#debug:
import copy
from nltk.inference import ResolutionProverCommand as rpc

expr = nltk.sem.Expression.fromstring

c = 1

brutus = role("agn", "brutus", c)
caesar = role("pat", "caesar", c)
knife = role("ins", "knife", c)
stab = event("stab", ["brutus", "caesar"], c)
forum = role("loc", "forum", c)
sword = role("ins", "sword", c)
rubicon = role("loc", "rubicon", c)
paddle = role("ins", "paddle", c)
row = event("row", ["brutus", "boat"],c)
boat = role("pat", "boat",c)
fun = role("gol", "fun",c)
others = role("ass", "others",c)
caesar_ag = role("agn", "caesar", c)
brutus_pat = role("pat", "brutus", c)
someone_else = role("agn", "someone_else",c)

last_statement_made = phrase([brutus, stab, caesar, forum, knife])

#how did it all happen??!?!?!
#no belief may be subset of another belief
beliefs = [
        last_statement_made,
        phrase([brutus, stab, caesar, rubicon, sword]),
        phrase([caesar_ag, stab, brutus_pat, forum, knife]),
        phrase([brutus, stab, caesar, rubicon, knife, fun]),
        phrase([brutus, stab, caesar, rubicon, sword, others, fun]),
        phrase([brutus, row, boat, rubicon, paddle]),
        phrase([someone_else, stab, caesar, rubicon, knife])
    ]

k1 = expr("all x.((exists e.(kill(e,x,y) & in(e,rome))) -> should_hang(x))")#killing in forum very illegal
k2 = expr("all x.((exists e.(kill(e,x,y))) -> mean(x))")#killing is mean
k3 = expr("all x.((exists e.(agn(e,x) & ins(e,knife))) -> should_be_lashed(x))")#using knife is slightly bad
k4 = expr("all x.((exists e.(loc(e,rubicon) & ins(e,paddle))) -> great(x))")#using paddle at rubicon is awesome

f1 = expr("all e.(loc(e,forum) -> in(e, rome))")
f2 = expr("all e.(loc(e,rubicon) -> -in(e, rome))")
f3 = expr("all e.(stab(e,x,y) -> kill(e,x,y))")

swk = [k1,k2,k3,k4,f1,f2,f3]

quds = {
        "hang": expr("should_hang(brutus)"),
        "mean": expr("mean(brutus)"),
        "lashed": expr("should_be_lashed(brutus)"),
        "great": expr("great(brutus)"),
}

#debug from here
from comp_implt import *

defense_belief = phrase([brutus, stab, caesar, rubicon, sword])
#defense_belief = last_statement_made #empty utterance?

def potenzmenge(l):
    combs = []
    for m in range(1, len(l)+1):
        combs += list(itertools.combinations(l, m))
    return combs

print("Constructing Potenzmenge for swk...")
pot = potenzmenge(swk)
invariants = copy.deepcopy(swk)

swk_str = "k1,k2,k3,k4,f1,f2,f3".split(",")

print("\n\n\n"+"=== Utterance combination debug ==="+"\n\n\n")

for combination in pot:
    try:
        defense_qud_self_ = rpc(goal=expr("should_be_lashed(brutus)"), assumptions=combination).prove()
    except RecursionError:
        print("Problematic combination of assumptions: "+ str([swk_str[i] if elem in combination else "  " for i, elem in enumerate(swk)]))
        invariants = [elem for elem in invariants if elem in combination]
print("Problematic invariants: "+str(invariants))
print("If True after colon ignore the above problems: ", invariants == swk)
print("\n\n\n"+"=== combo debug END END END ==="+"\n\n\n")


#last_statement_made = phrase([brutus, stab, caesar, forum, knife])

defense_attorney = S(swk, defense_belief, quds, beliefs)
t = time.time()
plot_dist(defense_attorney.interject(last_statement_made, "hang"))
print("interjection calc time: ", str(time.time()-t))


qudSelf = rpc(goal=quds["lashed"], assumptions=swk+[defense_belief.L()]).prove(verbose=False)
print("qudSelf: ", qudSelf)
exit()
qudOther = rpc(goal=quds["lashed"], assumptions=swk+[last_statement_made.L()]).prove(verbose=False)
print("qudOther: ", qudOther)
#TODO
#make production priors depend on utt cost
#1) Phrasal proof funktioniert nicht
#2) HashingMarginal Object umgehen (replace /make selfargs see through)
#3) infinite loop because ###
