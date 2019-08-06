from phrasal import *
import time
import nltk

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

#gladiator(x): beliefs not expressible in phrases
k1 = expr("all x. exists e. kill(e,x,y) & in(e,rome) -> should_hang(x)")#killing in forum very illegal
k2 = expr("all x. exists e. kill(e,x,y) -> mean(x)")#killing is mean
k3 = expr("all x.(exists e. (agn(e,x) & ins(e,knife)) -> should_be_lashed(x))")#using knife is slightly bad
k4 = expr("all x. exists e. loc(e,rubicon) & ins(e,paddle) -> great(x)")#using paddle at rubicon is awesome

f1 = expr("all e. loc(e,forum) -> in(e, rome)")
f2 = expr("all e. loc(e,rubicon) -> -in(e, rome)")
f3 = expr("all e. stab(e,x,y) -> kill(e,x,y)")

swk = [k1,k2,k3,k4,f1,f2,f3]

quds = {
        "hang": expr("should_hang(brutus)"),
        "mean": expr("mean(brutus)"),
        "lashed": expr("should_be_lashed(brutus)"),
        "great": expr("great(brutus)"),
}

from comp_implt import *
defense_belief = phrase([brutus, stab, caesar, rubicon, sword])

teste = event("sleep", ["brutus"], c)
testr = role("agn", "brutus",c)
testp = phrase([testr, teste])
altp = expr("exists e.(agn(e,brutus) & sleep(e,brutus))")
print(testp)
print(altp)
testk1 = expr("(exists e.(agn(e,x) & sleep(e,x))) -> inbed(x)")
testg = expr("inbed(brutus)")
print("\n t-knowledge ",testk1)
print("\n t-goal ",testg)
print("\n t-proof ", tpc(goal=testg, assumptions=[altp,testk1]).prove(verbose=False))


t1p = expr("dog(bello)")
t1k1 = expr("all x.(dog(x) -> dumb(x))")
t1g = expr("dumb(bello)")
print(t1p, t1k1, t1g)
print("Test pt1: ", tpc(goal=t1g, assumptions=[t1p, t1k1]).prove(verbose=False))

defense_attorney = S(swk, defense_belief, quds, beliefs)
t = time.time()
defense_attorney.interject(last_statement_made, "mean")
print("interjection calc time: ", str(time.time()-t))
#for obj in locals().values():
#        print(obj, "\n")

from nltk.inference import TableauProverCommand as tpc

print("hase")
qudSelf = tpc(goal=quds["lashed"], assumptions=swk+[defense_belief.L()]).prove(verbose=False)
print("qudSelf: ", qudSelf)
qudOther = tpc(goal=quds["lashed"], assumptions=swk+[last_statement_made.L()]).prove(verbose=False)
print("qudOther: ", qudOther)
#TODO
#make production priors depend on utt cost
#1) Phrasal proof funktioniert nicht
#2) HashingMarginal Object umgehen (replace /make selfargs see through)
#3) infinite loop because ###
