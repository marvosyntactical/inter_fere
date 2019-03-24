from phrasal import *

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
k3 = expr("all x. exists e. ag(e,x) & ins(e,knife) -> should_be_lashed(x)")#using knife is slightly bad
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

defense_attorney = S(swk, defense_belief, quds, beliefs)
defense_attorney.interject(last_statement_made, "lashed")
