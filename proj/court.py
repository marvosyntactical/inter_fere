from phrasal import *
import time
import nltk
import copy
from nltk.inference import ResolutionProverCommand as rpc
from comp_implt import *
import helpers

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


#how did it all happen??!?!?!
#no belief may be subset of another belief
beliefs = [
        phrase([brutus, stab, caesar, forum, knife]),#last statement made in ex
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

last_statement_made = phrase([brutus, stab, caesar, forum, knife])

defense_belief = phrase([brutus, stab, caesar, rubicon, sword])

qud = "hang"
alpha = 1.

TIME = helpers.Timer()
P2F = helpers.ProfileToFile()

defense_attorney = S(alpha, swk, beliefs, defense_belief, quds)
prosecutor = L(alpha, swk, beliefs, last_statement_made, quds)

with P2F("s1 and l1 calculation times", f="stats/calc1.stats"):

    defense_attorney_dist = defense_attorney.interject(last_statement_made, qud)
    plot_dist(defense_attorney_dist,
          output="plots/prag_speaker.png",addinfo="Speaker distribution.\n\n\t"+"Speaker event belief: "+str(defense_belief)+"\n\t"+str(quds[qud])+"\n\talpha = "+str(alpha))



    prag_listener_dist = prosecutor.L1(phrase([rubicon]))
    plot_dist(prag_listener_dist, output="plots/prag_listener.png", addinfo="Listener distribution.\n\n\t"+str(phrase([rubicon]))+"\n\talpha = "+str(alpha))

