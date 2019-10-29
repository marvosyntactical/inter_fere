from phrasal import *
import time
import nltk
import copy
from nltk.inference import ResolutionProverCommand as rpc
from comp_implt import *
import helpers

expr = nltk.sem.Expression.fromstring

print("Court.py started running...")

c = 1

stab = event("stab", ["brutus", "caesar"], c)
row = event("row", ["brutus", "boat"],c)

brutus = role("agn", "brutus", c)
caesar = role("pat", "caesar", c)
knife = role("ins", "knife", c)
forum = role("loc", "forum", c)
sword = role("ins", "sword", c)
rubicon = role("loc", "rubicon", c)
paddle = role("ins", "paddle", c)
boat = role("pat", "boat",c)
fun = role("gol", "fun",c)
others = role("ass", "others",c)
caesar_ag = role("agn", "caesar", c)
brutus_pat = role("pat", "brutus", c)
someone_else = role("agn", "someone_else",4*c)


beliefs = [
        phrase([brutus, stab, caesar, forum, knife]),#
        phrase([brutus, stab, caesar, forum]),#
        phrase([brutus, stab, caesar, knife]),#
        phrase([brutus, stab, caesar, forum, knife]),#
        phrase([brutus, stab, caesar, forum, sword]),#
        phrase([brutus, stab, caesar, forum, knife]),#
        phrase([brutus, stab, caesar, rubicon, sword]), 
        phrase([brutus, stab, caesar, rubicon]), 
        phrase([brutus, stab, caesar, sword]), 
        phrase([brutus, stab, caesar, sword, rubicon]), 
        phrase([brutus, stab, caesar, rubicon, knife]), 
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

defense_belief = beliefs[9]
last_statement_made = beliefs[0]

correction = phrase([rubicon])

qud = "hang"
alpha = 1.#finetune

TIME = helpers.Timer()
#P2F = helpers.ProfileToFile()

if __name__ == "__main__":

    defense_attorney = S(alpha, swk, beliefs, defense_belief, quds)
    prosecutor = L(alpha, swk, beliefs, last_statement_made, quds)

    with TIME("l0 calculation", x=False):
        #lit listener
        l0_info =  "Lit. Listener distribution.\n\n- Correction: "+str(correction)
        lit_listener_dist = prosecutor.L0(correction)
        helpers.plotter(lit_listener_dist, output="plots/lit_listener_1.png", addinfo=l0_info)
        helpers.plot_dist(lit_listener_dist, output="plots/lit_listener_2.png")

    with TIME("s1 calculcation", x=False):
        #speaker
        s1_info = "Speaker distribution.\n\n- "+"Speaker event belief: "+str(beliefs[1])+"\n- "+"QUD: "+str(quds[qud])+"\n- alpha = "+str(alpha)
        defense_attorney_dist = defense_attorney.interject(last_statement_made, qud, defense_belief,smoke_s1=False)
        helpers.plotter(defense_attorney_dist, output="plots/prag_speaker_1.png",addinfo=s1_info)
        helpers.plot_dist(defense_attorney_dist, output="plots/prag_speaker_2.png")

    with TIME("l1 calculation", x=True):
        #prag listener 
        prag_listener_dist = prosecutor.L1(correction, smoke_test=True)
        plotter(prag_listener_dist, output="plots/prag_listener_1.png", addinfo="Prag. Listener distribution.\n\n- "+"Correction: "+str(correction)+"\n- alpha = "+str(alpha))

        helpers.plot_dist(prag_listener_dist, output="plots/prag_listener_2.png")
