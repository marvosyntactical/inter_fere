from phrasal import *
import nltk

##### First some logic setup: nltk, phrasal

#try completely different settings, phrases, etc

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
someone_else = role("agn", "someone_else",2*c)


#the runtime of l_1 depends on len(beliefs)**2; try making this list shorter for quicker runtime!

beliefs = [
        phrase([brutus, stab, caesar, forum, knife]),
        phrase([brutus, stab, caesar, forum]),
        phrase([brutus, stab, caesar, knife]),
        phrase([brutus, stab, caesar, forum, sword]),
        phrase([brutus, stab, caesar, rubicon, sword]),
        phrase([brutus, stab, caesar, sword]),
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

#### Modify the below to get different distributions!

from comp_implt import *
import helpers
import time

s1_belief = beliefs[12]
given_statement = beliefs[0]

correction = phrase([rubicon])

qud = "hang"
alpha = 1.#try different values: 1.0: normal optimality; higher settings: sharper distribution

TIME = helpers.Timer()
smoketest = True#1: calculate small utterance prior in s_1; 0: calculate full big utterance prior in s_1
ext = "reverse"#useful plot file mnemonic
if __name__ == "__main__":

    s1_attorney = S(alpha, swk, beliefs, s1_belief, quds)
    Listener = L(alpha, swk, beliefs, given_statement, quds)

    with TIME("l0 calculation", x=True):
        #lit listener
        l0_info =  "Lit. Listener distribution.\n\n- Correction: "+str(correction)
        l0_dist = Listener.L0(correction)
        helpers.plotter(l0_dist, output="plots/l0_"+str(ext)+".png", addinfo=l0_info)
        helpers.plot_dist(l0_dist, output="plots/l0_prop_"+str(ext)+".png")

    with TIME("s1 calculcation", x=True):
        #speaker
        s1_info = "Speaker distribution.\n\n- "+"Speaker event belief: "+str(s1_belief)+"\n- "+"QUD: "+str(quds[qud])+"\n- alpha = "+str(alpha)
        s1_attorney_dist = s1_attorney.interject(given_statement,qud,s1_belief,smoke_s1=smoketest)
        helpers.plotter(s1_attorney_dist, output="plots/prag_speaker_"+str(ext)+".png",addinfo=s1_info)
        helpers.plot_dist(s1_attorney_dist, output="plots/prag_speaker_"+str(ext)+".png")

    with TIME("l1 calculation", x=True):
        #prag listener 
        l1_dist = Listener.L1(correction, smoke_test=smoketest)
        helpers.plotter(l1_dist, output="plots/prag_listener_4.png", addinfo="Prag. Listener distribution.\n\n- "+"Correction: "+str(correction)+"\n- alpha = "+str(alpha))

        helpers.plot_dist(prag_listener_dist, output="plots/prag_listener_"+str(ext)+".png")
