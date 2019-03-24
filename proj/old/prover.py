import nltk



expr = nltk.sem.Expression.fromstring

#QUD: Brutus should be imprisoned
qud = expr("tb_imprisoned(b)")

#LAW: Anyone who kills someone else within Rome should be sentenced
swk_1 = expr("all x. (exists e. (kill(e) & ag(e,x) & loc(e,rome))) -> tb_imprisoned(x)")

#Shared World Knowledge

#Stabing kills
swk_2 = expr("all f. stab(f) -> kill(f)")

#Forum in Rome
swk_3 = expr("all g. loc(g, forum) -> loc(g, rome)")

#Rubicon not in Rome

swk_4 = expr("all h. loc(h, rubicon) -> -loc(h,rome)")


world = [swk_1, swk_2, swk_3, swk_4]


prover = nltk.Prover9()

#Discourse

#Uttered by Bob:

bob_utt = expr("exists i. exists y. (stab(i) & ag(i,b) & pat(i,c) & loc(i, forum) & brutal(i) & knife(y) & ins(i,y))")

#calulate qud value for this utterance (in Anns head)

bob_qud = prover.prove(qud, world + [bob_utt])
print(bob_qud)

def bobpinion(s):
    print(prover.prove(expr(s), world + [bob_utt]))


bobpinion("exists e. kill(e)")

def p(theorem, facts):
    knowledge = [expr(fact) for fact in facts]    
    print(prover.prove(expr(theorem), knowledge))

p("tb_imprisoned(b)", ["all x. (exists e.(kill(e) & ag(e,x) & loc(e,rome))) -> tb_imprisoned(x)", "kill(f) & ag(f,b) & loc(f, rome)"])
