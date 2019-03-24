def marginal_dec(funct):
    def function_wrapper(*args):
        return ",".join([arg for arg in args]) + "; inference on arguments has been performed"
    return function_wrapper


def compute_belief(meaning, utterance, whatnot):
    print("belief computed!")
    return meaning, utterance, whatnot

#noob:

#construct composed function of marginal_dec.function_wrapper
my_participant = marginal_dec(compute_belief)
print(my_participant)
print(my_participant("no meaning in life :(", "fucking hell", "1010"))


#pro:
@marginal_dec
def pro_compute_belief(meaning, utterance, whatnot):
    print("belief computed!")
    return meaning, utterance, whatnot

print(pro_compute_belief("ubermensch lvl achieved!", "fucking heaven", "7070"))


