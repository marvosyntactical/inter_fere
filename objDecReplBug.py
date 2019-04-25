



def Marginal(fn):
    def wrapper(*args):
        print("1 wrap to go please sir")
        return HM(Google(fn).run(*args))
    return wrapper

class Attrb:
    def __init__(self):
        self.a = "something"

class TD:
    def __init__(self):
        self.v = "am traceposterior"
        
class Google(TD):
    def __init__(self, model_fn):
        self.model_fn = model_fn
        self.g = "am google"
    def run(self, *args):
        weird_obj = [*args]+[42]
        print("weird object run")
        return weird_obj

class HM:
    def __init__(self,trace_distribution):
        print("trace_dist arg in HM: ", trace_distribution)

@Marginal
def attrbprocessor(attrb):
    print(attrb.a, "geh heim")
    return attrb



utterance = Attrb()
u=attrbprocessor(utterance)
print(u.a)




