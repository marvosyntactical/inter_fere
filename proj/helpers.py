from contextlib import contextmanager
from cProfile import Profile
from pstats import Stats
import time
from nltk.inference import ResolutionProverCommand as rpc
from search_inference import memoize, HashingMarginal, memoize, Search
from functools import wraps
import matplotlib.pyplot as plt

VARA = 1
def FUNA(args):
    return args

def Marginal(fn):
    @wraps(fn)
    def shawarma(*args):
        return HashingMarginal(Search(fn, max_tries=int(1e3)).run(*args))
    return memoize(shawarma)

def Memo(fn):
    @wraps(fn)
    def yufka(*args):
        return fn(*args)
    return memoize(yufka)


class wrapped_rpc(rpc):
    def __init__(self, *args, **kwargs):
        super(wrapped_rpc, self).__init__(*args, **kwargs)

    @Memo
    def prove(self, verbose=False):
        """
        Perform the actual proof.  Store the result to prevent unnecessary
        re-proving.
        """
        if self._result is None:
            self._result, clauses = self._prover._prove(
                self.goal(), self.assumptions(), verbose
            )
            self._clauses = clauses
            self._proof = wrapped_rpc._decorate_clauses(clauses)
        return self._result

def plotter(d, output="plots/distplot.png", addinfo=None):
    """
    pyplot plotting function for lit list, prag speak, prag list

    Args:
        d: pyro HashingMarginal distribution with phrase values
        output: output directory/path/file.png
        addinfo: None or string to be put below plot
    Returns:
        output: output directory/path/file.png
    """

    support = [str((str(value), value.cost)) for value in d.enumerate_support()]

    data = [d.log_prob(s).exp().item() for s in d.enumerate_support()]
    #plt.gca().set_position((.1, .3, .8, .6))

    ax = plt.subplot(111)
    width=0.1
    bins = list(map(lambda x: x-width/2,range(1,len(data)+1)))
    ax.bar(bins,data,width=width)
    ax.set_xticks(list(map(lambda x: x, range(1,len(data)+1))))
    ax.set_xticklabels(support,rotation=90, rotation_mode="anchor", ha="right")

    dimensionmetadata = [("support", len(support)), ("data", len(data)), ("bins", len(bins))]

    if type(addinfo)==str:
        metadata = str(dimensionmetadata) + " " + addinfo
    else:
        metadata = str(dimensionmetadata)
    print("metadata: ", metadata)
    plt.figtext(.1,10, metadata)

    #plt.tight_layout()
    if output == "show":
        pass
    else:
        plt.savefig(output)
        return output



class ProfileToFile(Profile):
    """Profiler class with __call__()able contet manager from stackoverflow"""
    def __init__(self, *args, **kwargs):
        super(Profile, self).__init__(*args, **kwargs)
        self.disable()

    @contextmanager
    def __call__(self,info, f=None):
        self.enable()
        yield
        print(info)
        self.disable()
        if type(f)==str:
            self.dumpstats(f)

class Timer(object):
    def __init__(self, *args, **kwargs):
        super(object, self).__init__(*args, **kwargs)

    @contextmanager
    def __call__(self, info):
        t = time.time()
        yield
        dt = time.time()-t
        print("Time spent in {} Timer: ".format(str(info), str(dt)))
