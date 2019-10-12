This will one day be the github for my verb semantics project 

**Pragmatic Modelling of Ellipses Using First Order Logic Semantic Representations**

# ```TODO``` October 31 is deadline

#### Fix memoize for multiple return values of inner function: at the moment; memoize doesnt seem to work for precomputed values, possible fixes: ☑ 
1. find out how to **memoize one return value, but not the other** (this probably means still computing the second return value every time which Im trying to avoid in the first place
2. ditch second return value and find out how to extract the utterance from value map, use this in reasoning ☑ 
Solution was to ditch the second return value, just memoize the hashingmarginal and fix utterance prior to not be a marginal function, everything else doesnt even need the logical form

#### NULL\_Utt ☑ 
- seems to work fine for the moment

#### Value Error in HashingMarginal.log\_probs\(\)
- happens ~~when speaker belief is identical to previous utterance and null utterance should be produced~~ also happens other times
- probably caused by search run max tries number too low and hashingmarginal not having seen something it should have before;
- -> **MOVING TO CLUSTER** running with high search tries, .5 alpha so every option should be encountered during mcmc
- something thats not speaker project, I get a Value Error in log\_prob because 1, True was not seen prior by the trace
- The error looks as follows:
```python3
1134 current self distribution? Categorical(logits: torch.Size([1]))
1135 current values map? OrderedDict([(0, False)])
1136 &&&&&&&&&&
1137 log_prob values in HashingMarginal:
1138 d:  Categorical(logits: torch.Size([1]))
1139 values_map:  [(0, 'False')]
1140 val:  True
1141 value_hash:  1
1142 
1143 Traceback (most recent call last):
1144   File "/home/marvin/.local/lib/python3.6/site-packages/pyro/poutine/trace_struct.py", line 136, in log_prob_sum
1145     log_p = site["fn"].log_prob(site["value"], *site["args"], **site["kwargs"])
1146   File "/home/marvin/Workspace/problang/inter_fere/proj/search_inference.py", line 123, in log_prob
1147     return d.log_prob(torch.tensor([list(values_map.keys()).index(value_hash)]))
1148 ValueError: 1 is not in list
1149 
1150 During handling of the above exception, another exception occurred:
1151 
1152 Traceback (most recent call last):
1153   File "<stdin>", line 95, in <module>
1154   File "/home/marvin/Workspace/problang/inter_fere/proj/comp_implt.py", line 16, in shawarma
1155     return HashingMarginal(Search(fn, max_tries=int(1e3)).run(*args)), fn(*args)
1156   File "/home/marvin/Workspace/problang/inter_fere/proj/search_inference.py", line 191, in run
1157     for i, vals in enumerate(self._traces(*args, **kwargs)):
1158   File "/home/marvin/Workspace/problang/inter_fere/proj/search_inference.py", line 179, in _traces
1159     yield tr, tr.log_prob_sum()
1160   File "/home/marvin/.local/lib/python3.6/site-packages/pyro/poutine/trace_struct.py", line 143, in log_prob_sum
1161     traceback)
1162   File "/home/marvin/.local/lib/python3.6/site-packages/six.py", line 692, in reraise
1163     raise value.with_traceback(tb)
1164   File "/home/marvin/.local/lib/python3.6/site-packages/pyro/poutine/trace_struct.py", line 136, in log_prob_sum
1165     log_p = site["fn"].log_prob(site["value"], *site["args"], **site["kwargs"])
1166   File "/home/marvin/Workspace/problang/inter_fere/proj/search_inference.py", line 123, in log_prob
1167     return d.log_prob(torch.tensor([list(values_map.keys()).index(value_hash)]))
1168 ValueError: Error while computing log_prob_sum at site 'listener':
1169 1 is not in list
1170 
1171 
1172 shell returned 1
1173 
1174 Press ENTER or type command to continue
```


The below is taken from the [official pyro github examples](https://github.com/pyro-ppl/pyro/tree/dev/examples/rsa)

## Rational Speech Acts (RSA) examples

This folder contains examples of reasoning about reasoning with nested inference
adapted from work by @ngoodman and collaborators.

- `generics.py`: Taken from [Probabilistic Language Understanding](https://gscontras.github.io/probLang/chapters/07-generics.html)
- `hyperbole.py`: Taken from [Probabilistic Language Understanding](https://gscontras.github.io/probLang/chapters/03-nonliteral.html)
- `schelling.py`: Taken from [ForestDB](http://forestdb.org/models/schelling.html)
- `schelling_false.py`: Taken from [ForestDB](http://forestdb.org/models/schelling-falsebelief.html)
- `search.py`: Inference algorithms used in the example models. Adapted from [Design and Implementation of Probabilistic Programming Languages](http://dippl.org/chapters/03-enumeration.html)
- `semantic_parsing.py`: Taken from [Design and Implementation of Probabilistic Programming Languages](http://dippl.org/examples/zSemanticPragmaticMashup.html)
