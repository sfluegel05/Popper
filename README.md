
### Popper

Popper is an [inductive logic programming](https://arxiv.org/pdf/2008.07912.pdf) system. Popper learns logical rules from examples and background knowledge.

Ask questions on [Discord](https://discord.gg/Rv5mQCayAp) or email [Andrew Cropper](mailto:andrew.cropper@helsinki.fi).

If you use Popper, please cite the paper [learning programs by learning from failures](https://arxiv.org/abs/2005.02259) (MLJ 2021).


**Requirements**
- [SWI-Prolog](https://github.com/SWI-Prolog/swipl-devel) (`brew install swi-prolog`)
- GNU Coreutils (`brew install coreutils`)
- [uv](https://github.com/astral-sh/uv) package manager (`brew install uv`)

**Install and run**

```
git clone https://github.com/logic-and-learning-lab/Popper.git
cd Popper
```
Then run via `uv run popper.py <input dir>` such as `uv run popper.py examples/iggp-rps-next-score`


**Example problem**

Popper requires three input files:
- an examples file
- a background knowledge (BK) file
- a bias file

An examples file contains positive and negative examples of the relation you want to learn:

```prolog
pos(grandparent(ann,amelia)).
pos(grandparent(steve,amelia)).
pos(grandparent(ann,spongebob)).
neg(grandparent(amy,amelia)).
```

A BK file contains other information about the problem:

```prolog
mother(ann,amy).
mother(ann,andy).
mother(amy,amelia).
mother(linda,gavin).
father(steve,amy).
father(steve,andy).
father(gavin,amelia).
father(andy,spongebob).
```

A bias file defines the hypothesis space. The following statements tell Popper which predicate symbols it can use in the head (`head_pred`) or body (`body_pred`)  of a rule:
```prolog
head_pred(grandparent,2).
body_pred(mother,2).
body_pred(father,2).
```
These say that Popper can use the symbol *grandparent* with two arguments in the head of a rule and *mother* or *father* in the body, also each with two arguments.

**Noise**

Popper can learn from [noisy](https://arxiv.org/pdf/2308.09393.pdf) data with the `--noisy` flag. Popper learns a minimal description length hypothesis.

**Seeding the search with a hypothesis**

When you already have a good guess at the answer, you can give it to Popper with the `--best-hypothesis` (`-b`) flag, which requires `--noisy`. Popper scores the hypothesis, uses it as its initial best hypothesis, and uses its MDL score as an upper bound on the score of any hypothesis it still needs to consider. The tighter this bound is, the more of the search space Popper prunes.

The flag takes either a file of rules or the rules themselves:

```
uv run popper.py examples/noisy-molecule --noisy -b examples/noisy-molecule/hypothesis.pl
uv run popper.py examples/noisy-molecule --noisy -b 'active(A):- has_atom(A,B),carbon(B),charged(B).'
```

The rules must follow the bias file, i.e. they may only use the declared head and body predicates and may not exceed `max_vars` or `max_body`. Popper ignores a hypothesis which is no better than the empty hypothesis and never returns a worse hypothesis than the one you give it.

The `examples/noisy-molecule` problem shows what this buys you. The target concept is a chain of eight conditions and nearly every negative example is a near miss which satisfies seven of them, so Popper has to search all the way to eight body literals before it finds anything good. The hypothesis in `examples/noisy-molecule/hypothesis.pl` is the target concept with one condition missing, which is the kind of guess a chemist might make: it scores an MDL of 18, whereas Popper starts from a bound of 303 (the number of positive examples) and does not find anything that good on its own for well over a minute.

Giving Popper the guess makes it search faster for as long as it has not yet found something equally good. In a 60 second run it considered 86,096 hypotheses instead of 68,319 and its mean testing time per hypothesis fell by 40%, because the bound lets it stop counting negative examples early and keeps all but the most promising rules out of the combine stage. Seeding never makes Popper return a worse hypothesis, so it is also a floor on the answer you get when the search times out.

**Steering the search towards predicates**

Seeding needs a whole hypothesis. When you only know which predicates are likely to matter, you can bias the *order* in which Popper considers rules with `prefer_body_pred/2` in the bias file:

```prolog
prefer_body_pred(long,10).          % try rules containing long first
prefer_body_pred(three_wheels,10).
prefer_body_pred(roof_open,-10).    % leave roof_open for later
```

A positive level makes Popper try to put the predicate into a rule first, a negative one makes it try to leave it out first, and a larger absolute level wins over a smaller one. This only reorders rules **of the same size**: Popper still enumerates all rules of size 2 before any rule of size 3, so the search stays complete and the answer is unchanged. What changes is *when* a good rule turns up within each size, which matters with `--noisy` because Popper prunes against the best MDL score it has found so far — a good rule found early makes the rest of that size and every later size cheaper to search.

These declarations compile to clingo [domain heuristics](https://potassco.org/clingo/) over Popper's internal `body_literal/4` atoms. You can also write such directives by hand in the bias file if you want finer control, e.g. to prefer a predicate only in some argument positions:

```prolog
#heuristic body_literal(C,long,1,Vars) : clause(C), vars(1,Vars). [10,true]
```

Popper's own size heuristic uses levels just below 1000, so keep yours below that or you will break the size ordering; `prefer_body_pred/2` rejects levels outside -999..999. Hints apply to the non-recursive generator only: with recursion or predicate invention Popper does not run clingo with `--heuristic=Domain`, so they have no effect.

`examples/gadget-hints` shows the effect. The target is a gadget with a part that is red, heavy, shiny and rough at once, every part of a negative example has all but one of those properties, and the bias file declares 60 further part properties that are pure noise. Finding the target takes 89s and 361,946 hypotheses without the hints, and 46s and 174,849 hypotheses with them. Comment the `prefer_body_pred` lines out of its bias file to see both.

Hints pay off when irrelevant predicates dominate the branching factor. They do much less when the relevant predicates alone still span a large space: on `examples/noisy-molecule`, hinting the seven chemically meaningful relations out of fourteen changed the enumeration order but not the running time, because seven relations over three variables still generate more rules of each size than Popper gets through. Steering also cannot make Popper prune *more*: it enumerates strictly by increasing size, so every constraint learned at one size is already in place before the next size starts, whatever order the rules came in.

**Turning off the size ordering (experimental)**

Popper enumerates rules in increasing size because of one clingo directive, `#heuristic size(N). [1000-N,true]`. The `--no-size-order` flag drops it, so the solver picks its own order. This is worth knowing about but it is not a speed-up: it makes Popper much slower, and it forfeits the guarantee that the hypothesis returned is a smallest (or MDL-optimal) one.

| | in size order | with `--no-size-order` |
|---|---|---|
| `examples/gadget-hints` | solved in 46s, 174,849 hypotheses | no solution in 240s, 584,650 hypotheses |
| `examples/noisy-molecule` | mdl 18 after 92s, 136,257 hypotheses | nothing found in 120s, 69,187 hypotheses |

The reason is that Popper's pruning is bottom-up. When a rule is refuted, the constraint it yields rules out that rule's *specialisations* — the bigger rules built on top of it. Without the size heuristic the solver goes straight for maximum-size rules, and refuting one of those prunes almost nothing, so the search degenerates into brute force: on `gadget-hints` it got through 3.3 times as many hypotheses as the ordered run needed and still did not find the answer. Bigger rules are also slower to test, which is why the noisy run managed only half the throughput.

The flag only applies without recursion or predicate invention. With either of those, `gen_rec.py` and `gen_pi.py` solve one size at a time through an `#external size_in_literals` atom, so the ordering is structural rather than a heuristic and Popper rejects the flag.

**Settings**

 - `--noisy`, `-n` learn from [noisy](https://arxiv.org/pdf/2308.09393.pdf) data using an MDL cost function (default: false)
 - `--best-hypothesis F`, `-b F` seed the search with a hypothesis, given as a file of rules or as the rules themselves (requires `--noisy`, default: none)
 - `--no-size-order` EXPERIMENTAL: do not enumerate rules in increasing size; loses the smallest/optimal guarantee and is slower (default: false)
 - `--max-vars N` maximum number of variables in a rule (default: 6)
 - `--max-body N` maximum number of body literals in a rule (default: 10)
 - `--timeout N` maximum learning time in seconds (default: 3600)
 - `-v`, `-vv`, `-vvv` increase verbosity
 - `--nuwls` use the NuWLS solver (default: false)

See [search-guidance.md](search-guidance.md) for how these two features were measured, including how the hinted predicates were chosen and where they did not help.

**Solvers**

Popper uses the [CPSAT](https://drops.dagstuhl.de/storage/00lipics/lipics-vol280-cp2023/LIPIcs.CP.2023.3/LIPIcs.CP.2023.3.pdf) solver by default for its combine stage.
Popper also supports the [NuWLS](https://ojs.aaai.org/index.php/AAAI/article/view/25505) anytime MaxSAT solver. You can download and compile this solver from the [MaxSAT 2023 evaluation](https://maxsat-evaluations.github.io/2023/descriptions.html) website. **We strongly recommend using  NuWLS** as it greatly improves the performance of Popper. To use them, ensure that the solver is available on your path.  See the [install solvers](solvers.md) file for help.


**Recursion**

Popper can learn recursive rules (where a predicate symbol appears in both the head and body), such as to find a duplicate element (`uv run popper.py examples/synthesis-finddupl`) in a list:
```prolog
f(A,B):- tail(A,C),head(A,B),element(C,B).
f(A,B):- tail(A,C),f(C,B).
```
To enable recursion, add `enable_recursion.` to the bias file. However, recursion is expensive, so it is best to avoid it if possible.

**Types**

Popper supports type annotations in the bias file. A type annotation is of the form `type(p,(t1,t2,...,tk)` for a predicate symbol `p` with arity `k`, such as:

```prolog
type(head,(list,element)).
type(tail,(list,list)).
type(length,(list,int,)).
type(empty,(list,)).
type(prepend,(element,list,list)).
```
Types are **optional** but can substantially reduce learning times.

**Directions**

Prolog often requires arguments to be ground. For instance, when asking Prolog to answer the query:
```prolog
X is 3+K.
```
It throws an error:
```prolog
ERROR: Arguments are not sufficiently instantiated
```
To avoid these issues, Popper supports **optional** direction annotations. A direction annotation is of the form `direction(p,(d1,d2,...,dk)` for a predicate symbol `p` with arity `k`, where each `di` is either `in` or `out`. An `in` variable must be ground when calling the relation. By contrast, an `out` variable need not be ground. Here are example directions:

```prolog
direction(head,(in,out)).
direction(tail,(in,out)).
direction(length,(in,out)).
direction(prepend,(in,in,out)).
direction(geq,(in,in)).
```

Popper cannot learn with partial directions. If you provide them, you must provide them for all relations.

**Performance tips**

- Transform your BK to Datalog, which allows Popper to perform preprocessing on the BK
- Try the NuWLS anytime solver
- Use 7 variables or fewer
- Avoid recursion and predicate invention
