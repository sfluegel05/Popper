# Code and idea from the papers:
# Andrew Cropper, Céline Hocquette: Learning Logic Programs by Discovering Where Not to Search. AAAI 2023: 6289-6296
# Andrew Cropper, Filipe Gouveia, David M. Cerna: Honey, I shrunk the hypothesis space (through logical preprocessing). CoRR abs/2506.06739 (2025)

import gc
import string
from contextlib import contextmanager
import clingo
import numpy as np
from collections import defaultdict
from itertools import permutations, combinations
from . util import generate_binary_strings
from . import logger
from . import stats


def format_tuple(items):
    """Format items as a Clingo/Prolog tuple, e.g., (V0,V1) or (V0,)."""
    items = list(items)
    s = ','.join(map(str, items))
    return f'({s},)' if len(items) == 1 else f'({s})'

def format_args(items):
    """Format items as a Clingo/Prolog argument list, e.g., (V0,V1)."""
    return f"({','.join(map(str, items))})"

def canonicalise_vars(*var_seqs):
    """Rename variables to A, B, C... in first-appearance order across all sequences."""
    lookup = {}
    def get_var(v):
        if v not in lookup:
            lookup[v] = string.ascii_uppercase[len(lookup)]
        return lookup[v]
    return [tuple(get_var(v) for v in vs) for vs in var_seqs]

def connected(xs, ys):
    """Check if two variable sequences share at least one variable."""
    return not set(xs).isdisjoint(ys)

ALPHABET_CACHE = tuple(set(string.ascii_uppercase[:i]) for i in range(20))

def uses_in_order(xs, ys):
    """Check if variables are used in order (A, B, C...) without gaps."""
    zs = set(xs) | set(ys)
    return zs == ALPHABET_CACHE[len(zs)]

TIDY_OUTPUT = """
#defined body_literal/4.
#defined clause_var/2.
#defined var_type/3.
#defined clause/1.
"""

def prop_pairs(settings, arities):
    """Variable patterns (xs, ys) of the P(xs), Q(ys) literal pairs to check, e.g. (('A','B'), ('B','C'))."""
    myvars = tuple(string.ascii_uppercase[:settings.max_vars])

    pairs = set()
    for a1 in arities:
        xs = tuple(myvars[:a1])
        for a2 in arities:
            candidate_vars = myvars[:min(a1 + a2, settings.max_vars)]
            for ys in permutations(candidate_vars, a2):
                if not connected(xs, ys) or not uses_in_order(xs, ys):
                    continue

                # Canonicalize to handle symmetries
                longer, shorter = sorted([xs, ys], key=len, reverse=True)
                canon = canonicalise_vars(longer, shorter)
                pairs.add((canon[0], canon[1]))
    return sorted(pairs)

def build_props(settings, arities):
    """Generate potential properties (functional dependencies, etc.) between predicates."""
    props, cons = [], []
    for xs, ys in prop_pairs(settings, arities):
        xs_set, ys_set = set(xs), set(ys)
        left, right = ''.join(x.lower() for x in xs), ''.join(y.lower() for y in ys)

        t_left = format_tuple(f'T{x}' for x in xs)[1:-1]
        t_right = format_tuple(f'T{y}' for y in ys)[1:-1]
        atom_left = format_tuple(xs)[1:-1]
        ys_masked = [y if y in xs_set else '_' for y in ys]
        atom_right = format_tuple(ys_masked)[1:-1]

        # IMPLIES NOT: P(xs) => not Q(ys)
        key = f'not_{left}_{right}'
        sym_con = 'P<Q,' if xs == ys else ''

        l1 = f'prop({key},(P,Q)):- {sym_con} body_pred(P,{len(xs)}), body_pred(Q,{len(ys)}), type(P,({t_left})), type(Q,({t_right})), holds(P,({atom_left})), not {key}_aux((P,Q)).'
        l2 = f'{key}_aux((P,Q)):- {sym_con} body_pred(P,{len(xs)}), body_pred(Q,{len(ys)}), type(P,({t_left})), type(Q,({t_right})), holds(P,({atom_left})), holds(Q,({atom_right})).'
        props.extend([l1, l2])
        cons.append(f':- prop({key},(P,Q)), body_literal(Rule,P,_,({atom_left})), body_literal(Rule,Q,_,({atom_right})).')

        if not ys_set.issubset(xs_set):
            continue

        # IMPLIES: P(xs) => Q(ys) (only when ys variables are a subset of xs variables)
        key = f'{left}_{right}'
        sym_con = 'P!=Q,' if xs == ys else ''

        l1 = f'prop({key},(P,Q)):- {sym_con} body_pred(P,{len(xs)}), body_pred(Q,{len(ys)}), type(P,({t_left})), type(Q,({t_right})), holds(P,({atom_left})), holds(Q,({atom_right})), not {key}_aux((P,Q)).'
        l2 = f'{key}_aux((P,Q)):- {sym_con} body_pred(P,{len(xs)}), body_pred(Q,{len(ys)}), type(P,({t_left})), type(Q,({t_right})), holds(P,({atom_left})), not holds(Q,({atom_right})).'
        props.extend([l1, l2])

        checker = ','.join(f'valid_var(Rule,{v})' for v in ys_set)
        cons.append(f':- prop({key},(P,Q)), body_literal(Rule,P,_,({atom_left})), body_literal(Rule,Q,_,({atom_right})), {checker}.')

    return props, cons

def deduce_recalls(settings, solver):
    """Deduce maximum recalls for body predicates to prune search space."""
    if solver is None:
        return []

    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    counts_all = defaultdict(int)

    for pred, arity in settings.body_preds:
        binary_strings = generate_binary_strings(arity)
        for atom in solver.symbolic_atoms.by_signature(pred, arity=arity):
            counts_all[pred] += 1
            args = [str(arg) for arg in atom.symbol.arguments]
            for var_subset in binary_strings:
                key = tuple(args[i] for i, bit in enumerate(var_subset) if bit)
                counts[pred][var_subset][key] += 1

    all_recalls = {}
    for pred, arity in settings.body_preds:
        all_recalls[(pred, (0,) * arity)] = counts_all[pred]
        for var_subset, key_map in counts[pred].items():
            recall = max(key_map.values(), default=0)
            all_recalls[(pred, var_subset)] = recall

    settings.recalls.update(all_recalls)
    return recall_cons(all_recalls)

def recall_cons(all_recalls):
    """Constraints for the recalls up to 4 of {(pred, fixed argument mask): recall}."""
    out = []
    for (pred, key), recall in all_recalls.items():
        if recall > 4:
            continue
        arity = len(key)
        args = [f'V{i}' for i in range(arity)]
        subset_str = ','.join(v for v, k in zip(args, key) if k == 0)
        fixer_str = format_tuple(v if k else '_' for v, k in zip(args, key))[1:-1]
        args_str = format_tuple(args)[1:-1]
        
        con = f':- body_literal(Rule,{pred},_,({fixer_str})), #count{{{subset_str}: body_literal(Rule,{pred},_,({args_str}))}} > {recall}.'
        out.append(con)
    return out

def deduce_type_cons(settings, solver):
    """Deduce constraints based on the number of objects of each type."""
    if not settings.body_types or solver is None:
        return []

    type_objects = defaultdict(set)
    for pred, arity in settings.body_preds:
        for atom in solver.symbolic_atoms.by_signature(pred, arity=arity):
            for i, arg in enumerate(atom.symbol.arguments):
                arg_type = settings.body_types[pred][i]
                type_objects[arg_type].add(str(arg))

    cons = []
    for arg_type, objects in type_objects.items():
        n = len(objects)
        if n <= settings.max_vars:
            logger.debug(f'max_vars {arg_type} {n}')
            cons.append(f":- clause(C), #count{{V : var_type(C, V, {arg_type})}} > {n}.")
    return cons

def generate_singleton_encoding(max_size=4):
    """Programmatically generate the ASP encoding for finding total argument combinations."""
    lines = []
    for n in range(1, max_size + 1):
        suffix = str(n) if n > 1 else ""
        lines.append(f"#show total{suffix}/{n+1}.")

    for n in range(1, max_size + 1):
        suffix = str(n) if n > 1 else ""
        indices = [f"I{i}" for i in range(1, n + 1)]
        types = [f"T{i}" for i in range(1, n + 1)]
        vars = [f"X{i}" for i in range(1, n + 1)]
        
        idx_str = ", ".join(indices)
        type_str = ", ".join(f"type(P, {idx}, {t})" for idx, t in zip(indices, types))
        domain_str = ", ".join(f"domain({t}, {v})" for t, v in zip(types, vars))
        order_str = ", ".join(f"{indices[i]} < {indices[i+1]}" for i in range(n-1))
        if order_str: order_str += ", "
        
        pred_cond = f"pred(P, A), A > {n}, " if n > 1 else "pred(P, _), "
        
        # not_totalN: There exists a combination of domain values not present in the predicate
        not_total_rule = (f"not_total{suffix}(P, {idx_str}) :- {pred_cond}{order_str}{type_str}, "
                         f"{domain_str}, not holds{suffix}(P, {idx_str}, {', '.join(vars)}).")
        lines.append(not_total_rule)
        
        # totalN: For all domain values, a matching tuple exists in the predicate
        type_simple_str = ", ".join(f"type(P, {idx}, _)" for idx in indices)
        total_rule = (f"total{suffix}(P, {idx_str}) :- {pred_cond}{type_simple_str}, "
                      f"{order_str}not not_total{suffix}(P, {idx_str}).")
        lines.append(total_rule)
        
    return "\n".join(lines)

def _process_total_atoms(solver, signature, arity, pred_lookup, seen, cons):
    """Extract singleton constraints from grounded 'total' atoms."""
    for atom in solver.symbolic_atoms.by_signature(signature, arity=arity):
        p = str(atom.symbol.arguments[0])
        indices = [int(arg.number) for arg in atom.symbol.arguments[1:]]
        a = pred_lookup[p]
        args = [f'V{i}' for i in range(a)]
        total_vars = {args[idx] for idx in indices}
        singletons_checked = frozenset(v for v in args if v not in total_vars)
        
        if not singletons_checked or any(x.issubset(singletons_checked) for x in seen[p]):
            continue
            
        non_singletons = ','.join(f'singleton({v})' for v in singletons_checked)
        con = f':- body_literal(_, {p}, _, {format_args(args)}), {non_singletons}.'
        cons.append(con)
        seen[p].add(singletons_checked)

def deduce_non_singletons(settings, solver):
    """Find combinations of arguments that cover the whole domain to prune redundant rules."""
    if not settings.body_types:
        return []

    max_arity = max((a for p, a in settings.body_preds), default=0)
    max_size = min(max_arity, 4) # Limit to 4 as in the original code
    
    encoding = []
    for p, a in settings.body_preds:
        if a > 1:
            encoding.append(f'pred({p},{a}).')

        for i, t in enumerate(settings.body_types[p]):
            encoding.append(f'domain({t},X):- holds({p},{i},X).')
            encoding.append(f'type({p},{i},{t}).')

        args = [f'V{i}' for i in range(a)]
        arg_str = ','.join(args)
        
        for size in range(1, max_size + 1):
            pred_name = f"holds{size}" if size > 1 else "holds"
            for indices in combinations(range(a), size):
                idx_str = ','.join(map(str, indices))
                val_str = ','.join(args[i] for i in indices)
                encoding.append(f'{pred_name}({p},{idx_str},{val_str}):- {p}({arg_str}).')

    solver.add('singletons', [], '\n'.join(encoding + [generate_singleton_encoding(max_size)]))
    solver.ground([('singletons', [])])

    pred_lookup = {p: a for p, a in settings.body_preds}
    seen, cons = defaultdict(set), []
    for n in range(max_size, 0, -1):
        suffix = str(n) if n > 1 else ""
        _process_total_atoms(solver, f'total{suffix}', n + 1, pred_lookup, seen, cons)
    return cons

def read_bk_facts(bk_string, body_preds):
    """Body-predicate facts of a facts-only BK as tuples of interned ints, or None if the BK holds anything
    else (rules, directives, quoted or compound terms), whose consequences only clingo can derive."""
    arity = dict(body_preds)
    ids = {}
    facts = defaultdict(set)
    for line in bk_string.splitlines():
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if not line.endswith(').') or ':-' in line or '"' in line or "'" in line:
            return None
        name, _, rest = line[:-2].partition('(')
        if '(' in rest or ')' in rest or ' ' in name:
            return None
        if name not in arity:
            continue
        args = [a.strip() for a in rest.split(',')]
        if len(args) == arity[name]:
            facts[name].add(tuple(ids.setdefault(a, len(ids)) for a in args))
    return facts

@contextmanager
def gc_paused():
    """Pauses Python's cyclic GC, which otherwise rescans the millions of fact tuples of a large BK on every
    collection (226 s instead of 8 s to parse 9M facts)."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()

def fact_arrays(facts, arity, preds):
    """The facts of each pred as an int64 array of shape (facts, arity)."""
    return {p: np.array(list(facts.get(p, ())), dtype=np.int64).reshape(len(facts.get(p, ())), arity[p])
            for p in preds}

def _num_ids(arrays):
    return 1 + max((int(a.max()) for a in arrays.values() if a.size), default=0)

def _row_codes(cols, n_ids):
    """One int64 per row of cols (values below n_ids), equal iff the rows are equal."""
    code = cols[:, 0].copy()
    for k in range(1, cols.shape[1]):
        if int(code.max(initial=0)) + 1 > (2**62) // n_ids:
            code = np.unique(code, return_inverse=True)[1].reshape(-1).astype(np.int64)
        code = code * n_ids + cols[:, k]
    return code

def pointless_from_facts(settings, facts):
    """Tester.find_pointless_relations on the parsed facts of a facts-only BK: preds without facts, and all
    but one pred of each group with identical facts. Unlike the Prolog version it sees only this BK, not
    whatever else Prolog has loaded (settings.prolog_bk_files)."""
    body_types = settings.body_types
    pointless = {(p, pa) for p, pa in settings.body_preds if not facts.get(p)}
    missing = {p for p, _ in pointless}
    preds = [(p, pa) for p, pa in settings.body_preds if p not in missing]
    keep = set()
    for a, b in combinations(preds, 2):
        (p, pa), (q, qa) = a, b
        if pa != qa:
            continue
        if body_types and body_types.get(p) != body_types.get(q):
            continue
        if a in pointless and b in pointless:
            continue
        if facts[p] != facts[q]:
            continue
        if a not in pointless and b not in pointless:
            if a in keep:
                pointless.add(b)
            elif b in keep:
                pointless.add(a)
            else:
                keep.add(a)
                pointless.add(b)
        else:
            if a not in keep:
                pointless.add(a)
            if b not in keep:
                pointless.add(b)
    return pointless

def recalls_from_facts(settings, facts):
    """deduce_recalls on the parsed facts of a facts-only BK, without grounding it in clingo."""
    arity = dict(settings.body_preds)
    arrays = fact_arrays(facts, arity, arity)
    n_ids = _num_ids(arrays)
    all_recalls = {}
    for pred, a in settings.body_preds:
        arr = arrays[pred]
        all_recalls[(pred, (0,) * a)] = len(arr)
        if not len(arr):
            continue
        for var_subset in generate_binary_strings(a):
            cols = arr[:, [i for i, bit in enumerate(var_subset) if bit]]
            counts = np.unique(_row_codes(cols, n_ids), return_counts=True)[1]
            all_recalls[(pred, var_subset)] = int(counts.max())
    settings.recalls.update(all_recalls)
    return recall_cons(all_recalls)

def projection_relations(facts, arity, projections):
    """Overlap and containment between projections (pred, positions) of the facts onto argument positions.
    Gives index, meet, sub: meet[i,j] iff projections i and j share a tuple, sub[i,j] iff i is contained in j.

    Every tuple gets a bitmask of the projections holding it. Two projections meet iff some mask has both
    bits, and i is contained in j iff every mask with bit i has bit j, so only distinct masks matter."""
    arrays = fact_arrays(facts, arity, {p for p, _ in projections})
    n_ids = _num_ids(arrays)
    index, meet, sub = {}, {}, {}
    by_len = defaultdict(list)
    for pr in sorted(projections):
        by_len[len(pr[1])].append(pr)
    for L, prs in by_len.items():
        code = _row_codes(np.concatenate([arrays[p][:, list(pos)] for p, pos in prs]), n_ids)
        rows = np.unique(code, return_inverse=True)[1].reshape(-1)
        del code
        words = (len(prs) + 63) // 64
        masks = np.zeros((int(rows.max(initial=-1)) + 1, words), dtype=np.uint64)
        start = 0
        for j, (p, _) in enumerate(prs):
            end = start + len(arrays[p])
            masks[rows[start:end], j >> 6] |= np.uint64(1 << (j & 63))
            start = end
        del rows
        distinct = np.unique(masks.view(np.dtype((np.void, 8 * words))).reshape(-1))
        del masks
        bits = np.unpackbits(np.frombuffer(distinct.tobytes(), dtype=np.uint8).reshape(len(distinct), 8 * words),
                             axis=1, bitorder='little')[:, :len(prs)].astype(np.float32)
        both = bits.T @ bits
        own = np.diag(both)
        for j, pr in enumerate(prs):
            index[pr] = (L, j)
        meet[L] = both > 0
        sub[L] = both == own[:, None]
    return index, meet, sub

def deduce_bk_props(settings, solver, facts=None):
    """Deduce properties (props) from overlap and containment of the BK's fact projections; falls back to
    clingo for BKs that are not plain facts. Gives the same props as deduce_bk_props_clingo."""
    if facts is None:
        facts = read_bk_facts(settings.bk_string, settings.body_preds)
    if facts is None:
        return deduce_bk_props_clingo(settings, solver)

    preds = sorted(settings.body_preds)
    arity = dict(preds)
    if settings.body_types:
        types = {p: settings.body_types.get(p) for p, _ in preds}
    else:
        types = {p: ('t',) * a for p, a in preds}

    arities = {a for _, a in preds}
    checks = []  # (prop name, p, q, disjoint or subset, projection of p, projection of q)
    for xs, ys in prop_pairs(settings, arities):
        shared = [v for v in ys if v in xs]
        pos_p = tuple(xs.index(v) for v in shared)
        pos_q = tuple(ys.index(v) for v in shared)
        left = ''.join(x.lower() for x in xs)
        right = ''.join(y.lower() for y in ys)
        implies = set(ys) <= set(xs)
        for p, pa in preds:
            if pa != len(xs) or not facts[p] or types[p] is None:
                continue
            for q, qa in preds:
                if qa != len(ys) or types[q] is None:
                    continue
                if any(types[p][i] != types[q][j] for i, j in zip(pos_p, pos_q)):
                    continue
                # P(xs) and Q(ys) never agree on the shared variables
                if xs != ys or p < q:
                    checks.append((f'not_{left}_{right}', p, q, False, (p, pos_p), (q, pos_q)))
                # every P(xs) has a matching Q(ys)
                if implies and (xs != ys or p != q):
                    checks.append((f'{left}_{right}', p, q, True, (p, pos_p), (q, pos_q)))

    index, meet, sub = projection_relations(facts, arity, {pr for c in checks for pr in c[4:]})
    props = []
    for name, p, q, implies, pr_p, pr_q in checks:
        g, i = index[pr_p]
        j = index[pr_q][1]
        if sub[g][i, j] if implies else not meet[g][i, j]:
            props.append(f'prop({name},({p},{q})).')

    _, new_cons = build_props(settings, arities)
    return props + new_cons

def deduce_bk_props_clingo(settings, solver):
    """Deduce properties (props) by grounding them in clingo."""
    prog = []
    arities = set()
    for p, a in settings.body_preds:
        arities.add(a)
        prog.append(f'holds({p},{format_tuple([f"V{i}" for i in range(a)])}):- {p}{format_args([f"V{i}" for i in range(a)])}.')
        prog.append(f'body_pred({p},{a}).')

    if settings.head_types:
        prog.append(f'type({settings.head_literal[0]},{format_tuple(settings.head_types)}).')
    else:
        types = (f't',) if len(settings.head_literal.arguments) == 1 else tuple(['t'] * len(settings.head_literal.arguments))
        prog.append(f'type({settings.head_literal[0]},{format_tuple(types)}).')

    if settings.body_types:
        for pred, types in settings.body_types.items():
            prog.append(f'type({pred},{format_tuple(types)}).')
    else:
        for p, a in settings.body_preds:
            types = (f't',) if a == 1 else tuple(['t'] * a)
            prog.append(f'type({p},{format_tuple(types)}).')

    new_props, new_cons = build_props(settings, arities)
    solver.add('props', [], '\n'.join(prog + [TIDY_OUTPUT] + new_props + ['#show prop/2.']))
    solver.ground([('props', [])])
    
    props = []
    with solver.solve(yield_=True) as handle:
        for m in handle:
            for atom in m.symbols(shown=True):
                if atom.name == 'prop':
                    props.append(str(atom) + '.')
    return props + new_cons

def get_bk_cons(settings, tester):
    """Main function to gather all BK-derived constraints."""
    with open(settings.bk_file) as f:
        settings.bk_string = f.read()
    with gc_paused():
        facts = read_bk_facts(settings.bk_string, settings.body_preds)

    logger.info(f'Finding pointless rules')
    if facts is None:
        pointless = tester.find_pointless_relations()
    else:
        pointless = pointless_from_facts(settings, facts)
    settings.pointless = pointless
    for p, a in pointless:
        logger.info(f'Pointless relation: {p}/{a}')
        settings.body_preds.remove((p, a))

    # without types, a facts-only BK needs no grounding: the singleton and type constraints need types
    if facts is not None and not settings.body_types:
        settings.datalog = True
        logger.info(f'Loading recalls')
        with stats.duration('recalls'), gc_paused():
            bkcons = recalls_from_facts(settings, facts)
        logger.info(f'Loading BK cons')
        with stats.duration('bkcons'), gc_paused():
            bkcons += deduce_bk_props(settings, None, facts)
        return bkcons

    base_solver = clingo.Control(['-Wnone'], logger=lambda code, msg: None)
    try:
        base_solver.add('base', [], settings.bk_string)
        base_solver.ground([('base', [])])
    except Exception:
        logger.info('Loading recalls FAILURE')
        settings.datalog = False
        return []

    settings.datalog = True
    bkcons = []

    logger.info(f'Loading recalls')
    with stats.duration('recalls'):
        recalls_cons = deduce_recalls(settings, base_solver)
    for x in recalls_cons:
        logger.debug(f'recall: {x}')
    bkcons.extend(recalls_cons)

    singletons = deduce_non_singletons(settings, base_solver)
    for x in singletons:
        logger.debug(f'singletons {x}')
    bkcons.extend(singletons)

    type_cons = deduce_type_cons(settings, base_solver)
    for x in type_cons:
        logger.debug(f'type_con {x}')
    bkcons.extend(type_cons)

    logger.info(f'Loading BK cons')
    with stats.duration('bkcons'), gc_paused():
        results = deduce_bk_props(settings, base_solver, facts)
        for x in results:
            if x.startswith('prop'): logger.debug(f'BKCON {x}')
        bkcons.extend(results)

    return bkcons

def deduce_neg_example_recalls(settings, atoms):
    # Jan Struyf, Hendrik Blockeel: Query Optimization in Inductive Logic Programming by Reordering Literals. ILP 2003: 329-346
    arity = len(settings.head_literal.arguments)
    binary_strings = generate_binary_strings(arity)
    counts = {var_subset: defaultdict(set) for var_subset in binary_strings}

    for var_subset in binary_strings:
        d1 = counts[var_subset]
        for args in atoms:
            key = []
            value = []
            for i in range(arity):
                if var_subset[i]:
                    key.append(args[i])
                else:
                    value.append(args[i])
            key = tuple(key)
            value = tuple(value)
            d1[key].add(value)

    pred = 'neg_fact'
    all_recalls = {(pred, (0,)*arity): len(atoms)}
    for args, d2 in counts.items():
        all_recalls[(pred, args)] = max(len(xs) for xs in d2.values())

    settings.recalls.update(all_recalls)
