# code written by Andreas Niskanen (andreas.niskanen@helsinki.fi)
import multiprocessing, subprocess, tempfile

from pysat.formula import WCNF
from pysat.examples.rc2 import RC2Stratified
from pysat.card import *

ANYTIME_MAXSAT_SOLVER = 'NuWLS-c'
ANYTIME_MAXSAT_SOLVER_PARAMS = ""
ANYTIME_MAXSAT_SOLVER_SIGNAL = 15

def new_wcnf_to_file(hard_clauses, soft_clauses, weights, file):
    for clause in hard_clauses:
        #print("h " + " ".join(map(str, clause)) + " 0")
        file.write("h " + " ".join(map(str, clause)) + " 0" + "\n")
    for clause, w in zip(soft_clauses, weights):
        if w == 0:
            continue
        file.write(str(w) + " " + " ".join(map(str, clause)) + " 0" + "\n")
    file.flush()

def exact_maxsat_solve(hard_clauses, soft_clauses, weights, timeout=None):
    """Optimal RC2 solve. With a timeout it runs in a forked child and gives (inf, None) if the child
    does not finish in time."""
    if timeout is None:
        return _exact_maxsat_solve(hard_clauses, soft_clauses, weights)
    ctx = multiprocessing.get_context('fork')
    recv, send = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_exact_maxsat_child, args=(send, hard_clauses, soft_clauses, weights), daemon=True)
    child.start()
    send.close()
    try:
        result = recv.recv() if recv.poll(timeout) else (float("inf"), None)
    except EOFError:
        result = (float("inf"), None)
    finally:
        if child.is_alive():
            child.kill()
        child.join()
        recv.close()
    return result

def _exact_maxsat_child(conn, hard_clauses, soft_clauses, weights):
    conn.send(_exact_maxsat_solve(hard_clauses, soft_clauses, weights))
    conn.close()

def _exact_maxsat_solve(hard_clauses, soft_clauses, weights):
    wcnf = WCNF()
    for clause in hard_clauses:
        wcnf.append(clause)
    for clause, w in zip(soft_clauses, weights):
        if w == 0:
            continue
        wcnf.append(clause, weight=w)

    rc2 = RC2Stratified(wcnf, solver='g3', adapt=True, exhaust=True, blo='div', incr=False, minz=True, trim=0)
    model = rc2.compute()
    if model is not None:
        return rc2.cost, model
    return float("inf"), None

def anytime_maxsat_solve(hard_clauses, soft_clauses, weights, timeout):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".wcnf") as tmp:
        new_wcnf_to_file(hard_clauses, soft_clauses, weights, tmp)
        try:
            # output = subprocess.check_output(["timeout", str(timeout), os.path.join(os.path.dirname(__file__), settings.anytime_maxsat_solver)] + settings.anytime_maxsat_solver_params.split() + [tmp.name]).decode("utf-8").split("\n")
            args = ["timeout", "-s", str(ANYTIME_MAXSAT_SOLVER_SIGNAL), str(timeout), ANYTIME_MAXSAT_SOLVER] + ANYTIME_MAXSAT_SOLVER_PARAMS.split() + [tmp.name]
            output = subprocess.check_output(args).decode("utf-8").split("\n")
        except subprocess.CalledProcessError as error:
            output = error.output.decode("utf-8").split("\n")
    if "s UNSATISFIABLE" in output:
        # print('UNSATISFIABLE')
        return float("inf"), None
    elif "s OPTIMUM FOUND" in output or "s SATISFIABLE" in output:
        cost_line = [line for line in output if line.startswith("o ")][-1]
        cost = int(cost_line.replace("o ", "").replace("-oo", "0"))
        model_line = [line for line in output if line.startswith("v ")][-1]
        model_line = model_line.replace("v ", "")
        model = [i if model_line[i-1] == "1" else -i for i in range(1, len(model_line)+1)]
        return cost, model
    else:
        # print("WARNING: No solution found.")
        return None, None