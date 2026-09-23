"""Structural oracle, exact evaluation, and baseline gating.

Reconstruction category: CANONICAL SPECIFICATION.

The oracle provides structural validation (acyclicity, arity, admissibility),
exact Boolean reference evaluation over the exhaustive truth table, edge-necessity
proofs, decoy-relevance proofs, and the falsification baselines:

- ``oracle_gates``: the ground-truth gating (upper bound).
- ``copy_mask_gates``: gate ON for every occupied admissible edge. Without
  window decoys this scores 100%, which is why decoys exist.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .boolean_dag import Edge, Node, Program, evaluate_program, input_name_of
from .executor import CASMExecutor
from .grammar import ARITY, Op, boolean_eval
from .superset import Slot, SupersetTemplate

EdgeKey = Tuple[int, int, int]


# ---- structural validation --------------------------------------------------


def validate_program(program: Program) -> List[str]:
    """Return a list of structural violations. Empty means valid."""
    errors: List[str] = []
    if not program.nodes:
        return ["empty program"]
    if program.template is None:
        return ["program has no template"]

    # 1. Acyclicity (a tree over a fixed template is acyclic by construction,
    #    but the check is cheap and must never be skipped).
    if not is_acyclic(program):
        errors.append("program contains a cycle")

    # 2. Arity: each operator must have exactly its declared number of TRUE args.
    #    A decoy boundary edge occupies a port physically but is not ground
    #    truth, so it is excluded here -- otherwise a window decoy on port 0
    #    would be reported as an arity violation on a correct program.
    for n in program.nodes:
        if not n.in_program:
            continue
        if n.op == Op.INPUT:
            if n.args:
                errors.append(f"INPUT node {n.index} has {len(n.args)} args")
            continue
        true_args = [e for e in n.args if not e.is_decoy]
        if len(true_args) != ARITY[n.op]:
            errors.append(
                f"node {n.index} op={n.op.value} has {len(true_args)} true args, "
                f"expected {ARITY[n.op]}"
            )

    # 3. Admissibility: every true edge must connect an admissible child slot.
    for n in program.nodes:
        if not n.in_program or n.is_input:
            continue
        for e in n.args:
            if e.is_decoy:
                continue
            child = program.nodes[e.src]
            parent_slot = (n.depth, n.slot)
            child_slot = (child.depth, child.slot)
            if not program.template.is_admissible(parent_slot, child_slot):
                errors.append(
                    f"edge {e.key()} is inadmissible: {child_slot} is not an "
                    f"admissible child of {parent_slot}"
                )

    # 4. Port uniqueness over true edges: a parent cannot receive two true
    #    children on one port. A decoy boundary edge deliberately collides with
    #    a true child's port; that collision is the ambiguity under test.
    for n in program.nodes:
        if not n.in_program:
            continue
        ports = [e.port for e in n.args if not e.is_decoy]
        if len(ports) != len(set(ports)):
            errors.append(f"node {n.index} has duplicate ports {ports}")

    # 5. Exactly one root, and the root is at level 0 slot 0.
    roots = [n.index for n in program.nodes if n.in_program and n.depth == 0]
    if len(roots) != 1 or roots[0] != 0:
        errors.append(f"expected exactly one root at (0,0), got {roots}")
    return errors


def is_acyclic(program: Program) -> bool:
    """Topological-sort reachability check over the true edges."""
    indeg: Dict[int, int] = {}
    adj: Dict[int, List[int]] = {}
    for n in program.nodes:
        if not n.in_program:
            continue
        indeg.setdefault(n.index, 0)
        adj.setdefault(n.index, [])
    for n in program.nodes:
        if not n.in_program:
            continue
        for e in n.args:
            if e.is_decoy:
                continue
            adj[e.src].append(n.index)
            indeg[n.index] = indeg.get(n.index, 0) + 1
    queue = [i for i, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        i = queue.pop()
        seen += 1
        for j in adj.get(i, []):
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)
    return seen == len(indeg)


def topological_order(program: Program) -> List[int]:
    """Bottom-up execution order: deepest level first, then node index."""
    indexed = [(n.depth, n.index) for n in program.nodes if n.in_program]
    indexed.sort(key=lambda t: (-t[0], t[1]))
    return [i for _, i in indexed]


def edge_necessity(program: Program) -> Dict[EdgeKey, bool]:
    """Edge-necessity proof: is each true edge required for the root output?

    An edge is necessary if, holding all other edges fixed, negating the input
    that feeds it changes the root output under some input assignment.
    """
    out: Dict[EdgeKey, bool] = {}
    for n in program.nodes:
        if not n.in_program or n.is_input:
            continue
        for e in n.args:
            if e.is_decoy:
                continue
            out[e.key()] = False
    for assignment in truth_table_assignments(program):
        base = evaluate_program(program, assignment)
        for key in list(out):
            if out[key]:
                continue
            edge = edge_by_key(program, key)
            flipped = flip_input_for_edge(program, edge, assignment)
            try:
                alt = evaluate_program(program, flipped)
            except KeyError:
                continue
            if alt != base:
                out[key] = True
    return out


def edge_by_key(program: Program, key: EdgeKey) -> Edge:
    for n in program.nodes:
        if not n.in_program:
            continue
        for e in n.args:
            if e.is_decoy:
                continue
            if e.key() == key:
                return e
    raise KeyError(f"no true edge with key {key}")


def flip_input_for_edge(program: Program, edge: Edge,
                        assignment: Dict[str, bool]) -> Dict[str, bool]:
    """Copy of ``assignment`` with the input feeding ``edge`` negated.

    Used by edge-necessity only. Evaluation of the flipped program uses the exact
    reference evaluator, so the oracle never enters the router.
    """
    out = dict(assignment)
    node = program.nodes[edge.src]
    while not node.is_input:
        # Follow the true argument chain; a decoy boundary edge is not ground
        # truth and does not name a program input.
        node = program.nodes[next(e.src for e in node.args if not e.is_decoy)]
    name = input_name_of(node)
    if name in out:
        out[name] = not out[name]
    return out


# ---- exact evaluation --------------------------------------------------------


def program_inputs(program: Program) -> List[str]:
    return list(program.input_names)


def truth_table_assignments(program: Program) -> Iterable[Dict[str, bool]]:
    """Exhaustive Boolean truth table over the program's distinct inputs.

    Constrained to <= 3 distinct active inputs (<= 8 rows) by the feasibility of
    exhaustive evaluation, matching the recovered benchmark constraint C1.
    """
    names = program_inputs(program)
    if len(names) > 3:
        raise ValueError(
            f"exhaustive truth table requires <= 3 inputs, got {len(names)}"
        )
    for combo in itertools.product([False, True], repeat=len(names)):
        yield dict(zip(names, combo))


def exhaustive_reference_outputs(program: Program) -> List[bool]:
    """Exact root output for every row of the truth table."""
    return [evaluate_program(program, a) for a in truth_table_assignments(program)]


def exact_accuracy(program: Program, executor: CASMExecutor,
                   gates: Dict[EdgeKey, float]) -> float:
    """Exact accuracy: the hard execution must match the reference truth table.

    A row is correct iff the hard-executed root value rounds to the reference
    bit. Collisions and decoys are observable and unrepaired, so a wrong gate
    configuration shows up as a mismatch rather than being masked out.
    """
    ref = exhaustive_reference_outputs(program)
    if not ref:
        return 0.0
    hits = 0
    total = 0
    for assignment, ref_bit in zip(truth_table_assignments(program), ref):
        inputs = {name: (1.0 if val else 0.0) for name, val in assignment.items()}
        # Decoy leaves are not program inputs (they are not ground truth), but
        # the executor needs a value for them. Their value is fixed and public,
        # set by the generator, so it is supplied deterministically here.
        for n in program.nodes:
            if (n.is_input and not n.in_program
                    and n.input_value is not None):
                inputs[input_name_of(n)] = (1.0 if n.input_value else 0.0)
        got = executor.execute_hard(program, inputs, gates)
        if (got >= 0.5) == ref_bit:
            hits += 1
        total += 1
    return hits / total


# ---- baselines ---------------------------------------------------------------


def oracle_gates(program: Program) -> Dict[EdgeKey, float]:
    """Ground-truth gating: 1 on true edges, 0 on decoys.

    A decoy boundary edge is physically wired into a parent's args, so it is in
    ``candidate_edges``; the ground truth says it is not part of the program.
    """
    true = program.true_edge_set
    return {e.key(): (0.0 if e.is_decoy else
                      (1.0 if e.key() in true else 0.0))
            for e in program.candidate_edges}


def copy_mask_gates(program: Program) -> Dict[EdgeKey, float]:
    """Gate ON for every occupied admissible edge, regardless of truth.

    Without window decoys this scores 100%, because occupied admissible edges
    always match ground truth. It is retained as the falsification floor.
    """
    return {e.key(): 1.0 for e in program.candidate_edges}


def routing_precision_recall(program: Program,
                             gates: Dict[EdgeKey, float],
                             threshold: float = 0.5) -> Tuple[float, float]:
    """Routing precision/recall at the given threshold.

    Precision is over candidate edges; recall is over true edges. False positives
    inside decoy subtrees are reported separately, because they are firewalled
    from the root by the gated-off boundary edge and do not affect the root
    output.
    """
    true = program.true_edge_set
    predicted = {k for k, v in gates.items() if v >= threshold}
    tp = len(predicted & true)
    fp = len(predicted - true)
    fn = len(true - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall


def decoy_false_positives(program: Program,
                          gates: Dict[EdgeKey, float],
                          threshold: float = 0.5) -> int:
    """Count gated-on edges whose source is inside a decoy subtree."""
    predicted = {k for k, v in gates.items() if v >= threshold}
    return sum(1 for k in predicted if program.nodes[k[0]].decoy_subtree)


__all__ = [
    "validate_program",
    "is_acyclic",
    "topological_order",
    "edge_necessity",
    "program_inputs",
    "truth_table_assignments",
    "exhaustive_reference_outputs",
    "exact_accuracy",
    "oracle_gates",
    "copy_mask_gates",
    "routing_precision_recall",
    "decoy_false_positives",
]
