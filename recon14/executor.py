"""Single-pass soft-logic executor.

Reconstruction category: CANONICAL SPECIFICATION.

FlyVis-style recurrent dynamics rely on weighted sums followed by linear
threshold units, which cannot represent XOR. Phase 1 therefore replaces recurrence
with a **single-pass** execution model over typed Boolean DAGs using op-specific
differentiable soft-logic relaxations of AND, OR, XOR, NOT, and IDENTITY.

Execution flows bottom-up from the deepest level to the root. Each edge ``e`` is
gated by a scalar gate ``g_e in [0, 1]`` supplied by the router, and the
contribution of a child value is ``g_e * value`` (or, for NOT/NOT2/IDENTITY,
transformed as shown below) plus ``(1 - g_e) * neutral`` where ``neutral`` is the
op-specific absorbing value that leaves the output unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .autodiff import Value
from .boolean_dag import Edge, Node, Program, input_name_of
from .grammar import ARITY, Op


class CASMExecutor:
    """Differentiable single-pass executor over Boolean DAGs.

    ``gated_edges`` maps an edge key ``(src, dst, port)`` to a ``Value`` gate in
    [0, 1]. Edges with no entry default to ``0.0`` (gated off). Hard evaluation
    rounds the final root value at threshold 0.5.

    A port that carries a decoy boundary edge alongside the true edge is resolved
    by gate value: the arg with the higher gate wins the port, so a copy mask
    (all gates ON) feeds the decoy subtree into the op and is wrong. Decoy args
    are skipped only when a port has no competing true edge.
    """

    # Op-specific absorbing values used for the "gated-off" contribution. Chosen
    # so that an un-gated argument leaves the operation's output unchanged.
    NEUTRAL: Dict[Op, float] = {
        Op.AND: 1.0,
        Op.OR: 0.0,
        Op.XOR: 0.0,
        Op.NOT: None,      # unary; handled directly
        Op.NOT2: None,     # unary; handled directly
        Op.IDENTITY: None,  # unary; handled directly
    }

    def __init__(self) -> None:
        self.last_node_values: Optional[Dict[int, Value]] = None
        self.last_node_values_hard: Optional[Dict[int, float]] = None

    # ---- execution -----------------------------------------------------
    def execute(self, program: Program, inputs: Dict[str, Value],
                gated_edges: Dict[Tuple[int, int, int], Value],
                soft: bool = True) -> Value:
        """Execute the true program bottom-up and return the root's soft value.

        ``inputs`` maps input names to soft values in [0, 1].
        """
        values: Dict[int, Value] = {}
        order = self._topological_order(program)
        for node_index in order:
            node = program.nodes[node_index]
            if node.is_input:
                name = self._input_name(node)
                if name not in inputs:
                    raise KeyError(f"executor missing input {name}")
                values[node_index] = inputs[name]
                continue
            # A decoy node is not gated: its internal wiring is fixed and public,
            # so its value is computed once and reused. Only the boundary edge
            # into the true program is gated.
            if not node.in_program:
                values[node_index] = self._apply_op(node, values, gated_edges, soft)
                continue
            values[node_index] = self._apply_op(node, values, gated_edges, soft)
        self.last_node_values = values
        if program.root_index not in values:
            raise ValueError("root has no value; program is empty")
        return values[program.root_index]

    def execute_hard(self, program: Program, inputs: Dict[str, float],
                     gate_decisions: Dict[Tuple[int, int, int], float]) -> float:
        """Non-differentiable execution with hard (binary) gate decisions.

        Used for the exact accuracy metric and for the oracle/copy-mask controls.
        """
        values: Dict[int, float] = {}
        order = self._topological_order(program)
        for node_index in order:
            node = program.nodes[node_index]
            if node.is_input:
                name = self._input_name(node)
                values[node_index] = float(inputs[name])
                continue
            # A decoy node is not gated (see ``execute``).
            if not node.in_program:
                values[node_index] = self._apply_op_hard(node, values,
                                                         gate_decisions)
                continue
            values[node_index] = self._apply_op_hard(node, values, gate_decisions)
        self.last_node_values_hard = values
        return values[program.root_index]

    # ---- op semantics --------------------------------------------------
    def _apply_op(self, node: Node, values: Dict[int, Value],
                  gated_edges: Dict[Tuple[int, int, int], Value],
                  soft: bool) -> Value:
        op = node.op
        if op in (Op.NOT, Op.NOT2, Op.IDENTITY):
            return self._unary(node, values, gated_edges, op)
        return self._nary(node, values, gated_edges, op)

    def _unary(self, node: Node, values: Dict[int, Value],
               gated_edges: Dict[Tuple[int, int, int], Value], op: Op) -> Value:
        edge = self._port_arg(node, 0, gated_edges, soft=True)
        g = self._gate(edge, gated_edges)
        x = values[edge.src]
        if op == Op.NOT or op == Op.NOT2:
            # `not x` in [0,1] is `1 - x`. A gated-off argument contributes
            # `g*x`; its negation contributes `1 - g*x`. The identity at g=0 is
            # preserved by anchoring to the neutral input value 1.0 (NOT of the
            # absorbing constant), keeping the relaxation continuous in g.
            carried = g * x
            anchor = 1.0 - g
            return Value(1.0) - (carried + anchor)
        if op == Op.IDENTITY:
            return g * x + (Value(1.0) - g) * x
        raise ValueError(f"unary op not handled: {op}")

    def _nary(self, node: Node, values: Dict[int, Value],
              gated_edges: Dict[Tuple[int, int, int], Value], op: Op) -> Value:
        neutral = self.NEUTRAL[op]
        assert neutral is not None
        terms: List[Value] = []
        for port in range(ARITY[op]):
            edge = self._port_arg(node, port, gated_edges, soft=True)
            g = self._gate(edge, gated_edges)
            x = values[edge.src]
            terms.append(g * x + (Value(1.0) - g) * Value(neutral))
        if op == Op.AND:
            out = terms[0]
            for t in terms[1:]:
                out = out * t
            return out
        if op == Op.OR:
            # In [0,1], OR is `a + b - a*b` (probabilistic sum).
            out = terms[0]
            for t in terms[1:]:
                out = out + t - out * t
            return out
        if op == Op.XOR:
            # XOR is parity. In [0,1] soft parity is `a + b - 2ab`.
            out = terms[0]
            for t in terms[1:]:
                out = out + t - Value(2.0) * out * t
            return out
        raise ValueError(f"n-ary op not handled: {op}")

    def _apply_op_hard(self, node: Node, values: Dict[int, float],
                       gate_decisions: Dict[Tuple[int, int, int], float]) -> float:
        op = node.op
        if op in (Op.NOT, Op.NOT2):
            edge = self._port_arg(node, 0, gate_decisions, soft=False)
            g = float(gate_decisions.get(edge.key(), 0.0))
            x = values[edge.src]
            carried = g * x
            anchor = 1.0 - g
            return 1.0 - (carried + anchor)
        if op == Op.IDENTITY:
            edge = self._port_arg(node, 0, gate_decisions, soft=False)
            g = float(gate_decisions.get(edge.key(), 0.0))
            return g * values[edge.src] + (1.0 - g) * values[edge.src]
        neutral = self.NEUTRAL[op]
        terms = []
        for port in range(ARITY[op]):
            edge = self._port_arg(node, port, gate_decisions, soft=False)
            g = float(gate_decisions.get(edge.key(), 0.0))
            terms.append(g * values[edge.src] + (1.0 - g) * neutral)
        if op == Op.AND:
            out = terms[0]
            for t in terms[1:]:
                out = out * t
            return out
        if op == Op.OR:
            out = terms[0]
            for t in terms[1:]:
                out = out + t - out * t
            return out
        if op == Op.XOR:
            out = terms[0]
            for t in terms[1:]:
                out = out + t - 2.0 * out * t
            return out
        raise ValueError(f"n-ary op not handled: {op}")

    # ---- helpers --------------------------------------------------------
    def _gate(self, edge: Edge, gates) -> Any:
        """Gate value for an edge; absent edges default to 0.0 (gated off).

        Accepts either the differentiable ``Value`` map or the hard float map.
        """
        if edge.key() not in gates:
            return Value(0.0)
        return gates[edge.key()]

    def _gate_scalar(self, edge: Edge, gates) -> float:
        v = self._gate(edge, gates)
        return float(v.data if isinstance(v, Value) else v)

    def _port_arg(self, node: Node, port: int, gates, soft: bool) -> Edge:
        """The edge that feeds ``port`` of ``node`` under the current gates.

        A port may carry a true edge plus a decoy boundary edge (the ambiguity
        the benchmark is built on). Selection is by gate value: the higher gate
        wins, with a decoy winning an exact tie, because a copy mask sets every
        gate to 1 and must lose to the true wiring. If only a decoy occupies the
        port, it is used -- the op still needs a value there.
        """
        on_port = [e for e in node.args if e.port == port]
        true_here = [e for e in on_port if not e.is_decoy]
        if not on_port:
            raise ValueError(
                f"node {node.index} op={node.op.value} has no arg on port {port}"
            )
        # A decoy subtree's own args are never gated; pick the one true arg.
        if not node.in_program:
            return true_here[0] if true_here else on_port[0]
        if not true_here:
            # Only a decoy occupies this port. The port still needs a value, and
            # the decoy is the only candidate.
            return on_port[0]
        if len(true_here) == 1:
            # Exactly one true edge on this port. It is selected unless a decoy
            # on the same port has a gate at least as high -- an exact tie means
            # the mask is not discriminating (the copy mask sets every gate to
            # 1), so the tie goes to the decoy and the copy is wrong. The oracle
            # gives the true edge 1.0 and the decoy 0.0, so it has no tie.
            decoy_here = [e for e in on_port if e.is_decoy]
            if decoy_here:
                best_decoy = max(decoy_here,
                                 key=lambda e: self._gate_scalar(e, gates))
                if self._gate_scalar(best_decoy, gates) >= \
                        self._gate_scalar(true_here[0], gates):
                    return best_decoy
            return true_here[0]
        # Multiple true edges on one port: highest gate decides.
        return max(true_here, key=lambda e: self._gate_scalar(e, gates))

    def _input_name(self, node: Node) -> str:
        return input_name_of(node)

    def _topological_order(self, program: Program) -> List[int]:
        """Bottom-up order: deepest level first, then by node index within level.

        This is the canonical Phase-1 execution order. Decoy nodes are included:
        a decoy subtree is wired into a true parent's args, so its value must be
        available before the parent is evaluated.
        """
        indexed: List[Tuple[int, int]] = [
            (n.depth, n.index) for n in program.nodes
        ]
        indexed.sort(key=lambda t: (-t[0], t[1]))
        return [idx for _, idx in indexed]


__all__ = ["CASMExecutor"]
