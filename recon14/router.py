"""Routers: the L2 factorized router under study plus its controls.

Reconstruction category: RECOVERED (specification).

The pilot uses a factorized router parameterized by **L2 distance**:

    g = sigmoid( ( b - || e_op - e_role ||^2 ) / tau ),  tau = 1

where ``e_op`` is the parent operator's embedding and ``e_role`` is the argument
role's embedding. The router is *structure-only*: it sees public template
information (operator, port/canonicality) and never sees the true edge set or any
runtime value.

Controls follow the historical experiment series:

- ``ConditionalRouter``: a 15-parameter ``(parent_op, edge_role)`` lookup table,
  ``sigmoid(logit[parent_op, edge_role])``. It reached 1.0000 accuracy on the
  5-op grammar in the historical series, so it is the *capability ceiling*
  control: if the L2 router cannot match it, the failure is architectural, not
  informational.
- ``StaticMaskRouter``: one non-conditional parameter per physical admissible
  edge; the geometric-rule control that motivated the arity decoys.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .autodiff import Value
from .boolean_dag import Edge, Node, Program, role_for
from .grammar import ARITY1_OPS, INTERNAL_OPS, Op, PILOT_OPS, PRETRAIN_OPS
from .superset import SupersetTemplate, role_of_edge

ROLES: Tuple[str, ...] = ("arg1", "arg2", "noncanonical")


def _no_backward() -> None:
    """Retained for backwards compatibility; ``requires_grad=False`` is the
    mechanism that actually stops gradient flow."""
    return None


class FactorizedRouterL2:
    """L2-distance factorized router (the pilot's study router).

    ``dim`` is the embedding dimension (frozen at 2 by the pilot protocol).
    ``init_scale`` is the standard deviation of the Gaussian initialisation
    (frozen at 0.3 by the pilot protocol). ``tau`` is the gate temperature,
    frozen at 1.
    """

    def __init__(self, ops: Sequence[Op], dim: int = 2, init_scale: float = 0.3,
                 tau: float = 1.0, seed: int = 0, roles: Sequence[str] = ROLES,
                 bias_init: float = 0.0) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be > 0")
        if tau <= 0.0:
            raise ValueError("tau must be > 0")
        self.ops: Tuple[Op, ...] = tuple(ops)
        self.roles: Tuple[str, ...] = tuple(roles)
        self.dim = int(dim)
        self.init_scale = float(init_scale)
        self.tau = float(tau)
        # Ops whose embeddings are frozen (detached). Excluded from
        # ``parameters()`` so Adam never updates them: the persistence mechanism
        # under study requires that a frozen geometry keeps its values.
        self._frozen_ops: set = set()
        rng = random.Random(seed)
        # One embedding per operator plus one per role.
        self.op_emb: Dict[Op, List[Value]] = {
            op: [Value(rng.gauss(0.0, init_scale)) for _ in range(self.dim)]
            for op in self.ops
        }
        self.role_emb: Dict[str, List[Value]] = {
            r: [Value(rng.gauss(0.0, init_scale)) for _ in range(self.dim)]
            for r in self.roles
        }
        # The frozen protocol writes the gate as sigmoid((b - ||e_op - e_role||^2)
        # / tau). `b` is a *learned* free parameter of the same kind as the
        # embeddings, not a fixed constant: with b pinned at 0 the gate is
        # sigmoid(-d^2) <= 0.5 everywhere, so a matching op/role pair can never
        # exceed 0.5 and the frozen convergence criterion g(NOT2, arg1) > 0.9
        # would be unreachable.
        self.bias: Value = Value(bias_init)

    # ---- gating ---------------------------------------------------------
    def embedding_for(self, op: Op) -> Optional[List[Value]]:
        """The op's embedding, or ``None`` if it has never been introduced.

        A NOT2 episode can appear in the shared dataset while the router is still
        in its 5-op pretraining phase (the pilot holds NOT2 out until its
        measured phase). The router is structure-only, so the honest answer for
        an operator it has no embedding for is a deterministic fallback, not a
        crash: ``gate`` returns ``None`` and the caller skips the edge.
        """
        return self.op_emb.get(op)

    def gate(self, parent_op: Op, role: str) -> Optional[Value]:
        if parent_op not in self.op_emb:
            return None
        if role not in self.role_emb:
            raise KeyError(f"no embedding for role {role!r}")
        e_op = self.op_emb[parent_op]
        e_role = self.role_emb[role]
        sq = sum((a - b) * (a - b) for a, b in zip(e_op, e_role))
        return (self.bias - sq) / Value(self.tau)

    def gate_value(self, parent_op: Op, role: str) -> Optional[Value]:
        """Sigmoid-squashed gate in [0, 1], or ``None`` if the op is unknown."""
        g = self.gate(parent_op, role)
        return None if g is None else g.sigmoid()

    def gates_for(self, program: Program) -> Dict[Tuple[int, int, int], Value]:
        """Gate every candidate edge of a program using public information only.

        An edge whose parent operator has no embedding (NOT2 during the 5-op
        pretraining phase) is skipped: the router has no geometry for it, so it
        does not produce a gate. The edge is simply absent from the map rather
        than defaulting to 0 or 1, so no fabricated signal reaches the executor.
        """
        out: Dict[Tuple[int, int, int], Value] = {}
        for edge in program.candidate_edges:
            parent = program.nodes[edge.dst]
            child = program.nodes[edge.src]
            is_canon = bool(program.template and program.template.is_canonical(
                (parent.depth, parent.slot), (child.depth, child.slot)))
            role = role_for(parent.op, edge.port, is_canon)
            g = self.gate_value(parent.op, role)
            if g is None:
                continue
            out[edge.key()] = g
        return out

    # ---- parameters ------------------------------------------------------
    def parameters(self) -> List[Value]:
        out: List[Value] = []
        for op, emb in self.op_emb.items():
            if op in self._frozen_ops:
                continue
            out.extend(emb)
        for emb in self.role_emb.values():
            out.extend(emb)
        out.append(self.bias)
        return out

    def trainable_ops(self) -> Tuple[Op, ...]:
        return self.ops

    def freeze_ops(self, ops: Iterable[Op]) -> List[Value]:
        """Freeze the given operators' embeddings and return them as constants.

        Frozen parameters are removed from ``op_emb`` and stored as constants so
        that no gradient can flow to them; their values are preserved exactly.
        This is the persistence mechanism under study.
        """
        frozen: List[Value] = []
        for op in ops:
            if op not in self.op_emb:
                continue
            self._frozen_ops.add(op)
            for v in self.op_emb[op]:
                # Detach and mark non-trainable. Both are needed: ``_prev=()``
                # cuts the reverse-mode *traversal* to this node, and
                # ``requires_grad=False`` stops the parent op's closure from
                # accumulating gradient into it (the sweep reaches a leaf via
                # its parent, not via its own ``_prev``).
                v.data = float(v.data)
                v.requires_grad = False
                v._backward = _no_backward
                v._prev = ()
                frozen.append(v)
            # Replace the live embeddings with detached constants carrying the
            # same values, so the geometry is reused but cannot be trained.
            self.op_emb[op] = [Value(v.data) for v in self.op_emb[op]]
        return frozen

    def add_op(self, op: Op, seed: int) -> None:
        """Introduce a new operator with a fresh embedding (Condition B)."""
        if op in self.op_emb:
            raise ValueError(f"operator {op!r} already present")
        rng = random.Random(seed)
        self.op_emb[op] = [Value(rng.gauss(0.0, self.init_scale))
                           for _ in range(self.dim)]
        if op not in self.ops:
            self.ops = tuple(list(self.ops) + [op])

    def gate_distance(self, op: Op, role: str) -> float:
        e_op = self.op_emb[op]
        e_role = self.role_emb[role]
        return math.sqrt(sum((a.data - b.data) ** 2 for a, b in zip(e_op, e_role)))

    def state(self) -> Dict[str, object]:
        return {
            "ops": [op.value for op in self.ops],
            "dim": self.dim,
            "init_scale": self.init_scale,
            "tau": self.tau,
            "op_emb": {op.value: [v.data for v in emb]
                       for op, emb in self.op_emb.items()},
            "role_emb": {r: [v.data for v in emb] for r, emb in self.role_emb.items()},
            "bias": self.bias.data,
        }


class ConditionalRouter:
    """15-parameter ``(parent_op, edge_role)`` lookup-table router (control).

    ``sigmoid(logit[parent_op, role])``. Historical series: 1.0000 accuracy on
    the 5-op grammar, so this is the capability ceiling control. 5 ops x 3 roles
    = 15 entries.
    """

    def __init__(self, ops: Sequence[Op], roles: Sequence[str] = ROLES,
                 seed: int = 0, init: float = 0.0) -> None:
        self.ops: Tuple[Op, ...] = tuple(ops)
        self.roles: Tuple[str, ...] = tuple(roles)
        if len(self.ops) * len(self.roles) != 15:
            raise ValueError(
                f"ConditionalRouter requires 5 ops x 3 roles = 15 entries, got "
                f"{len(self.ops)} x {len(self.roles)} = "
                f"{len(self.ops) * len(self.roles)}"
            )
        rng = random.Random(seed)
        self.logit: Dict[Tuple[Op, str], Value] = {
            (op, r): Value(rng.gauss(init, 1e-3))
            for op in self.ops
            for r in self.roles
        }

    def gate_value(self, parent_op: Op, role: str) -> Value:
        return self.logit[(parent_op, role)].sigmoid()

    def gates_for(self, program: Program) -> Dict[Tuple[int, int, int], Value]:
        out: Dict[Tuple[int, int, int], Value] = {}
        for edge in program.candidate_edges:
            parent = program.nodes[edge.dst]
            child = program.nodes[edge.src]
            is_canon = bool(program.template and program.template.is_canonical(
                (parent.depth, parent.slot), (child.depth, child.slot)))
            role = role_for(parent.op, edge.port, is_canon)
            out[edge.key()] = self.gate_value(parent.op, role)
        return out

    def parameters(self) -> List[Value]:
        return list(self.logit.values())

    def state(self) -> Dict[str, object]:
        return {"logit": {f"{op.value}|{r}": v.data for (op, r), v in self.logit.items()}}


class StaticMaskRouter:
    """One non-conditional parameter per physical admissible edge (control).

    This is the geometric-rule control that motivated the arity decoys: without
    them it fits "gate ON iff the child is a geometric canonical child" and
    scores 97.6%. It is included so the benchmark can be verified to resist the
    geometric shortcut.
    """

    def __init__(self, template: SupersetTemplate, seed: int = 0,
                 init: float = 0.0) -> None:
        self.template = template
        rng = random.Random(seed)
        self.params: Dict[Tuple[int, int], Value] = {
            (parent_level, parent_index): Value(rng.gauss(init, 1e-3))
            for (parent_level, parent_index) in self.template.all_slots()
        }

    def gate_value(self, parent: Node) -> Value:
        key = (parent.depth, parent.slot)
        if key not in self.params:
            self.params[key] = Value(0.0)
        return self.params[key].sigmoid()

    def gates_for(self, program: Program) -> Dict[Tuple[int, int, int], Value]:
        out: Dict[Tuple[int, int, int], Value] = {}
        for edge in program.candidate_edges:
            parent = program.nodes[edge.dst]
            out[edge.key()] = self.gate_value(parent)
        return out

    def parameters(self) -> List[Value]:
        return list(self.params.values())


def parameter_count(router) -> int:
    return len(router.parameters())


__all__ = [
    "ROLES",
    "FactorizedRouterL2",
    "ConditionalRouter",
    "StaticMaskRouter",
    "parameter_count",
]
