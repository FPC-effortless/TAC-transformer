"""Typed Boolean program grammar used by RECON-14.

Reconstruction category: CANONICAL SPECIFICATION + RECOVERED pilot extensions.

Canonical Phase-1 grammar is ``AND, OR, XOR, NOT, IDENTITY``. The pilot adds
``NOT2``: arity 1, behaviourally identical to ``NOT`` (``not args[0]``), and
deliberately excluded from the default internal op set so that its edges are not
pre-populated by construction.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Tuple


class Op(str, Enum):
    INPUT = "INPUT"
    NOT = "NOT"
    IDENTITY = "IDENTITY"
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    # Pilot extension (RECOVERED): arity 1, behaviourally identical to NOT,
    # excluded from INTERNAL_OPS so the router must place its edges from scratch.
    NOT2 = "NOT2"


# Arity by operator. RECOVERED from the historical experiment description.
ARITY: Dict[Op, int] = {
    Op.INPUT: 0,
    Op.NOT: 1,
    Op.IDENTITY: 1,
    Op.NOT2: 1,
    Op.AND: 2,
    Op.OR: 2,
    Op.XOR: 2,
}

# Operators the benchmark generator uses to build internal program nodes.
# NOT2 is intentionally excluded (RECOVERED).
INTERNAL_OPS: Tuple[Op, ...] = (Op.AND, Op.OR, Op.XOR, Op.NOT, Op.IDENTITY)

# The five operators the pilot pretrains on before introducing NOT2.
PRETRAIN_OPS: Tuple[Op, ...] = (Op.AND, Op.OR, Op.XOR, Op.NOT, Op.IDENTITY)

# The full pilot operator set, including the held-in new operator.
PILOT_OPS: Tuple[Op, ...] = (Op.AND, Op.OR, Op.XOR, Op.NOT, Op.IDENTITY, Op.NOT2)

# Commutative operators: their operand order is irrelevant to program semantics.
COMMUTATIVE_OPS: FrozenSet[Op] = frozenset({Op.AND, Op.OR, Op.XOR})

# Operators of arity 1, which create the arity-decoy ambiguity: an unused
# canonical slot is a valid edge for an arity-2 parent but a decoy for an
# arity-1 parent.
ARITY1_OPS: FrozenSet[Op] = frozenset({Op.NOT, Op.IDENTITY, Op.NOT2})


def arity_of(op: Op) -> int:
    return ARITY[op]


def is_internal(op: Op) -> bool:
    return op in INTERNAL_OPS


def is_pilot_op(op: Op) -> bool:
    return op in PILOT_OPS


def commutative(op: Op) -> bool:
    return op in COMMUTATIVE_OPS


def boolean_eval(op: Op, args: Tuple[bool, ...]) -> bool:
    """Exact Boolean evaluation. Reference for the oracle and hard evaluation."""
    a = args
    if op == Op.INPUT:
        if len(a) != 0:
            raise ValueError(f"INPUT takes no arguments, got {len(a)}")
        raise ValueError("INPUT nodes carry their own value; boolean_eval is for operators")
    if op == Op.NOT or op == Op.NOT2:
        if len(a) != 1:
            raise ValueError(f"{op} requires 1 argument, got {len(a)}")
        return not a[0]
    if op == Op.IDENTITY:
        if len(a) != 1:
            raise ValueError(f"IDENTITY requires 1 argument, got {len(a)}")
        return a[0]
    if op == Op.AND:
        if len(a) != 2:
            raise ValueError(f"AND requires 2 arguments, got {len(a)}")
        return a[0] and a[1]
    if op == Op.OR:
        if len(a) != 2:
            raise ValueError(f"OR requires 2 arguments, got {len(a)}")
        return a[0] or a[1]
    if op == Op.XOR:
        if len(a) != 2:
            raise ValueError(f"XOR requires 2 arguments, got {len(a)}")
        return a[0] != a[1]
    raise ValueError(f"unknown operator {op!r}")


def op_name(op: Op) -> str:
    return op.value


__all__ = [
    "Op",
    "ARITY",
    "INTERNAL_OPS",
    "PRETRAIN_OPS",
    "PILOT_OPS",
    "COMMUTATIVE_OPS",
    "ARITY1_OPS",
    "arity_of",
    "is_internal",
    "is_pilot_op",
    "commutative",
    "boolean_eval",
    "op_name",
]
