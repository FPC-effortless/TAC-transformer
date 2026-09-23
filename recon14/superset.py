"""Fixed admissible structural graph (the CASM "connectome").

Reconstruction category: CANONICAL SPECIFICATION.

Slots are indexed by ``(level, index)`` where level 0 is the root. Every occupied
parent has exactly 2 canonical branching positions. Admissible children are the
canonical children **plus** same-level neighbouring positions within ``window``
slots, which is what creates the window-decoy ambiguity.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

Slot = Tuple[int, int]


class SupersetTemplate:
    """Fixed binary tree over which candidate edges are drawn.

    ``max_depth`` bounds the template span (canonical benchmark: 8). ``window``
    extends admissibility to same-level neighbours (canonical benchmark: 1).
    """

    def __init__(self, max_depth: int = 8, window: int = 1) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if window < 0:
            raise ValueError("window must be >= 0")
        self.max_depth = int(max_depth)
        self.window = int(window)

    # ---- slot bookkeeping -------------------------------------------
    def canonical_children(self, level: int, index: int) -> List[Slot]:
        """The two canonical branching slots of an occupied parent."""
        if level >= self.max_depth - 1:
            return []
        return [(level + 1, 2 * index), (level + 1, 2 * index + 1)]

    def admissible_children(self, level: int, index: int) -> List[Slot]:
        """Canonical children plus same-level neighbours within ``window``."""
        out: List[Slot] = list(self.canonical_children(level, index))
        if self.window == 0:
            return out
        child_level = level + 1
        if child_level >= self.max_depth:
            return out
        lo = max(0, 2 * index - self.window)
        hi = min(self.width(child_level) - 1, 2 * index + 1 + self.window)
        for i in range(lo, hi + 1):
            s = (child_level, i)
            if s not in out:
                out.append(s)
        return out

    def width(self, level: int) -> int:
        if level < 0 or level >= self.max_depth:
            return 0
        return 1 << level

    def all_slots(self) -> List[Slot]:
        out: List[Slot] = []
        for lvl in range(self.max_depth):
            out.extend((lvl, i) for i in range(self.width(lvl)))
        return out

    def is_canonical(self, parent: Slot, child: Slot) -> bool:
        return child in set(self.canonical_children(*parent))

    def is_admissible(self, parent: Slot, child: Slot) -> bool:
        return child in set(self.admissible_children(*parent))

    def slot_index(self, slot: Slot) -> int:
        """Flatten ``(level, index)`` to a unique linear id."""
        level, index = slot
        if level < 0 or level >= self.max_depth:
            raise ValueError(f"level out of range: {level}")
        if index < 0 or index >= self.width(level):
            raise ValueError(f"index out of range at level {level}: {index}")
        return (1 << level) - 1 + index

    def num_slots(self) -> int:
        return (1 << self.max_depth) - 1

    def depth_of(self, slot: Slot) -> int:
        return slot[0]

    def max_level(self) -> int:
        return self.max_depth - 1

    # ---- equality / repr ---------------------------------------------
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SupersetTemplate)
            and self.max_depth == other.max_depth
            and self.window == other.window
        )

    def __hash__(self) -> int:
        return hash((self.max_depth, self.window))

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"SupersetTemplate(max_depth={self.max_depth}, window={self.window})"


def role_of_edge(parent_op_is_arity1: bool, port: int, is_canonical: bool) -> str:
    """Router role label for a candidate edge.

    A window-decoy edge takes the ``noncanonical`` role regardless of port.
    Canonical edges take their port role. This gives the router only public
    template information and yields the three role columns the L2 geometry
    operates over.
    """
    if not is_canonical:
        return "noncanonical"
    return "arg1" if port == 0 else "arg2"


def admissible_edges_for(parent: Slot, parent_arity: int, template: SupersetTemplate,
                         occupied: Set[Slot]) -> List[Slot]:
    """Candidate child slots for a parent given the current occupancy.

    Occupied slots are excluded: each slot receives at most one parent in this
    benchmark, so an occupied slot is not a live candidate for wiring but may
    still host a decoy subtree root.
    """
    adm = [s for s in template.admissible_children(*parent) if s not in occupied]
    return adm


__all__ = [
    "Slot",
    "SupersetTemplate",
    "role_of_edge",
    "admissible_edges_for",
]
