"""Typed Boolean DAG data model and generator.

Reconstruction category: CANONICAL SPECIFICATION (structure + decoys) +
RECOVERED (pilot parameters).

The generator samples a random Boolean DAG over the fixed ``SupersetTemplate``,
attaches window decoys at non-canonical admissible slots and arity decoys at
unused canonical slots of arity-1 operators, then exposes the union of true edges
and decoy edges as the router's candidate set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .grammar import ARITY, ARITY1_OPS, INTERNAL_OPS, Op, PILOT_OPS, boolean_eval
from .superset import Slot, SupersetTemplate, role_of_edge


@dataclass(frozen=True)
class Edge:
    """A candidate edge from ``src`` to ``dst`` occupying argument port ``port``."""

    src: int  # child node index
    dst: int  # parent node index
    port: int  # argument slot on the parent (0 or 1)
    # False for the ground-truth wiring. True only for a decoy boundary edge that
    # is physically wired into the parent's args (competing with the true child
    # on the same port) but must not count as ground truth. This is what makes a
    # copy mask score below 100%.
    is_decoy: bool = False

    def key(self) -> Tuple[int, int, int]:
        return (self.src, self.dst, self.port)


@dataclass
class Node:
    """One node of the program DAG."""

    index: int
    op: Op
    depth: int
    slot: int
    arity: int = field(default=0)
    # True argument edges, ordered by port.
    args: List[Edge] = field(default_factory=list)
    # Value for INPUT leaves. INPUT leaves may share an identity: the benchmark
    # bounds the number of *distinct* active inputs (constraint C1) so the
    # exhaustive truth table stays at <= 8 rows, but two leaves can name the
    # same variable.
    input_value: Optional[bool] = None
    input_id: Optional[int] = None
    # Disjoint name for a decoy leaf. A decoy leaf is not a program input and must
    # never be confused with one: without this the executor maps a decoy leaf and
    # a true input to the same name and the two become interchangeable.
    decoy_name: Optional[str] = None
    # Whether this node is part of the true program (vs a decoy subtree).
    in_program: bool = True
    # Whether this node is the root of a decoy subtree.
    is_decoy_root: bool = False
    # Marked when this node's subtree is only reachable via a gated-off boundary
    # edge, so false positives inside it are firewalled from the root.
    decoy_subtree: bool = False

    @property
    def is_input(self) -> bool:
        return self.op == Op.INPUT

    def signature(self) -> Tuple[str, int, int]:
        return (self.op.value, self.depth, self.slot)


@dataclass
class Program:
    """A complete episode: a true program plus its decoy candidate edges."""

    nodes: List[Node] = field(default_factory=list)
    template: Optional[SupersetTemplate] = None
    # All candidate edges offered to the router (true + decoys).
    candidate_edges: List[Edge] = field(default_factory=list)
    # The subset of candidate edges that are the ground-truth wiring.
    true_edges: List[Edge] = field(default_factory=list)
    root_index: int = 0
    active_count: int = 0
    # Distinct named inputs of the true program, in order of first appearance.
    input_names: List[str] = field(default_factory=list)

    @property
    def true_edge_set(self) -> Set[Tuple[int, int, int]]:
        return {e.key() for e in self.true_edges}

    @property
    def candidate_edge_set(self) -> Set[Tuple[int, int, int]]:
        return {e.key() for e in self.candidate_edges}

    def node_by_slot(self, level: int, index: int) -> Optional[Node]:
        for n in self.nodes:
            if n.depth == level and n.slot == index and n.in_program:
                return n
        return None


@dataclass
class GeneratorConfig:
    """Frozen generator configuration (see RECON_14.md protocol table)."""

    max_depth: int = 5
    min_depth: int = 1
    decoy_prob: float = 0.7
    max_decoys_per_episode: int = 3
    ops: Tuple[Op, ...] = PILOT_OPS
    template_depth: int = 8
    window: int = 1
    input_count: int = 3
    seed: int = 20260922

    def validate(self) -> None:
        if self.min_depth < 1:
            raise ValueError("min_depth must be >= 1")
        if self.max_depth < self.min_depth:
            raise ValueError("max_depth must be >= min_depth")
        if not (0.0 <= self.decoy_prob <= 1.0):
            raise ValueError("decoy_prob must be in [0, 1]")
        if self.max_decoys_per_episode < 0:
            raise ValueError("max_decoys_per_episode must be >= 0")
        if self.input_count < 1 or self.input_count > 8:
            raise ValueError("input_count must be in [1, 8] (truth-table feasibility)")


class BooleanDAGGenerator:
    """Samples random typed Boolean DAGs with window and arity decoys."""

    def __init__(self, config: Optional[GeneratorConfig] = None) -> None:
        self.config = (config or GeneratorConfig())
        self.config.validate()
        self.template = SupersetTemplate(self.config.template_depth, self.config.window)

    # ---- construction ------------------------------------------------
    def generate(self, rng: random.Random) -> Program:
        cfg = self.config
        depth = rng.randint(cfg.min_depth, cfg.max_depth)
        program = Program(template=self.template)
        occupied: Dict[Slot, int] = {}
        # Constraint C1: the exhaustive Boolean truth table required by the
        # recovered executor/oracle bounds the number of *distinct* active
        # inputs, not the number of leaves. A depth-5 binary tree has up to 16
        # leaves, so leaves must share input identities. Each leaf draws an
        # identity from a pool of `input_count` variables; the pool is fixed
        # before generation so the table has at most 2**input_count rows.
        max_inputs = max(1, min(cfg.input_count, 3))
        program.input_pool = [f"x{i}" for i in range(max_inputs)]

        # Build the true program top-down from the root.
        root_slot = (0, 0)
        self._grow(program, rng, root_slot, depth, occupied)
        program.root_index = 0
        program.active_count = len(program.nodes)
        program.input_names = self._input_order(program)
        self._attach_decoys(program, rng)
        self._collect_candidates(program)
        return program

    def _input_order(self, program: Program) -> List[str]:
        """Distinct input identities in order of first appearance in the program."""
        out: List[str] = []
        seen: Set[str] = set()
        for n in program.nodes:
            if n.is_input and n.in_program and n.input_id is not None:
                name = input_name_of(n)
                if name not in seen:
                    seen.add(name)
                    out.append(name)
        return out

    def _slot_of(self, node: Node) -> Slot:
        return (node.depth, node.slot)

    def _make_node(self, program: Program, op: Op, slot: Slot, in_program: bool) -> Node:
        node = Node(
            index=len(program.nodes),
            op=op,
            depth=slot[0],
            slot=slot[1],
            arity=ARITY[op],
            in_program=in_program,
        )
        program.nodes.append(node)
        return node

    def _pick_op(self, rng: random.Random, allow_not2: bool) -> Op:
        ops = [o for o in self.config.ops if o in INTERNAL_OPS]
        if allow_not2 and Op.NOT2 in self.config.ops:
            ops = ops + [Op.NOT2]
        if not ops:
            raise ValueError("no internal operators available")
        return rng.choice(ops)

    def _grow(self, program: Program, rng: random.Random, slot: Slot, depth_left: int,
              occupied: Dict[Slot, int]) -> Node:
        """Recursively build the true program. Returns the created node."""
        if depth_left <= 1:
            node = self._make_node(program, Op.INPUT, slot, in_program=True)
            node.input_value = bool(rng.getrandbits(1))
            # Assign an identity from the fixed pool so that every consumer of
            # an input name (evaluate_program, oracle, executor) agrees. Without
            # this, input_names stays empty and the exhaustive truth table is
            # vacuous.
            node.input_id = rng.randrange(len(program.input_pool))
            occupied[slot] = node.index
            return node
        op = self._pick_op(rng, allow_not2=True)
        node = self._make_node(program, op, slot, in_program=True)
        occupied[slot] = node.index
        children = self.template.canonical_children(*slot)
        rng.shuffle(children)
        # Arity-1 ops use only the first canonical child.
        keep = children[: ARITY[op]]
        for port, child_slot in enumerate(keep):
            child_node = self._grow(program, rng, child_slot, depth_left - 1, occupied)
            node.args.append(Edge(src=child_node.index, dst=node.index, port=port))
        return node

    # ---- decoys -------------------------------------------------------
    def _attach_decoys(self, program: Program, rng: random.Random) -> None:
        """Attach window decoys and arity decoys.

        - Window decoys: extra subtrees attached at non-canonical admissible
          slots near true parents. Without these, a copy mask (gate ON for every
          occupied admissible edge) scores 100%.
        - Arity decoys: subtrees attached to unused canonical slots of arity-1
          operators. Without these, a static mask fits the geometric rule
          "gate ON iff the child is a geometric canonical child" and scores 97.6%.
        """
        cfg = self.config
        if cfg.max_decoys_per_episode <= 0:
            return
        true_nodes = [n for n in program.nodes if n.in_program and not n.is_input]
        rng.shuffle(true_nodes)
        decoys_made = 0
        occupied: Dict[Slot, int] = {
            (n.depth, n.slot): n.index for n in program.nodes if n.in_program
        }
        for node in true_nodes:
            if decoys_made >= cfg.max_decoys_per_episode:
                break
            if rng.random() > cfg.decoy_prob:
                continue
            slot = self._slot_of(node)
            admissible = self.template.admissible_children(*slot)
            # Window decoy candidates: admissible but non-canonical.
            window_slots = [s for s in admissible if s not in occupied
                            and not self.template.is_canonical(slot, s)]
            # Arity decoy candidates: a canonical slot that the parent's arity
            # leaves unused (only possible for arity-1 parents).
            arity_slots = []
            if node.arity == 1:
                for s in self.template.canonical_children(*slot):
                    if s not in occupied:
                        arity_slots.append(s)
            # Prefer an arity decoy when the parent is arity-1 (that is the
            # ambiguity this benchmark is designed to create), else a window
            # decoy.
            pool = arity_slots if arity_slots else window_slots
            if not pool:
                continue
            decoy_slot = rng.choice(pool)
            depth_left = max(1, self.config.max_depth - decoy_slot[0])
            before = len(program.nodes)
            decoy_root = self._grow_decoy(program, rng, decoy_slot, depth_left, occupied)
            if decoy_root is None:
                continue
            decoys_made += 1
            # Mark the whole new subtree as decoy.
            for n in program.nodes[before:]:
                n.decoy_subtree = True
            decoy_root.is_decoy_root = True
            # Wire the boundary edge into the parent's args so the decoy subtree
            # physically competes with the true child on a real argument port.
            # Without this the decoy is dangling: the executor never sees it, a
            # copy mask scores 100%, and the benchmark's ambiguity engineering is
            # silently inert. `is_decoy` keeps it out of the ground truth while
            # keeping it in the execution graph.
            # Both decoy kinds land on port 0 -- the port the parent actually
            # reads. A unary op only evaluates port 0, so an arity decoy placed
            # on port 1 would never be executed and could not be penalised.
            edge = Edge(src=decoy_root.index, dst=node.index, port=0,
                        is_decoy=True)
            node.args.append(edge)
            program.candidate_edges.append(edge)

    def _grow_decoy(self, program: Program, rng: random.Random, slot: Slot,
                    depth_left: int, occupied: Dict[Slot, int]) -> Optional[Node]:
        """Grow a decoy subtree rooted at ``slot``. Returns its root node."""
        if slot in occupied:
            return None
        if depth_left <= 0:
            return None
        # Decoy subtrees are built from the internal op set only (no NOT2): they
        # exist to create ambiguity, not to introduce the held-in operator.
        op = rng.choice([o for o in INTERNAL_OPS if o in self.config.ops])
        if not any(o in self.config.ops for o in INTERNAL_OPS):
            return None
        # At the depth floor an op would have no room for its children, leaving
        # it with fewer args than its arity -- an unsatisfiable node. Make a
        # leaf instead, mirroring `_grow`'s handling of the true program.
        if depth_left <= 1 or op == Op.INPUT:
            root = self._make_node(program, Op.INPUT, slot, in_program=False)
            root.input_value = bool(rng.getrandbits(1))
            root.decoy_name = self._fresh_decoy_name(program)
            occupied[slot] = root.index
            return root
        root = self._make_node(program, op, slot, in_program=False)
        occupied[slot] = root.index
        children = self.template.canonical_children(*slot)
        rng.shuffle(children)
        for port, child_slot in enumerate(children[: root.arity]):
            child: Optional[Node] = None
            if child_slot not in occupied and depth_left - 1 > 0:
                child = self._grow_decoy(program, rng, child_slot,
                                         depth_left - 1, occupied)
            if child is None:
                # Attach a leaf to keep the parent's arity satisfied.
                child = self._make_node(program, Op.INPUT, child_slot,
                                        in_program=False)
                child.input_value = bool(rng.getrandbits(1))
                child.decoy_name = self._fresh_decoy_name(program)
                occupied[child_slot] = child.index
            root.args.append(Edge(src=child.index, dst=root.index, port=port))
        return root

    def _fresh_decoy_name(self, program: Program) -> str:
        """A name unique to this episode's decoy leaves, disjoint from `x*`.

        Two decoy leaves may share a name (they are the same kind of object), but
        no decoy name may collide with a true program input.
        """
        n = getattr(program, "_decoy_counter", 0)
        program._decoy_counter = n + 1
        return f"d{n}"

    # ---- candidates ----------------------------------------------------
    def _collect_candidates(self, program: Program) -> None:
        """Union true edges and decoy boundary edges into the candidate set."""
        seen: Set[Tuple[int, int, int]] = set()
        out: List[Edge] = []
        for n in program.nodes:
            if not n.in_program:
                continue
            for e in n.args:
                if e.key() not in seen:
                    seen.add(e.key())
                    out.append(e)
        for e in program.candidate_edges:
            if e.key() not in seen:
                seen.add(e.key())
                out.append(e)
        # Dedup preserves first-seen order.
        program.candidate_edges = out
        # Rebuild true_edges from node args, which are authoritative. A decoy
        # boundary edge is in args but never ground truth.
        program.true_edges = []
        for n in program.nodes:
            if not n.in_program:
                continue
            for e in n.args:
                if e.is_decoy:
                    continue
                program.true_edges.append(e)


def role_for(parent_op: Op, port: int, is_canonical: bool) -> str:
    """Public router role for a candidate edge (see superset.role_of_edge)."""
    return role_of_edge(parent_op in ARITY1_OPS, port, is_canonical)


def input_name_of(node: Node) -> str:
    """Canonical input name for a leaf node.

    Single source of truth shared by ``evaluate_program``, the oracle's
    input-flip, and the executor. All three must agree, otherwise the exhaustive
    truth table and every accuracy metric derived from it is silently vacuous.

    Two true leaves *share* an identity iff they carry the same ``input_id``: the
    benchmark bounds distinct active inputs (constraint C1) so the exhaustive
    truth table stays at <= 8 rows, but a depth-5 binary tree has up to 16
    leaves. The name is therefore drawn from the identity pool, not from the
    leaf's position -- a positional name would make every leaf distinct and
    silently exceed the exhaustive limit.

    A decoy leaf carries its own disjoint name so it can never be confused with
    a program input; see ``Node.decoy_name``.
    """
    if node.decoy_name is not None:
        return node.decoy_name
    if node.input_id is None:
        raise ValueError(
            f"leaf node {node.index} has no input_id; the generator must assign "
            f"one before any consumer names it"
        )
    return f"x{node.input_id}"


def evaluate_program(program: Program, assignment: Dict[str, bool]) -> bool:
    """Exact Boolean evaluation of the true program under ``assignment``."""
    cache: Dict[int, bool] = {}

    def value_of(node_index: int) -> bool:
        if node_index in cache:
            return cache[node_index]
        node = program.nodes[node_index]
        if not node.in_program:
            raise ValueError(f"decoy node {node_index} has no defined value")
        if node.is_input:
            name = input_name_of(node)
            if name not in assignment:
                raise KeyError(f"missing input {name}")
            v = assignment[name]
        else:
            # A decoy boundary edge is physically wired into args but is not
            # ground truth; the exact reference evaluates true args only.
            args = tuple(value_of(e.src) for e in node.args if not e.is_decoy)
            v = boolean_eval(node.op, args)
        cache[node_index] = v
        return v

    return value_of(program.root_index)


__all__ = [
    "Edge",
    "Node",
    "Program",
    "GeneratorConfig",
    "BooleanDAGGenerator",
    "role_for",
    "evaluate_program",
    "input_name_of",
]
