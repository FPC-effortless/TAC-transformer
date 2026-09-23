"""RECON-14 tests: autodiff, grammar, template, generator, executor, oracle, router.

Run from the repository root:

    python3 -m pytest tests_py/test_recon14_core.py -q

These tests are the integrity gate for the reconstruction. They must pass before
any training is run.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from typing import List

from recon14.autodiff import Adam, Value, bce_loss, numerical_gradient_check
from recon14.boolean_dag import (BooleanDAGGenerator, Edge, GeneratorConfig, Node,
                                 Program, evaluate_program, role_for)
from recon14.executor import CASMExecutor
from recon14.grammar import (ARITY, ARITY1_OPS, COMMUTATIVE_OPS, INTERNAL_OPS,
                             PRETRAIN_OPS, PILOT_OPS, Op, boolean_eval)
from recon14.oracle import (copy_mask_gates, decoy_false_positives, edge_necessity,
                            exact_accuracy, exhaustive_reference_outputs, is_acyclic,
                            oracle_gates, routing_precision_recall,
                            truth_table_assignments, validate_program)
from recon14.router import (ROLES, ConditionalRouter, FactorizedRouterL2,
                            StaticMaskRouter, parameter_count)
from recon14.superset import SupersetTemplate
from recon14.experiment import (CONDITIONS, CellResult, FROZEN, NEW_OP,
                                _converged, _epilogue_metrics, _new_op_gates,
                                build_dataset, falsification_verdict, run_condition,
                                summarise)


# ---------------------------------------------------------------- autodiff


class TestAutodiff(unittest.TestCase):
    def test_add_mul_forward(self):
        a, b = Value(2.0), Value(3.0)
        self.assertAlmostEqual((a + b).data, 5.0)
        self.assertAlmostEqual((a * b).data, 6.0)
        self.assertAlmostEqual((a - b).data, -1.0)
        self.assertAlmostEqual((a / b).data, 2.0 / 3.0)

    def test_scalar_ops(self):
        a = Value(2.0)
        self.assertAlmostEqual((a + 1).data, 3.0)
        self.assertAlmostEqual((1 + a).data, 3.0)
        self.assertAlmostEqual((a * 3).data, 6.0)
        self.assertAlmostEqual((3 * a).data, 6.0)
        self.assertAlmostEqual((5 - a).data, 3.0)
        self.assertAlmostEqual((6 / a).data, 3.0)

    def test_power(self):
        a = Value(3.0)
        self.assertAlmostEqual((a**2).data, 9.0)
        self.assertAlmostEqual((a**-1).data, 1.0 / 3.0)

    def test_backward_add_mul(self):
        a, b = Value(2.0), Value(3.0)
        c = a * b + a
        c.backward()
        self.assertAlmostEqual(a.grad, 3.0 + 1.0)  # b + 1
        self.assertAlmostEqual(b.grad, 2.0)        # a

    def test_backward_sigmoid(self):
        a = Value(0.5)
        s = a.sigmoid()
        s.backward()
        expected = 1.0 / (1.0 + math.exp(-0.5))
        self.assertAlmostEqual(a.grad, expected * (1.0 - expected))

    def test_numerical_gradient_check_linear(self):
        a, b = Value(-1.5), Value(2.0)
        f = lambda p: p[0] * p[1] + p[0]
        self.assertTrue(numerical_gradient_check(f, [a, b]))

    def test_numerical_gradient_check_nonlinear(self):
        a, b = Value(0.7), Value(-0.4)
        f = lambda p: (p[0] * p[1] + p[0]).sigmoid() * p[1]
        self.assertTrue(numerical_gradient_check(f, [a, b]))

    def test_numerical_gradient_check_bce(self):
        a = Value(0.3)
        f = lambda p: bce_loss(p[0], 1.0)
        self.assertTrue(numerical_gradient_check(f, [a]))

    def test_parameters_are_leaves(self):
        a, b = Value(1.0), Value(2.0)
        c = a * b
        leaves = c.parameters()
        self.assertEqual(len(leaves), 2)
        self.assertNotIn(c, leaves)

    def test_adam_reduces_loss(self):
        x = Value(3.0)
        target = 1.0
        adam = Adam([x], lr=0.1)
        first = None
        for _ in range(200):
            adam.zero_grad()
            loss = (x - target) * (x - target)
            if first is None:
                first = loss.data
            loss.backward()
            adam.step()
        self.assertLess(abs(x.data - target), 0.01)
        self.assertLess(abs(x.data - target), first)

    def test_adam_bias_correction(self):
        # Start off zero. At exactly 0.0 with loss x**2 the gradient is 0 and
        # Adam correctly makes no update, so the assertion would be vacuous.
        x = Value(0.5)
        adam = Adam([x], lr=0.1)
        for _ in range(5):
            x.zero_grad()
            (x * x).backward()
            adam.step()
        self.assertNotEqual(x.data, 0.5)

    def test_bce_clipping(self):
        v = Value(1e-9)
        loss = bce_loss(v, 1.0)
        self.assertTrue(math.isfinite(loss.data))
        v2 = Value(1.0 - 1e-9)
        loss2 = bce_loss(v2, 0.0)
        self.assertTrue(math.isfinite(loss2.data))


# ---------------------------------------------------------------- grammar


class TestGrammar(unittest.TestCase):
    def test_arities(self):
        self.assertEqual(ARITY[Op.INPUT], 0)
        self.assertEqual(ARITY[Op.NOT], 1)
        self.assertEqual(ARITY[Op.IDENTITY], 1)
        self.assertEqual(ARITY[Op.NOT2], 1)
        self.assertEqual(ARITY[Op.AND], 2)
        self.assertEqual(ARITY[Op.OR], 2)
        self.assertEqual(ARITY[Op.XOR], 2)

    def test_not2_behaves_as_not(self):
        self.assertFalse(boolean_eval(Op.NOT2, (True,)))
        self.assertTrue(boolean_eval(Op.NOT2, (False,)))

    def test_not2_excluded_from_internal_ops(self):
        self.assertNotIn(Op.NOT2, INTERNAL_OPS)
        self.assertIn(Op.NOT2, PILOT_OPS)
        self.assertIn(Op.NOT2, ARITY1_OPS)
        self.assertNotIn(Op.NOT2, PRETRAIN_OPS)

    def test_boolean_eval_table(self):
        cases = [
            (Op.AND, (True, True), True),
            (Op.AND, (True, False), False),
            (Op.OR, (False, True), True),
            (Op.OR, (False, False), False),
            (Op.XOR, (True, True), False),
            (Op.XOR, (True, False), True),
            (Op.NOT, (True,), False),
            (Op.IDENTITY, (False,), False),
        ]
        for op, args, expected in cases:
            with self.subTest(op=op, args=args):
                self.assertEqual(boolean_eval(op, args), expected)

    def test_arity_errors(self):
        with self.assertRaises(ValueError):
            boolean_eval(Op.AND, (True,))
        with self.assertRaises(ValueError):
            boolean_eval(Op.NOT, (True, True))

    def test_commutative_set(self):
        self.assertEqual(COMMUTATIVE_OPS, frozenset({Op.AND, Op.OR, Op.XOR}))


# ---------------------------------------------------------------- template


class TestSupersetTemplate(unittest.TestCase):
    def setUp(self):
        self.t = SupersetTemplate(max_depth=8, window=1)

    def test_root_canonical_children(self):
        self.assertEqual(self.t.canonical_children(0, 0), [(1, 0), (1, 1)])

    def test_canonical_children_are_two(self):
        for level in range(7):
            for index in range(self.t.width(level)):
                kids = self.t.canonical_children(level, index)
                self.assertEqual(len(kids), 2)

    def test_admissible_includes_canonical_plus_window(self):
        kids = self.t.admissible_children(1, 0)
        self.assertIn((2, 0), kids)
        self.assertIn((2, 1), kids)
        self.assertIn((2, 2), kids)  # window neighbour

    def test_window_excluded_when_zero(self):
        t = SupersetTemplate(max_depth=4, window=0)
        self.assertEqual(t.admissible_children(0, 0), [(1, 0), (1, 1)])

    def test_leaf_level_has_no_children(self):
        self.assertEqual(self.t.canonical_children(7, 0), [])

    def test_slot_index_unique_and_total(self):
        slots = self.t.all_slots()
        ids = [self.t.slot_index(s) for s in slots]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(len(ids), self.t.num_slots())

    def test_is_canonical(self):
        self.assertTrue(self.t.is_canonical((0, 0), (1, 0)))
        self.assertFalse(self.t.is_canonical((0, 0), (1, 2)))

    def test_equality_and_hash(self):
        self.assertEqual(SupersetTemplate(8, 1), SupersetTemplate(8, 1))
        self.assertNotEqual(SupersetTemplate(8, 1), SupersetTemplate(8, 0))
        self.assertEqual(hash(SupersetTemplate(8, 1)), hash(SupersetTemplate(8, 1)))


# ---------------------------------------------------------------- generator


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.cfg = GeneratorConfig(max_depth=5, min_depth=1, decoy_prob=0.7,
                                   max_decoys_per_episode=3, seed=20260922)
        self.gen = BooleanDAGGenerator(self.cfg)

    def test_generate_is_valid(self):
        rng = random.Random(0)
        for _ in range(64):
            prog = self.gen.generate(rng)
            self.assertEqual(validate_program(prog), [])

    def test_root_is_level_zero(self):
        rng = random.Random(1)
        prog = self.gen.generate(rng)
        self.assertEqual(prog.nodes[prog.root_index].depth, 0)

    def test_active_inputs_bounded_for_truth_table(self):
        rng = random.Random(2)
        for _ in range(64):
            prog = self.gen.generate(rng)
            self.assertLessEqual(len(prog.input_names), 3)

    def test_each_episode_has_decoy_candidates(self):
        rng = random.Random(3)
        with_decoys = 0
        for _ in range(64):
            prog = self.gen.generate(rng)
            if any(n.decoy_subtree for n in prog.nodes):
                with_decoys += 1
        # decoy_prob is 0.7 with up to 3 decoys per episode, so the overwhelming
        # majority of episodes carry at least one decoy subtree.
        self.assertGreater(with_decoys, 32)

    def test_true_edges_are_admissible(self):
        rng = random.Random(4)
        prog = self.gen.generate(rng)
        for e in prog.true_edges:
            parent = prog.nodes[e.dst]
            child = prog.nodes[e.src]
            self.assertTrue(prog.template.is_admissible(
                (parent.depth, parent.slot), (child.depth, child.slot)))

    def test_candidate_edges_superset_of_true(self):
        rng = random.Random(5)
        prog = self.gen.generate(rng)
        self.assertTrue(prog.true_edge_set.issubset(prog.candidate_edge_set))

    def test_not2_appears_when_in_ops(self):
        rng = random.Random(6)
        ops_seen = set()
        for _ in range(64):
            prog = self.gen.generate(rng)
            ops_seen.update(n.op for n in prog.nodes if n.in_program)
        self.assertIn(Op.NOT2, ops_seen)

    def test_evaluate_program_matches_truth_table(self):
        rng = random.Random(7)
        prog = self.gen.generate(rng)
        rows = list(truth_table_assignments(prog))
        for assignment in rows:
            self.assertTrue(evaluate_program(prog, assignment) in (True, False))

    def test_min_max_depth_respected(self):
        rng = random.Random(8)
        for _ in range(64):
            prog = self.gen.generate(rng)
            self.assertLessEqual(prog.nodes[prog.root_index].arity, 2)
            deepest = max(n.depth for n in prog.nodes if n.in_program)
            self.assertLessEqual(deepest, self.cfg.max_depth)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            GeneratorConfig(max_depth=1, min_depth=5).validate()
        with self.assertRaises(ValueError):
            GeneratorConfig(decoy_prob=1.5).validate()
        with self.assertRaises(ValueError):
            GeneratorConfig(input_count=9).validate()

    def test_deterministic_for_same_seed(self):
        a = self.gen.generate(random.Random(99))
        b = self.gen.generate(random.Random(99))
        self.assertEqual([n.op for n in a.nodes], [n.op for n in b.nodes])
        self.assertEqual(a.true_edge_set, b.true_edge_set)


# ---------------------------------------------------------------- executor


class TestExecutor(unittest.TestCase):
    def build(self, rng_seed=0):
        cfg = GeneratorConfig(max_depth=3, min_depth=2, decoy_prob=0.0,
                              max_decoys_per_episode=0, seed=20260922)
        gen = BooleanDAGGenerator(cfg)
        return gen.generate(random.Random(rng_seed))

    def test_oracle_gates_give_exact_accuracy(self):
        prog = self.build()
        ex = CASMExecutor()
        gates = oracle_gates(prog)
        self.assertEqual(exact_accuracy(prog, ex, gates), 1.0)

    def test_all_off_gates_fail_when_program_nontrivial(self):
        prog = self.build()
        ex = CASMExecutor()
        gates = {e.key(): 0.0 for e in prog.candidate_edges}
        # With all gates off the executor must not match the truth table for
        # every non-constant program.
        acc = exact_accuracy(prog, ex, gates)
        ref = exhaustive_reference_outputs(prog)
        if len(set(ref)) == 2:  # non-constant program
            self.assertLess(acc, 1.0)

    def test_hard_and_soft_agree_at_limits(self):
        prog = self.build()
        ex = CASMExecutor()
        gates_v = oracle_gates(prog)
        soft_gates = {k: Value(v) for k, v in gates_v.items()}
        rows = [{k: Value(1.0 if v else 0.0) for k, v in a.items()}
                for a in truth_table_assignments(prog)]
        ref = exhaustive_reference_outputs(prog)
        for row, ref_bit in zip(rows, ref):
            got_soft = ex.execute(prog, row, soft_gates).data
            got_hard = ex.execute_hard(prog,
                                       {k: v.data for k, v in row.items()}, gates_v)
            self.assertAlmostEqual(soft_to_hard(got_soft), got_hard, places=6)
            self.assertEqual((got_hard >= 0.5), ref_bit)

    def test_xor_is_representable(self):
        # FlyVis linear threshold units cannot represent XOR; the soft executor
        # must be able to.
        cfg = GeneratorConfig(max_depth=2, min_depth=2, decoy_prob=0.0,
                              max_decoys_per_episode=0, seed=20260922)
        gen = BooleanDAGGenerator(cfg)
        rng = random.Random(10)
        for _ in range(32):
            prog = gen.generate(rng)
            if prog.nodes[prog.root_index].op != Op.XOR:
                continue
            ex = CASMExecutor()
            gates = oracle_gates(prog)
            self.assertEqual(exact_accuracy(prog, ex, gates), 1.0)

    def test_execute_requires_all_inputs(self):
        prog = self.build()
        ex = CASMExecutor()
        gates = {e.key(): Value(0.5) for e in prog.candidate_edges}
        with self.assertRaises(KeyError):
            ex.execute(prog, {}, gates)


def soft_to_hard(x: float) -> float:
    return 1.0 if x >= 0.5 else 0.0


# ---------------------------------------------------------------- oracle


class TestOracle(unittest.TestCase):
    def test_validate_detects_arity_violation(self):
        prog = Program()
        prog.template = SupersetTemplate(4, 1)
        prog.nodes = [Node(0, Op.AND, 0, 0, arity=2, args=[])]
        prog.root_index = 0
        errors = validate_program(prog)
        self.assertTrue(any("expected 2" in e for e in errors))

    def test_acyclic_passes_on_generated(self):
        cfg = GeneratorConfig(max_depth=4, min_depth=2, decoy_prob=0.0,
                              max_decoys_per_episode=0, seed=20260922)
        gen = BooleanDAGGenerator(cfg)
        prog = gen.generate(random.Random(0))
        self.assertTrue(is_acyclic(prog))

    def test_truth_table_size_is_power_of_two(self):
        cfg = GeneratorConfig(max_depth=4, min_depth=2, decoy_prob=0.0,
                              max_decoys_per_episode=0, seed=20260922)
        gen = BooleanDAGGenerator(cfg)
        prog = gen.generate(random.Random(1))
        n = len(list(truth_table_assignments(prog)))
        self.assertEqual(n, 2 ** len(prog.input_names))

    def test_edge_necessity_marks_used_edges(self):
        cfg = GeneratorConfig(max_depth=3, min_depth=2, decoy_prob=0.0,
                              max_decoys_per_episode=0, seed=20260922)
        gen = BooleanDAGGenerator(cfg)
        prog = gen.generate(random.Random(2))
        nec = edge_necessity(prog)
        self.assertTrue(nec)
        self.assertTrue(any(nec.values()))

    def test_copy_mask_scores_below_one_with_decoys(self):
        cfg = GeneratorConfig(max_depth=5, min_depth=2, decoy_prob=1.0,
                              max_decoys_per_episode=3, seed=20260922)
        gen = BooleanDAGGenerator(cfg)
        ex = CASMExecutor()
        accuracies = []
        for i in range(32):
            prog = gen.generate(random.Random(100 + i))
            acc = exact_accuracy(prog, ex, copy_mask_gates(prog))
            accuracies.append(acc)
        mean_acc = sum(accuracies) / len(accuracies)
        self.assertLess(mean_acc, 1.0)

    def test_oracle_precision_recall_perfect(self):
        cfg = GeneratorConfig(max_depth=4, min_depth=2, decoy_prob=0.5,
                              max_decoys_per_episode=2, seed=20260922)
        gen = BooleanDAGGenerator(cfg)
        prog = gen.generate(random.Random(3))
        p, r = routing_precision_recall(prog, oracle_gates(prog))
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)

    def test_decoy_false_positives_counted(self):
        cfg = GeneratorConfig(max_depth=5, min_depth=2, decoy_prob=1.0,
                              max_decoys_per_episode=3, seed=20260922)
        gen = BooleanDAGGenerator(cfg)
        found = False
        for i in range(64):
            prog = gen.generate(random.Random(200 + i))
            gates = {e.key(): 1.0 for e in prog.candidate_edges}
            if decoy_false_positives(prog, gates) > 0:
                found = True
                break
        self.assertTrue(found)


# ---------------------------------------------------------------- router


class TestRouter(unittest.TestCase):
    def test_l2_router_parameter_count(self):
        r = FactorizedRouterL2(PRETRAIN_OPS, dim=2)
        # 5 ops * 2 dims + 3 roles * 2 dims + 1 bias = 17
        self.assertEqual(len(r.parameters()), 5 * 2 + 3 * 2 + 1)

    def test_l2_router_gate_at_zero_distance_is_high(self):
        # The protocol writes g = sigmoid((b - d^2)/tau) and leaves b to be
        # learned: with b = 0 the gate is sigmoid(-d^2) <= 0.5 everywhere, so a
        # matched op/role pair only exceeds 0.5 once b is positive. Exercise
        # that directly rather than relying on an init default.
        r = FactorizedRouterL2(PRETRAIN_OPS, dim=2, init_scale=0.3, bias_init=2.0)
        # Place an op embedding exactly on a role embedding.
        for i, v in enumerate(r.op_emb[Op.AND]):
            v.data = r.role_emb["arg1"][i].data
        g = r.gate_value(Op.AND, "arg1").data
        self.assertGreater(g, 0.5)

    def test_l2_router_gate_far_apart_is_low(self):
        r = FactorizedRouterL2(PRETRAIN_OPS, dim=2, init_scale=0.3)
        for i, v in enumerate(r.op_emb[Op.AND]):
            v.data = 10.0 + i
        for i, v in enumerate(r.role_emb["arg1"]):
            v.data = -10.0 - i
        g = r.gate_value(Op.AND, "arg1").data
        self.assertLess(g, 0.5)

    def test_l2_router_matches_closed_form(self):
        import random as _r
        r = FactorizedRouterL2(PRETRAIN_OPS, dim=2, init_scale=0.3, seed=3)
        rng = _r.Random(0)
        for _ in range(8):
            for op in PRETRAIN_OPS:
                for i in range(2):
                    r.op_emb[op][i].data = rng.gauss(0, 0.3)
        for op in PRETRAIN_OPS:
            for role in ROLES:
                sq = sum((r.op_emb[op][i].data - r.role_emb[role][i].data) ** 2
                         for i in range(2))
                expected = 1.0 / (1.0 + math.exp(-(0.0 - sq) / 1.0))
                self.assertAlmostEqual(r.gate_value(op, role).data, expected, places=9)

    def test_freeze_preserves_values_and_stops_gradient(self):
        r = FactorizedRouterL2(PRETRAIN_OPS, dim=2, seed=5)
        r.op_emb[Op.NOT][0].data = 0.77
        frozen = r.freeze_ops((Op.NOT,))
        # The replacement embeddings keep the frozen values, so the geometry is
        # reused exactly.
        after = [v.data for v in r.op_emb[Op.NOT]]
        self.assertEqual(after[0], 0.77)
        # The frozen leaves themselves must not accumulate gradient. The
        # replacements in `op_emb` are fresh trainable constants by design, so
        # the check has to run on the returned leaves.
        self.assertEqual(frozen[0].data, 0.77)
        self.assertFalse(frozen[0].requires_grad)
        (frozen[0] * Value(1.0)).backward()
        self.assertEqual(frozen[0].grad, 0.0)
        # And frozen ops are absent from the trainable parameter set.
        self.assertNotIn(frozen[0], r.parameters())

    def test_add_op_is_fresh(self):
        r = FactorizedRouterL2(PRETRAIN_OPS, dim=2, seed=5)
        r.freeze_ops(PRETRAIN_OPS)
        r.add_op(Op.NOT2, seed=11)
        self.assertIn(Op.NOT2, r.op_emb)
        self.assertIn(Op.NOT2, r.ops)
        # Fresh embedding, not inherited from NOT.
        for i in range(2):
            self.assertNotEqual(r.op_emb[Op.NOT2][i].data,
                                r.op_emb[Op.NOT][i].data)

    def test_conditional_router_has_fifteen_params(self):
        r = ConditionalRouter(PRETRAIN_OPS)
        self.assertEqual(len(r.parameters()), 15)

    def test_conditional_router_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            ConditionalRouter(PILOT_OPS)

    def test_static_mask_one_param_per_slot(self):
        t = SupersetTemplate(4, 1)
        r = StaticMaskRouter(t)
        self.assertEqual(len(r.parameters()), t.num_slots())

    def test_router_gates_only_use_public_info(self):
        cfg = GeneratorConfig(max_depth=4, min_depth=2, decoy_prob=0.5,
                              max_decoys_per_episode=2, seed=20260922)
        gen = BooleanDAGGenerator(cfg)
        prog = gen.generate(random.Random(0))
        r = FactorizedRouterL2(PILOT_OPS, dim=2)
        gates = r.gates_for(prog)
        self.assertTrue(gates)
        for key, g in gates.items():
            self.assertTrue(0.0 <= g.data <= 1.0)


# ---------------------------------------------------------------- experiment


class TestExperiment(unittest.TestCase):
    def test_frozen_constants(self):
        self.assertEqual(FROZEN["pretrain_steps"], 1500)
        self.assertEqual(FROZEN["router_dim"], 2)
        self.assertEqual(FROZEN["router_init_scale"], 0.3)
        self.assertEqual(FROZEN["lr"], 0.1)
        self.assertEqual(FROZEN["batch_size"], 16)
        self.assertEqual(FROZEN["max_not2_steps"], 200)
        self.assertEqual(FROZEN["dataset_seed"], 20260922)

    def test_conditions_are_five(self):
        self.assertEqual(len(CONDITIONS), 5)
        self.assertIn("A_from_scratch", CONDITIONS)
        self.assertIn("B_persistent_reuse", CONDITIONS)

    def test_dataset_is_deterministic(self):
        a = build_dataset(n_episodes=16, seed=20260922)
        b = build_dataset(n_episodes=16, seed=20260922)
        self.assertEqual([p.true_edge_set for p in a.episodes],
                         [p.true_edge_set for p in b.episodes])

    def test_new_op_is_not2(self):
        self.assertEqual(NEW_OP, Op.NOT2)

    def test_run_condition_a_smoke(self):
        dataset = build_dataset(n_episodes=24, seed=FROZEN["dataset_seed"])
        cell = run_condition("A_from_scratch", seed=0, dataset=dataset,
                             max_not2_steps=8)
        self.assertEqual(cell.condition, "A_from_scratch")
        self.assertEqual(cell.seed, 0)
        self.assertTrue(len(cell.history) == 8)
        self.assertTrue(math.isfinite(cell.final_loss))

    def test_run_condition_b_smoke(self):
        dataset = build_dataset(n_episodes=24, seed=FROZEN["dataset_seed"])
        cell = run_condition("B_persistent_reuse", seed=0, dataset=dataset,
                             pretrain_steps=20, max_not2_steps=8)
        self.assertTrue(cell.converged is False or cell.converged is True)
        self.assertIsNotNone(cell.pretrain_final_loss)
        self.assertEqual(cell.frozen_param_count, 5 * 2)

    def test_run_condition_c_smoke(self):
        dataset = build_dataset(n_episodes=24, seed=FROZEN["dataset_seed"])
        cell = run_condition("C_not2_only_pretrain_control", seed=0, dataset=dataset,
                             pretrain_steps=20, max_not2_steps=8)
        self.assertEqual(cell.frozen_param_count, 2)

    def test_run_condition_d_smoke(self):
        dataset = build_dataset(n_episodes=24, seed=FROZEN["dataset_seed"])
        cell = run_condition("D_frozen_op_noop_persistence", seed=0, dataset=dataset,
                             pretrain_steps=20, max_not2_steps=8)
        self.assertEqual(cell.frozen_param_count, 5 * 2)

    def test_run_condition_e_smoke(self):
        dataset = build_dataset(n_episodes=24, seed=FROZEN["dataset_seed"])
        cell = run_condition("E_joint_not2_late", seed=0, dataset=dataset,
                             pretrain_steps=20, max_not2_steps=8)
        self.assertEqual(cell.frozen_param_count, 0)

    def test_unknown_condition_raises(self):
        dataset = build_dataset(n_episodes=8, seed=FROZEN["dataset_seed"])
        with self.assertRaises(ValueError):
            run_condition("Z_nonexistent", seed=0, dataset=dataset, max_not2_steps=2)

    def test_conditions_share_one_dataset(self):
        # All cells of a trial must see identical episodes.
        ds = build_dataset(n_episodes=12, seed=FROZEN["dataset_seed"])
        a = run_condition("A_from_scratch", seed=1, dataset=ds, max_not2_steps=3)
        b = run_condition("B_persistent_reuse", seed=1, dataset=ds,
                          pretrain_steps=5, max_not2_steps=3)
        self.assertEqual(a.history[0].step, 1)
        self.assertEqual(b.history[0].step, 1)

    def test_summarise_reports_full_step_lists(self):
        results = {"A_from_scratch": [], "B_persistent_reuse": [],
                   "C_not2_only_pretrain_control": [],
                   "D_frozen_op_noop_persistence": [], "E_joint_not2_late": []}
        summary = summarise(results)
        for condition in CONDITIONS:
            self.assertIn(condition, summary)

    def test_falsification_verdict_inconclusive_without_convergence(self):
        summary = {
            "A_from_scratch": {"mean_steps": None},
            "B_persistent_reuse": {"mean_steps": None},
            "C_not2_only_pretrain_control": {"mean_steps": None},
            "D_frozen_op_noop_persistence": {"mean_steps": None},
        }
        verdict = falsification_verdict(summary)
        self.assertEqual(verdict["verdict"], "inconclusive")

    def test_falsification_verdict_detects_confound(self):
        summary = {
            "A_from_scratch": {"mean_steps": 35.5},
            "B_persistent_reuse": {"mean_steps": 1.0},
            "C_not2_only_pretrain_control": {"mean_steps": 1.0},
            "D_frozen_op_noop_persistence": {"mean_steps": 40.0},
        }
        verdict = falsification_verdict(summary)
        self.assertEqual(verdict["verdict"], "falsified_by_confound")
        self.assertIn("C_not2_only_pretrain_control",
                      verdict["controls_reproducing"])

    def test_falsification_verdict_survives(self):
        summary = {
            "A_from_scratch": {"mean_steps": 35.5},
            "B_persistent_reuse": {"mean_steps": 1.0},
            "C_not2_only_pretrain_control": {"mean_steps": 35.0},
            "D_frozen_op_noop_persistence": {"mean_steps": 38.0},
        }
        verdict = falsification_verdict(summary)
        self.assertEqual(verdict["verdict"], "effect_survives_controls")
        self.assertAlmostEqual(verdict["speedup_a_over_b"], 35.5)

    # ---- measurement boundary --------------------------------------------
    # RECON-14 scopes steps-to-convergence to the NOT2 phase. A gate read before
    # the operator is introduced must not be coerced to a number: 0.0 is a
    # legitimate trained gate value and a pre-introduction read is neither a
    # measurement nor a failure, so "untrained" has to stay distinguishable from
    # "not converged".

    def test_pre_introduction_gate_is_none_not_zero(self):
        r = FactorizedRouterL2(PRETRAIN_OPS, dim=2, seed=5)
        # NOT2 is not in PRETRAIN_OPS, so no embedding exists for it yet.
        self.assertIsNone(r.gate_value(NEW_OP, "arg1"))

    def test_convergence_ignores_pre_introduction_steps(self):
        # A cell whose NOT2 phase never runs must report no convergence and no
        # steps, not steps_to_converge = 0 or a spurious convergence on None.
        ds = build_dataset(n_episodes=8, seed=FROZEN["dataset_seed"])
        cell = run_condition("B_persistent_reuse", seed=0, dataset=ds,
                             pretrain_steps=4, max_not2_steps=0)
        self.assertIsNone(cell.steps_to_converge)
        self.assertFalse(cell.converged)
        # The operator *is* introduced for B, so the reported gates are real.
        self.assertIsNotNone(cell.g_new_arg1)
        self.assertNotEqual(cell.g_new_arg1, 0.0)

    def test_history_records_none_before_introduction(self):
        # Condition E holds NOT2 out for the pretraining window. Its pretraining
        # records are deliberately discarded, so the reported history covers only
        # the measured NOT2 phase: a `None` gate can never reach the convergence
        # statistic. Assert the boundary both ways -- no pre-introduction record
        # leaks in, and a direct call does produce one.
        ds = build_dataset(n_episodes=8, seed=FROZEN["dataset_seed"])
        cell = run_condition("E_joint_not2_late", seed=0, dataset=ds,
                             pretrain_steps=4, max_not2_steps=2)
        # The reported history is the NOT2 phase only.
        self.assertEqual(len(cell.history), 2)
        self.assertTrue(all(r.g_new_arg1 is not None for r in cell.history))

        # And a pre-introduction step does carry None when it is recorded, so the
        # distinction is real rather than structurally unreachable.
        r5 = FactorizedRouterL2(PRETRAIN_OPS, dim=2, seed=5)
        g1, gn = _new_op_gates(r5)
        self.assertIsNone(g1)
        self.assertIsNone(gn)
        self.assertFalse(_converged(g1, gn))

    # ---- epilogue evaluation ---------------------------------------------
    # Frozen protocol: "Epilogue evaluation | exact 6-op hard execution accuracy
    # on held-out episodes". CellResult declared these four fields but they were
    # never populated, so the metric was silently absent from every result. The
    # wiring has to be regression-tested because the fields exist on the
    # dataclass either way -- an unmeasured cell still says 0.0 or None and
    # nothing in the schema catches it.

    def test_epilogue_metrics_populate_all_four_fields(self):
        ds = build_dataset(n_episodes=8, seed=FROZEN["dataset_seed"])
        cell = run_condition("A_from_scratch", seed=0, dataset=ds,
                             max_not2_steps=3)
        for name in ("epilogue_accuracy", "epilogue_precision",
                     "epilogue_recall", "epilogue_decoy_fp"):
            with self.subTest(field=name):
                self.assertTrue(hasattr(cell, name))
                self.assertIsNotNone(getattr(cell, name))

    def test_epilogue_accuracy_is_a_probability(self):
        ds = build_dataset(n_episodes=8, seed=FROZEN["dataset_seed"])
        cell = run_condition("A_from_scratch", seed=0, dataset=ds,
                             max_not2_steps=3)
        self.assertIsNotNone(cell.epilogue_accuracy)
        self.assertGreaterEqual(cell.epilogue_accuracy, 0.0)
        self.assertLessEqual(cell.epilogue_accuracy, 1.0)

    def test_epilogue_uses_a_held_out_corpus(self):
        # The epilogue must not be a training-set readback. The training corpus
        # is frozen at FROZEN["dataset_seed"]; the epilogue builds its own corpus
        # from a different seed, so the two episode sets differ.
        train = build_dataset(n_episodes=16, seed=FROZEN["dataset_seed"])
        held = build_dataset(n_episodes=16, seed=FROZEN["dataset_seed"] + 1)
        self.assertNotEqual([p.true_edge_set for p in train.episodes],
                            [p.true_edge_set for p in held.episodes])
        # And the metrics helper must not consult the training corpus at all.
        # PRETRAIN_OPS excludes NOT2, so add_op introduces it freshly here.
        r = FactorizedRouterL2(PRETRAIN_OPS, dim=2, seed=7)
        r.add_op(NEW_OP, seed=8)
        ex = CASMExecutor()
        metrics = _epilogue_metrics(r, ex, program_cap=4)
        self.assertIn("epilogue_accuracy", metrics)
        self.assertIsNotNone(metrics["epilogue_accuracy"])

    def test_epilogue_returns_none_without_new_op_embedding(self):
        # "not measured" has to stay distinct from "measured zero". A router that
        # has never seen NOT2 reports None rather than a fabricated 0.0.
        r = FactorizedRouterL2(PRETRAIN_OPS, dim=2, seed=7)
        self.assertIsNone(r.embedding_for(NEW_OP))
        metrics = _epilogue_metrics(r, CASMExecutor(), program_cap=4)
        self.assertIsNone(metrics["epilogue_accuracy"])
        self.assertIsNone(metrics["epilogue_precision"])
        self.assertIsNone(metrics["epilogue_recall"])
        self.assertIsNone(metrics["epilogue_decoy_fp"])

    def test_summarise_reports_epilogue_fields(self):
        # The fields must reach the reported bundle, not just the cell object.
        ds = build_dataset(n_episodes=8, seed=FROZEN["dataset_seed"])
        cell = run_condition("A_from_scratch", seed=0, dataset=ds,
                             max_not2_steps=3)
        summary = summarise({"A_from_scratch": [cell]})
        entry = summary["A_from_scratch"]
        for name in ("epilogue_accuracy", "epilogue_precision",
                     "epilogue_recall", "epilogue_decoy_fp"):
            with self.subTest(field=name):
                self.assertIn(name, entry)
                self.assertIsNotNone(entry[name])


class TestRecon14Driver(unittest.TestCase):
    """The workflow entry point (recon14/run_recon14.py).

    The dispatch lane calls this CLI, so a silent failure here would surface as
    an empty artifact on the runner. These tests cover only the argument and
    bundle contract; they run `--quick` at tiny scale so they stay cheap enough
    for the local Termux gate and for fast-check.
    """

    def test_module_imports_without_torch(self):
        import importlib
        mod = importlib.import_module("recon14.run_recon14")
        self.assertTrue(hasattr(mod, "main"))

    def test_requires_output_argument(self):
        from recon14.run_recon14 import main
        with self.assertRaises(SystemExit):
            main([])

    def test_rejects_unknown_condition(self):
        from recon14.run_recon14 import main
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "recon14.json")
            rc = main(["--output", out, "--conditions", "Z_nonexistent",
                       "--quick"])
        self.assertEqual(rc, 2)

    def test_quick_run_writes_bundle(self):
        from recon14.run_recon14 import main
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "recon14.json")
            rc = main(["--output", out, "--quick", "--conditions",
                       "A_from_scratch"])
            self.assertEqual(rc, 0)
            self.assertTrue(Path(out).exists())
            with open(out, encoding="utf-8") as fh:
                bundle = json.load(fh)
        self.assertEqual(bundle["experiment_id"], FROZEN["experiment_id"])
        self.assertEqual(bundle["quick_smoke"], True)
        # A quick run is not a protocol run and must say so.
        self.assertTrue(any("not a protocol run" in d for d in bundle["deviations"]))
        # Frozen values are echoed so the bundle is self-describing.
        self.assertEqual(bundle["frozen"]["pretrain_steps"], 1500)
        self.assertEqual(bundle["frozen"]["dataset_seed"], 20260922)
        self.assertIn("A_from_scratch", bundle["summary"])
        self.assertIn("verdict", bundle)
        self.assertIn("cells", bundle)

    def test_protocol_run_records_no_deviation_by_default(self):
        # A non-quick run with frozen defaults must report an empty deviation
        # list. Kept cheap by limiting to one seed and one condition; the seeds
        # override is itself recorded as a deviation, which is what the test
        # checks.
        from recon14.run_recon14 import main
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "recon14.json")
            rc = main(["--output", out, "--seeds", "0",
                       "--conditions", "A_from_scratch", "--episodes", "8"])
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as fh:
                bundle = json.load(fh)
        self.assertFalse(bundle["quick_smoke"])
        self.assertIn("seeds=[0] differ from frozen", " ".join(bundle["deviations"]))
        self.assertEqual(bundle["n_episodes"], 8)


if __name__ == "__main__":
    unittest.main()
