from __future__ import annotations

import copy
import logging
import math
import os
import pickle
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .adapters import (
    apply_initial_drone_layout_overlay,
    build_ga_context,
    clone_state_for_decode,
    greedy_plan_to_individual,
)
from .chromosome import Individual
from .config import GAConfig, make_ga_config
from .diagnostics import write_evolution_csv, write_evolution_plots, write_mode_precheck_csv
from .decoder import DispatchPlan, GADecoder
from .fitness import compute_fitness
from .operators import mutate, order_crossover, tournament_select
from .physical_evaluator import PhysicalEvaluator
from .population import enforce_fixed_tail, initialize_population


logger = logging.getLogger(__name__)
DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / "ga_mmce_debug_log"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_PLAN_CACHE_SCHEMA = 2
STATIC_PLAN_CACHE_ENV = "GA_MMCE_REUSE_STATIC_PLAN"
STATIC_PLAN_CACHE_PATH_ENV = "GA_MMCE_STATIC_PLAN_CACHE_PATH"


@dataclass
class GAStateView:
    entity_mgr: Any
    orders: dict[str, Any]
    current_time: float
    bbox: dict | None = None
    scene_id: str | None = None
    order_mgr: Any | None = None


class GAMMCESolver:
    """Main GA-MMCE solver orchestrating population evolution and decoding."""

    def __init__(self, entity_mgr: Any, config: GAConfig | None = None):
        self.entity_mgr = entity_mgr
        self.config = config or GAConfig()
        self.greedy_helper = self._make_greedy_helper(entity_mgr)
        self.dynamic_greedy_helper = self._make_dynamic_greedy_helper(entity_mgr)
        self.evaluator = PhysicalEvaluator(entity_mgr, self.greedy_helper, self.config)
        self.decoder = GADecoder(self.config, self.evaluator)
        self.last_best_individual: Individual | None = None
        self.last_best_decode_result: Any | None = None
        self._debug_eval_errors: dict[str, int] = {}
        self._evolution_rows: list[dict[str, Any]] = []
        self._mutation_stats: dict[str, Any] = self._empty_mutation_stats()
        self._last_generation_seconds: float = 0.0
        self._early_stop_info: dict[str, Any] = {}
        self._b_precheck_by_order: dict[str, dict[str, Any]] = {}
        self._active_time_budget_seconds: float | None = None
        self._time_budget_hit: bool = False
        self._actual_generations: int = 0
        self._active_diagnostics_label: str = "static"
        self._reuse_static_plan_cache_override: bool | None = None
        self._last_route_bbox: dict | None = None
        self._last_route_scene_id: str | None = None
        self._order_mgr: Any | None = None

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

    def bind_order_manager(self, order_mgr: Any) -> None:
        """Expose scheduled future orders to the static GA objective."""
        self._order_mgr = order_mgr

    def set_static_plan_cache_reuse(self, enabled: bool | None) -> None:
        """Enable/disable static plan cache reuse for the next GA static dispatch.

        Passing None clears the request-level override and falls back to the
        GA_MMCE_REUSE_STATIC_PLAN environment variable.
        """
        self._reuse_static_plan_cache_override = None if enabled is None else bool(enabled)

    def set_path_planner(self, planner: Any) -> None:
        """Forward the shared UAV obstacle-avoidance planner to GA helper solvers."""
        for helper in (self.greedy_helper, self.dynamic_greedy_helper):
            if hasattr(helper, "set_path_planner"):
                helper.set_path_planner(planner)

    def dispatch(
        self,
        pending_orders: dict[str, Any],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        self._remember_route_context(bbox, scene_id)
        state = GAStateView(
            entity_mgr=self.entity_mgr,
            orders=dict(pending_orders),
            current_time=current_time,
            bbox=bbox,
            scene_id=scene_id,
            order_mgr=self._order_mgr,
        )
        return self.solve(state, dispatch_type="full")

    def dispatch_incremental(
        self,
        new_orders: dict[str, Any],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        self._remember_route_context(bbox, scene_id)
        state = GAStateView(
            entity_mgr=self.entity_mgr,
            orders=dict(new_orders),
            current_time=current_time,
            bbox=bbox,
            scene_id=scene_id,
            order_mgr=self._order_mgr,
        )
        setattr(state, "_ga_solver", self)
        return self.reschedule_on_event(state, new_orders, current_time)

    def dispatch_replan_current_state(
        self,
        replan_orders: dict[str, Any],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        self._remember_route_context(bbox, scene_id)
        state = GAStateView(
            entity_mgr=self.entity_mgr,
            orders=dict(replan_orders),
            current_time=current_time,
            bbox=bbox,
            scene_id=scene_id,
            order_mgr=self._order_mgr,
        )
        setattr(state, "_ga_solver", self)
        return self.reschedule_on_event(state, {}, current_time)

    def reschedule_on_event(
        self,
        state: Any,
        new_orders: dict[str, Any],
        event_time: float,
    ) -> DispatchPlan:
        from .dynamic_rescheduler import reschedule_on_event

        self._remember_route_context(
            getattr(state, "bbox", None),
            getattr(state, "scene_id", None),
        )
        setattr(state, "_ga_solver", self)
        return reschedule_on_event(state, new_orders, event_time)

    def should_replan_unfinished(self) -> bool:
        return True

    def get_active_contracts(self) -> list[Any]:
        return []

    def fulfill_contract(self, contract_id: str) -> None:
        return None

    def build_incremental_route_from_stops(
        self,
        truck: Any,
        ordered_stops: list[dict],
        current_time: float,
    ) -> Any:
        """Rebuild an execution suffix using the helper that owns the route context.

        Dynamic GA dispatch currently delegates planning to ``dynamic_greedy_helper``
        (GreedyMMCEBackboneInsertion).  The execution layer later calls this method
        through the GA solver, so using only ``greedy_helper`` can miss the road
        graph cache loaded by the dynamic helper and return ``None``.  Try the
        dynamic helper first, then the static helper as a backup.
        """
        for helper in self._incremental_route_helpers():
            self._ensure_helper_route_context(helper)
            try:
                route = helper.build_incremental_route_from_stops(
                    truck=truck,
                    ordered_stops=ordered_stops,
                    current_time=current_time,
                )
            except Exception as exc:
                logger.warning(
                    "[GA-MMCE] %s 后缀路线重建异常，尝试下一个 helper: %s",
                    helper.__class__.__name__,
                    exc,
                )
                continue
            if route is not None:
                return route
        return None

    def _remember_route_context(self, bbox: dict | None, scene_id: str | None) -> None:
        if bbox:
            self._last_route_bbox = dict(bbox)
            self._last_route_scene_id = scene_id

    def _incremental_route_helpers(self) -> list[Any]:
        helpers: list[Any] = []
        for helper in (
            getattr(self, "dynamic_greedy_helper", None),
            getattr(self, "greedy_helper", None),
        ):
            if helper is None:
                continue
            if any(helper is existing for existing in helpers):
                continue
            helpers.append(helper)
        return helpers

    def _ensure_helper_route_context(self, helper: Any) -> None:
        if not self._last_route_bbox:
            return
        load_graph = getattr(helper, "_load_road_graph", None)
        if not callable(load_graph):
            return
        try:
            load_graph(self._last_route_bbox, self._last_route_scene_id)
        except Exception as exc:
            logger.warning(
                "[GA-MMCE] %s 路网上下文同步失败: %s",
                helper.__class__.__name__,
                exc,
            )

    def solve(
        self,
        state: Any,
        warm_start: Individual | list[Individual] | None = None,
        config: GAConfig | dict[str, Any] | None = None,
        time_budget_seconds: float | None = None,
        dispatch_type: str = "ga_mmce",
    ) -> DispatchPlan:
        previous_config = self.config
        previous_evaluator_config = self.evaluator.config
        previous_decoder_config = self.decoder.config
        if config is not None or time_budget_seconds is not None:
            self.config = make_ga_config(config, base=self.config)
            if time_budget_seconds is not None:
                self.config.max_runtime_seconds = float(time_budget_seconds)
            self.evaluator.config = self.config
            self.decoder.config = self.config

        started = time.time()
        self._debug_eval_errors = {}
        self._evolution_rows = []
        self._mutation_stats = self._empty_mutation_stats()
        self._last_generation_seconds = 0.0
        self._actual_generations = 0
        self._time_budget_hit = False
        self._active_time_budget_seconds = (
            float(time_budget_seconds)
            if time_budget_seconds is not None
            else float(self.config.max_runtime_seconds)
            if self.config.max_runtime_seconds is not None
            else None
        )
        self._active_diagnostics_label = self._resolve_diagnostics_label(dispatch_type)
        self._early_stop_info = {
            "last_improvement_gen": -1,
            "no_improvement_count": 0,
            "early_stop_triggered": False,
            "early_stop_reason": "",
        }
        self._b_precheck_by_order = {}
        if dispatch_type == "full":
            self._apply_initial_runtime_drone_layout(state)
        context = build_ga_context(
            state,
            self.config,
            mode="dynamic" if "dynamic" in str(dispatch_type) or "incremental" in str(dispatch_type) else "static",
        )
        self._debug_run_start(state, context, dispatch_type)
        if dispatch_type == "full" and self._static_plan_cache_reuse_enabled():
            cached_plan = self._load_static_plan_cache(
                state=state,
                context=context,
                started=started,
                dispatch_type=dispatch_type,
            )
            if cached_plan is not None:
                self._restore_runtime_config(previous_config, previous_evaluator_config, previous_decoder_config)
                return cached_plan

        if not context.order_ids:
            plan = self._empty_plan(dispatch_type=dispatch_type)
            self._attach_solve_result(plan, None, started)
            self._debug_run_end(plan, None, context, started)
            self._restore_runtime_config(previous_config, previous_evaluator_config, previous_decoder_config)
            return plan
        if self.config.population_size <= 0:
            plan = self._empty_plan(dispatch_type=dispatch_type, reason="population_size_not_positive")
            self._attach_solve_result(plan, None, started)
            self._debug_run_end(plan, None, context, started)
            self._restore_runtime_config(previous_config, previous_evaluator_config, previous_decoder_config)
            return plan

        self._prepare_distance_context(state)

        greedy_seed = self._build_greedy_seed(state, context)
        b_seed_rendezvous_by_order = self._build_b_precheck_and_seed_data(state, context)
        warm_start_seeds = self._prepare_warm_start_seeds(
            self._normalize_warm_start(warm_start),
            state,
            context,
        )
        population = initialize_population(
            order_ids=context.order_ids,
            gene_pool=context.gene_pool,
            support_node_ids=context.support_node_ids,
            pop_size=self.config.population_size,
            greedy_seed=greedy_seed,
            warm_start=warm_start_seeds,
            use_truck_only_seed=self.config.use_truck_only_seed,
            use_obl_seed=self.config.use_obl_seed,
            allow_c_recover_station=self.config.allow_depot_drone_recover_at_station,
            use_balanced_initialization=self.config.use_balanced_initialization,
            b_seed_rendezvous_by_order=(
                b_seed_rendezvous_by_order
                if self.config.use_b_seeded_initialization
                else None
            ),
            mutation_mode_probabilities=self._mutation_mode_probabilities(),
            fixed_tail_order_ids=context.fixed_tail_order_ids,
            fixed_tail_gene_by_order=context.fixed_tail_gene_by_order,
        )
        if not population:
            plan = self._empty_plan(dispatch_type=dispatch_type, reason="population_init_failed")
            self._attach_solve_result(plan, None, started)
            self._debug_run_end(plan, None, context, started)
            self._restore_runtime_config(previous_config, previous_evaluator_config, previous_decoder_config)
            return plan

        self._evaluate_population(population, state, context)
        self._debug_population_snapshot("initial", population)

        best_seen = math.inf
        last_improvement_gen = -1
        no_improvement_count = 0
        final_population_needs_record = False
        final_generation_index = 0

        for generation in range(self.config.generations):
            if self._timeout(started):
                self._time_budget_hit = True
                break

            final_population_needs_record = False
            final_generation_index = generation
            self._actual_generations = max(self._actual_generations, generation + 1)
            generation_started = time.time()
            population.sort(key=lambda ind: ind.fitness)
            current_best = float(population[0].fitness)
            if current_best < best_seen - float(self.config.improvement_tolerance):
                best_seen = current_best
                last_improvement_gen = generation
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            self._early_stop_info.update(
                {
                    "last_improvement_gen": last_improvement_gen,
                    "no_improvement_count": no_improvement_count,
                    "early_stop_triggered": False,
                    "early_stop_reason": "",
                }
            )

            self._record_generation(generation, population, started)
            if self._should_log_generation(generation):
                self._debug_population_snapshot(f"gen={generation}", population)

            if self._should_early_stop(generation, no_improvement_count):
                self._early_stop_info.update(
                    {
                        "early_stop_triggered": True,
                        "early_stop_reason": (
                            f"no improvement for {no_improvement_count} generations "
                            f"after min_generations={self.config.min_generations}"
                        ),
                    }
                )
                if self._evolution_rows:
                    self._evolution_rows[-1]["early_stop_triggered"] = True
                    self._evolution_rows[-1]["early_stop_reason"] = self._early_stop_info["early_stop_reason"]
                self._debug_write(
                    "early_stop "
                    f"gen={generation} "
                    f"last_improvement_gen={last_improvement_gen} "
                    f"no_improvement_count={no_improvement_count} "
                    "early_stop_triggered=True "
                    f"reason={self._early_stop_info['early_stop_reason']}"
                )
                break

            new_population: list[Individual] = []
            elite_n = max(1, int(self.config.elite_ratio * self.config.population_size))
            elite_n = min(elite_n, len(population), self.config.population_size)
            new_population.extend(copy.deepcopy(population[:elite_n]))

            while len(new_population) < self.config.population_size:
                if self._timeout(started):
                    self._time_budget_hit = True
                    break
                p1 = tournament_select(population, self.config.tournament_k)
                p2 = tournament_select(population, self.config.tournament_k)

                if random.random() < self.config.crossover_rate:
                    c1, c2 = order_crossover(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

                for child in (c1, c2):
                    mutation_stats = mutate(
                        child,
                        gene_pool=context.gene_pool,
                        support_node_ids=context.support_node_ids,
                        p_seq=self.config.mutation_rate_sequence,
                        p_assign=self.config.mutation_rate_assignment,
                        p_rendezvous=self.config.mutation_rate_rendezvous,
                        allow_c_recover_station=self.config.allow_depot_drone_recover_at_station,
                        mode_probabilities=self._mutation_mode_probabilities(),
                    )
                    enforce_fixed_tail(
                        child,
                        context.fixed_tail_order_ids,
                        context.fixed_tail_gene_by_order,
                    )
                    self._accumulate_mutation_stats(mutation_stats)
                    self._evaluate_individual(child, state, context)
                    new_population.append(child)
                    if len(new_population) >= self.config.population_size:
                        break

            if self._time_budget_hit and len(new_population) < self.config.population_size:
                population.sort(key=lambda ind: ind.fitness)
                new_population.extend(copy.deepcopy(population[: self.config.population_size - len(new_population)]))
            population = new_population
            self._last_generation_seconds = time.time() - generation_started
            final_population_needs_record = True
            final_generation_index = generation + 1
            if self._time_budget_hit:
                break

        if final_population_needs_record and population:
            population.sort(key=lambda ind: ind.fitness)
            self._record_generation(final_generation_index, population, started)

        population.sort(key=lambda ind: ind.fitness)
        best = population[0]
        self.last_best_individual = copy.deepcopy(best)
        self.last_best_decode_result = getattr(best, "decoded_result", None)
        plan = best.decoded_plan or self._empty_plan(dispatch_type=dispatch_type, reason="best_has_no_plan")
        self._annotate_plan(plan, context, started, dispatch_type)
        plan.summary["b_mode_final_diag"] = self._build_b_final_summary(best)
        self._attach_solve_result(plan, best, started)
        if dispatch_type == "full":
            self._save_static_plan_cache(state=state, context=context, plan=plan, best=best)
        self._write_evolution_outputs()
        self._debug_run_end(plan, best, context, started)
        self._restore_runtime_config(previous_config, previous_evaluator_config, previous_decoder_config)
        return plan

    def _evaluate_population(self, population: list[Individual], state: Any, context: Any) -> None:
        for individual in population:
            self._evaluate_individual(individual, state, context)

    def _evaluate_individual(self, individual: Individual, state: Any, context: Any) -> Any | None:
        try:
            individual.validate_with_context(
                truck_drone_ids=context.truck_drone_ids,
                depot_drone_ids=context.depot_drone_ids,
                valid_drone_ids=context.all_drone_ids,
                support_node_ids=context.support_node_ids,
            )
            result = self.decoder.decode(individual, state, context)
            individual.fitness = compute_fitness(result, self.config)
            individual.decoded_plan = result.plan
            individual.penalties = dict(result.penalties)
            setattr(individual, "decoded_result", result)
            return result
        except Exception as exc:
            individual.fitness = float(self.config.big_m)
            individual.decoded_plan = self._empty_plan(dispatch_type="ga_mmce_invalid", reason=str(exc))
            individual.penalties = {"evaluation_exception": float(self.config.weight_infeasible)}
            setattr(individual, "decoded_result", None)
            key = str(exc)
            self._debug_eval_errors[key] = self._debug_eval_errors.get(key, 0) + 1
            if self.config.verbose:
                logger.debug("[GA-MMCE] individual evaluation failed: %s", exc, exc_info=True)
            return None

    def _build_b_precheck_and_seed_data(
        self,
        state: Any,
        context: Any,
    ) -> dict[str, tuple[str, dict[str, str]]]:
        if not self.config.b_candidate_precheck:
            return {}

        result: dict[str, tuple[str, dict[str, str]]] = {}
        self._debug_write("b_candidate_precheck_start")
        evaluation_state = clone_state_for_decode(state)
        apply_initial_drone_layout_overlay(evaluation_state, context, self.config)

        for order_id in context.order_ids:
            a_candidate = self._evaluate_initial_mode_a(evaluation_state, context, order_id)
            b_candidate = self._best_initial_mode_b(evaluation_state, context, order_id)
            c_candidate = self._best_initial_mode_c(evaluation_state, context, order_id)
            self._b_precheck_by_order[order_id] = {
                "payload": self._order_payload(state, order_id),
                "A": self._candidate_debug_dict(a_candidate),
                "B": self._candidate_debug_dict(b_candidate),
                "C": self._candidate_debug_dict(c_candidate),
            }
            if b_candidate is not None and b_candidate.feasible and b_candidate.drone_id:
                result[order_id] = (
                    f"B_{b_candidate.drone_id}",
                    {
                        "launch": b_candidate.launch_node_id,
                        "recover": b_candidate.recover_node_id,
                    },
                )
            self._debug_write(
                "candidate_precheck "
                f"order_id={order_id} "
                f"A={self._candidate_debug_dict(a_candidate)} "
                f"B={self._candidate_debug_dict(b_candidate)} "
                f"C={self._candidate_debug_dict(c_candidate)}"
            )
        self._debug_write(f"b_candidate_precheck_seeded_orders={sorted(result)}")
        return result

    def _order_payload(self, state: Any, order_id: str) -> float:
        orders = self.evaluator._mapping(state, "orders") if hasattr(self.evaluator, "_mapping") else {}
        order = orders.get(order_id) if isinstance(orders, dict) else None
        if order is None:
            return 0.0
        try:
            return float(self.evaluator._read_field(order, "payload_weight", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _evaluate_initial_mode_a(self, state: Any, context: Any, order_id: str) -> Any | None:
        if not context.truck_ids:
            return None
        try:
            return self.evaluator.evaluate_fixed_mode_a(state, order_id, context.truck_ids[0])
        except Exception as exc:
            return self._exception_candidate(order_id, "A", str(exc))

    def _best_initial_mode_b(self, state: Any, context: Any, order_id: str) -> Any | None:
        if not context.truck_ids or not context.truck_drone_ids:
            return None
        best = None
        failures: Counter = Counter()
        for drone_id in context.truck_drone_ids:
            for launch in context.support_node_ids:
                for recover in context.support_node_ids:
                    try:
                        candidate = self.evaluator.evaluate_fixed_mode_b(
                            state,
                            order_id,
                            context.truck_ids[0],
                            drone_id,
                            launch,
                            recover,
                        )
                    except Exception as exc:
                        candidate = self._exception_candidate(order_id, "B", str(exc))
                    if candidate.feasible:
                        if best is None or float(candidate.score_total) < float(best.score_total):
                            best = candidate
                    else:
                        failures[candidate.reason or "infeasible"] += 1
        if best is not None:
            return best
        if failures:
            reason = failures.most_common(1)[0][0]
            candidate = self._exception_candidate(order_id, "B", reason)
            candidate.reason = reason
            return candidate
        return None

    def _best_initial_mode_c(self, state: Any, context: Any, order_id: str) -> Any | None:
        best = None
        failures: Counter = Counter()
        for drone_id in context.depot_drone_ids:
            for recover in context.support_node_ids:
                try:
                    candidate = self.evaluator.evaluate_fixed_mode_c(state, order_id, drone_id, recover)
                except Exception as exc:
                    candidate = self._exception_candidate(order_id, "C", str(exc))
                if candidate.feasible:
                    if best is None or float(candidate.score_total) < float(best.score_total):
                        best = candidate
                else:
                    failures[candidate.reason or "infeasible"] += 1
        if best is not None:
            return best
        if failures:
            reason = failures.most_common(1)[0][0]
            candidate = self._exception_candidate(order_id, "C", reason)
            candidate.reason = reason
            return candidate
        return None

    def _exception_candidate(self, order_id: str, mode: str, reason: str) -> Any:
        from .physical_evaluator import GACandidate

        return GACandidate(order_id=order_id, mode=mode, feasible=False, reason=reason or "evaluation_exception")

    def _candidate_debug_dict(self, candidate: Any | None) -> dict[str, Any]:
        if candidate is None:
            return {"feasible": False, "reason": "not_available"}
        return {
            "feasible": bool(candidate.feasible),
            "reason": candidate.reason,
            "drone_id": candidate.drone_id,
            "launch": candidate.launch_node_id,
            "recover": candidate.recover_node_id,
            "truck_dist": float(candidate.truck_distance or 0.0),
            "uav_dist": float(candidate.uav_distance or 0.0),
            "energy_cost": float(candidate.cost_energy or 0.0),
            "completion_time": float(candidate.completion_time or 0.0),
            "time_cost": 0.0,
            "sync_waiting_cost": float(candidate.waiting_time or 0.0) * float(self.config.weight_waiting),
            "penalty_cost": float(candidate.cost_penalty or 0.0),
            "mode_reward": float(getattr(candidate, "mode_reward", 0.0) or 0.0),
            "closure_cost": 0.0,
            "repair_penalty": 0.0,
            "station_queue_penalty": 0.0,
            "total_score": float(candidate.score_total) if math.isfinite(float(candidate.score_total)) else math.inf,
        }

    def _build_greedy_seed(self, state: Any, context: Any) -> Individual | None:
        if not self.config.use_greedy_seed:
            return None

        greedy_plan = self._make_greedy_seed_plan(state)
        if greedy_plan is None:
            return None

        try:
            return greedy_plan_to_individual(
                greedy_plan,
                context.order_ids,
                context.gene_pool,
                context.support_node_ids,
                allow_c_recover_station=self.config.allow_depot_drone_recover_at_station,
            )
        except Exception as exc:
            logger.warning("[GA-MMCE] greedy seed 转换失败，跳过 seed: %s", exc)
            return None

    def _make_greedy_seed_plan(self, state: Any) -> Any | None:
        bbox = getattr(state, "bbox", None)
        orders = getattr(state, "orders", None)
        if not bbox or not orders:
            return None

        try:
            return self.greedy_helper.dispatch_replan_current_state(
                replan_orders=orders,
                current_time=float(getattr(state, "current_time", 0.0) or 0.0),
                bbox=bbox,
                scene_id=getattr(state, "scene_id", None),
            )
        except Exception as exc:
            logger.warning("[GA-MMCE] greedy seed 生成失败，跳过 seed: %s", exc)
            return None

    def _prepare_distance_context(self, state: Any) -> None:
        bbox = getattr(state, "bbox", None)
        if not bbox:
            return

        try:
            self.greedy_helper._road_distance_memo.clear()
            self.greedy_helper._uav_path_distance_memo.clear()
            self.greedy_helper._activate_path_planner(getattr(state, "scene_id", None))
            self.greedy_helper._load_road_graph(bbox, getattr(state, "scene_id", None))
        except Exception as exc:
            logger.warning("[GA-MMCE] OSM 路网预加载失败，Decoder 将 fallback 到直接距离: %s", exc)

    def _normalize_warm_start(
        self,
        warm_start: Individual | list[Individual] | None,
    ) -> list[Individual]:
        if not bool(getattr(self.config, "use_warm_start", True)):
            return []
        if warm_start is None:
            return [self.last_best_individual] if self.last_best_individual is not None else []
        if isinstance(warm_start, list):
            return [seed for seed in warm_start if seed is not None]
        return [warm_start]

    def _prepare_warm_start_seeds(
        self,
        warm_start: list[Individual],
        state: Any,
        context: Any,
    ) -> list[Individual]:
        if not warm_start:
            return []

        ranked: list[Individual] = []
        for seed in warm_start:
            copied = copy.deepcopy(seed)
            try:
                enforce_fixed_tail(
                    copied,
                    context.fixed_tail_order_ids,
                    context.fixed_tail_gene_by_order,
                )
                if set(copied.sequence) != set(context.order_ids):
                    continue
                if any(gene not in set(context.gene_pool) for gene in copied.assignment):
                    continue
                self._evaluate_individual(copied, state, context)
                ranked.append(copied)
            except Exception:
                continue
        ranked.sort(key=lambda ind: float(getattr(ind, "fitness", float("inf"))))
        return ranked

    def _timeout(self, started: float) -> bool:
        budget = self._active_time_budget_seconds
        if budget is None:
            return False
        return time.time() - started >= budget

    def _restore_runtime_config(
        self,
        config: GAConfig,
        evaluator_config: GAConfig,
        decoder_config: GAConfig,
    ) -> None:
        self.config = config
        self.evaluator.config = evaluator_config
        self.decoder.config = decoder_config
        self._active_time_budget_seconds = None

    def _static_plan_cache_reuse_enabled(self) -> bool:
        if self._reuse_static_plan_cache_override is not None:
            return self._reuse_static_plan_cache_override
        return self._truthy_env(os.environ.get(STATIC_PLAN_CACHE_ENV))

    @staticmethod
    def _truthy_env(value: Any) -> bool:
        return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    def _static_plan_cache_path(self) -> Path:
        raw = os.environ.get(STATIC_PLAN_CACHE_PATH_ENV, "logs/ga_static_plan_cache.pkl")
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def _load_static_plan_cache(
        self,
        *,
        state: Any,
        context: Any,
        started: float,
        dispatch_type: str,
    ) -> DispatchPlan | None:
        path = self._static_plan_cache_path()
        if not path.exists():
            self._debug_write(f"static_plan_cache miss reason=file_not_found path={path}")
            return None

        expected_signature = self._static_plan_cache_signature(state, context)
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:
            logger.warning("[GA-MMCE] 静态计划缓存读取失败: %s", exc)
            self._debug_write(f"static_plan_cache miss reason=load_failed error={exc}")
            return None

        if not isinstance(payload, dict) or payload.get("schema") != STATIC_PLAN_CACHE_SCHEMA:
            self._debug_write("static_plan_cache miss reason=schema_mismatch")
            return None
        cached_signature = payload.get("signature")
        signature_relaxed = False
        if self._normalized_static_plan_cache_signature(cached_signature) != expected_signature:
            mismatch = self._static_plan_cache_mismatch_reasons(
                cached_signature,
                expected_signature,
            )
            if self._allow_explicit_static_plan_cache_reuse(mismatch):
                signature_relaxed = True
                self._debug_write(
                    "static_plan_cache relaxed_hit "
                    f"ignored_signature_details={mismatch}"
                )
            else:
                self._debug_write(
                    "static_plan_cache miss reason=signature_mismatch "
                    f"details={mismatch}"
                )
                return None

        cached_plan = payload.get("plan")
        if cached_plan is None:
            self._debug_write(
                "static_plan_cache miss reason=missing_plan"
            )
            return None

        cached_best = copy.deepcopy(payload.get("best_individual"))
        plan = copy.deepcopy(cached_plan)
        self.last_best_individual = cached_best
        self.last_best_decode_result = (
            getattr(cached_best, "decoded_result", None)
            if cached_best is not None
            else None
        )
        if self.last_best_decode_result is None:
            self.last_best_decode_result = SimpleNamespace(plan=plan)

        self._refresh_cached_plan_drone_routes(plan, state)
        self._actual_generations = 0
        self._annotate_plan(plan, context, started, dispatch_type)
        plan.summary["static_cache_reused"] = True
        plan.summary["static_cache_path"] = str(path)
        plan.summary["static_cache_saved_at"] = payload.get("saved_at")
        if signature_relaxed:
            plan.summary["static_cache_signature_relaxed"] = True
        self._attach_solve_result(plan, cached_best, started)
        self._debug_write(
            "static_plan_cache hit "
            f"path={path} "
            f"saved_at={payload.get('saved_at')} "
            f"orders={len(context.order_ids)} "
            f"signature_relaxed={signature_relaxed}"
        )
        logger.info("[GA-MMCE] 复用静态计划缓存: %s", path)
        self._debug_run_end(plan, cached_best, context, started)
        return plan

    def _refresh_cached_plan_drone_routes(self, plan: DispatchPlan, state: Any) -> None:
        """Refresh cached drone route geometry with the current UAV obstacle planner."""
        routes = getattr(plan, "drone_routes", None)
        if not routes:
            return

        try:
            self.greedy_helper._activate_path_planner(getattr(state, "scene_id", None))
        except Exception as exc:
            self._debug_write(f"static_plan_cache drone_route_refresh skipped reason=planner_activate_failed error={exc}")
            return

        planner = getattr(self.greedy_helper, "_path_planner", None)
        if planner is None:
            self._debug_write("static_plan_cache drone_route_refresh skipped reason=no_path_planner")
            return

        altitude = float(getattr(self.greedy_helper, "UAV_CRUISE_ALTITUDE_M", 80.0) or 80.0)
        refreshed = 0
        for route in routes.values():
            path = list(getattr(route, "path", []) or [])
            launch_loc = getattr(route, "launch_loc", None) or (path[0] if path else None)
            delivery_loc = getattr(route, "delivery_loc", None) or (path[1] if len(path) > 1 else None)
            recovery_loc = getattr(route, "recovery_loc", None) or (path[-1] if path else None)
            if launch_loc is None or delivery_loc is None or recovery_loc is None:
                continue

            outbound = list(planner.plan(launch_loc, delivery_loc, altitude) or [])
            inbound = list(planner.plan(delivery_loc, recovery_loc, altitude) or [])
            if not outbound:
                outbound = [launch_loc, delivery_loc]
            if not inbound:
                inbound = [delivery_loc, recovery_loc]
            route.path = outbound + inbound[1:]
            route.launch_loc = launch_loc
            route.delivery_loc = delivery_loc
            route.recovery_loc = recovery_loc
            refreshed += 1

        self._debug_write(f"static_plan_cache drone_route_refresh refreshed={refreshed}")

    def _save_static_plan_cache(
        self,
        *,
        state: Any,
        context: Any,
        plan: DispatchPlan,
        best: Individual | None,
    ) -> None:
        if not getattr(plan, "allocations", None):
            return

        path = self._static_plan_cache_path()
        payload = {
            "schema": STATIC_PLAN_CACHE_SCHEMA,
            "saved_at": time.time(),
            "signature": self._static_plan_cache_signature(state, context),
            "plan": copy.deepcopy(plan),
            "best_individual": copy.deepcopy(best),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            self._debug_write(f"static_plan_cache saved path={path}")
        except Exception as exc:
            logger.warning("[GA-MMCE] 静态计划缓存写入失败: %s", exc)
            self._debug_write(f"static_plan_cache save_failed path={path} error={exc}")

    def _static_plan_cache_signature(self, state: Any, context: Any) -> dict[str, Any]:
        mgr = getattr(state, "entity_mgr", None) or self.entity_mgr
        orders = getattr(state, "orders", {}) or {}

        def order_by_id(order_id: str) -> Any:
            if isinstance(orders, dict):
                return orders.get(order_id)
            return None

        order_sig = []
        for order_id in context.order_ids:
            order = order_by_id(order_id)
            order_sig.append(
                {
                    "id": str(order_id),
                    "payload_weight": self._rounded_number(self._field(order, "payload_weight", 0.0)),
                    "deadline": self._rounded_number(self._field(order, "deadline", 0.0)),
                    "delivery": self._loc_signature(self._field(order, "delivery_loc", None)),
                }
            )

        return {
            "scene_id": str(getattr(state, "scene_id", "") or ""),
            "orders": order_sig,
            "depots": self._host_signature(getattr(mgr, "depots", {}) or {}, context.depot_ids),
            "stations": self._host_signature(getattr(mgr, "stations", {}) or {}, context.station_ids),
            "trucks": self._truck_signature(getattr(mgr, "trucks", {}) or {}, context.truck_ids),
            "truck_drone_ids": tuple(str(v) for v in context.truck_drone_ids),
            "depot_drone_ids": tuple(str(v) for v in context.depot_drone_ids),
            "all_drone_ids": tuple(str(v) for v in context.all_drone_ids),
            "support_node_ids": tuple(str(v) for v in context.support_node_ids),
            "future_dynamic_orders": self._future_dynamic_cache_signature(state),
        }

    def _future_dynamic_cache_signature(self, state: Any) -> tuple[tuple[str, float, float, float, float], ...]:
        order_mgr = getattr(state, "order_mgr", None) or self._order_mgr
        entries = getattr(order_mgr, "_scheduled_dynamic", []) if order_mgr is not None else []
        cursor = int(getattr(order_mgr, "_scheduled_dynamic_i", 0) or 0) if order_mgr is not None else 0
        rows = []
        for entry in list(entries)[max(0, cursor):]:
            if not isinstance(entry, dict):
                continue
            rows.append(
                (
                    str(entry.get("order_id", "")),
                    self._rounded_number(entry.get("spawn_sim_s", 0.0)),
                    self._rounded_number(entry.get("deadline_offset_s", entry.get("deadline_sim_s", 0.0))),
                    self._rounded_number(entry.get("delivery_lng", 0.0)),
                    self._rounded_number(entry.get("delivery_lat", 0.0)),
                )
            )
        return tuple(rows)

    def _normalized_static_plan_cache_signature(self, signature: Any) -> Any:
        if not isinstance(signature, dict):
            return signature
        normalized = dict(signature)
        # Older caches included the current map viewport. That value changes
        # with pan/zoom and does not affect an already decoded static plan.
        normalized.pop("bbox", None)
        return normalized

    def _static_plan_cache_mismatch_reasons(
        self,
        cached_signature: Any,
        expected_signature: dict[str, Any],
    ) -> list[str]:
        cached = self._normalized_static_plan_cache_signature(cached_signature)
        if not isinstance(cached, dict):
            return ["invalid_cached_signature"]
        reasons: list[str] = []
        for key, expected_value in expected_signature.items():
            if cached.get(key) != expected_value:
                reasons.append(str(key))
        for key in cached:
            if key not in expected_signature:
                reasons.append(f"extra:{key}")
        return reasons

    def _allow_explicit_static_plan_cache_reuse(self, mismatch: list[str]) -> bool:
        return self._reuse_static_plan_cache_override is True

    def _host_signature(self, hosts: dict[str, Any], host_ids: list[str]) -> tuple[tuple[str, tuple[float, float, float]], ...]:
        rows = []
        for host_id in host_ids:
            host = hosts.get(host_id)
            rows.append((str(host_id), self._loc_signature(getattr(host, "location", None))))
        return tuple(rows)

    def _truck_signature(self, trucks: dict[str, Any], truck_ids: list[str]) -> tuple[tuple[str, tuple[float, float, float], float], ...]:
        rows = []
        for truck_id in truck_ids:
            truck = trucks.get(truck_id)
            rows.append(
                (
                    str(truck_id),
                    self._loc_signature(getattr(truck, "current_loc", None)),
                    self._rounded_number(getattr(truck, "speed", 0.0)),
                )
            )
        return tuple(rows)

    def _loc_signature(self, loc: Any) -> tuple[float, float, float]:
        if loc is None:
            return (0.0, 0.0, 0.0)
        return (
            self._rounded_number(getattr(loc, "x", 0.0)),
            self._rounded_number(getattr(loc, "y", 0.0)),
            self._rounded_number(getattr(loc, "z", 0.0)),
        )

    def _field(self, record: Any, field_name: str, default: Any = None) -> Any:
        return self.evaluator._read_field(record, field_name, default)  # noqa: SLF001 - shared GA adapter helper.

    @staticmethod
    def _rounded_number(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(number):
            return 0.0
        return round(number, 6)

    def _resolve_diagnostics_label(self, dispatch_type: str) -> str:
        configured = str(getattr(self.config, "diagnostics_label", "") or "").strip().lower()
        if configured in {"static", "dynamic"}:
            return configured
        return "dynamic" if "dynamic" in str(dispatch_type) or "incremental" in str(dispatch_type) else "static"

    def _mutation_mode_probabilities(self) -> dict[str, float]:
        return {
            "A": float(self.config.mutation_mode_prob_a),
            "B": float(self.config.mutation_mode_prob_b),
            "C": float(self.config.mutation_mode_prob_c),
        }

    def _should_early_stop(self, generation: int, no_improvement_count: int) -> bool:
        patience = int(self.config.early_stopping_patience or 0)
        if patience <= 0:
            return False
        return (
            generation >= int(self.config.min_generations)
            and no_improvement_count >= patience
        )

    def _apply_initial_runtime_drone_layout(self, state: Any) -> None:
        """Synchronize the configured 9/3 initial drone layout to runtime entities."""
        if not bool(getattr(self.config, "initial_drone_layout_enabled", False)):
            return
        current_time = float(getattr(state, "current_time", 0.0) or 0.0)
        max_time = float(getattr(self.config, "initial_drone_layout_max_time_s", 0.0) or 0.0)
        if current_time > max_time:
            return

        trucks = getattr(self.entity_mgr, "trucks", {}) or {}
        depots = getattr(self.entity_mgr, "depots", {}) or {}
        drones = getattr(self.entity_mgr, "drones", {}) or {}
        if not trucks or not depots or not drones:
            return

        truck_id = next(iter(trucks.keys()))
        depot_id = next(iter(depots.keys()))
        truck = trucks.get(truck_id)
        depot = depots.get(depot_id)
        if truck is None or depot is None:
            return

        truck_drone_ids = self._existing_drone_ids(getattr(self.config, "initial_truck_drone_ids", ()), drones)
        depot_drone_ids = self._existing_drone_ids(getattr(self.config, "initial_depot_drone_ids", ()), drones)
        if not truck_drone_ids and not depot_drone_ids:
            return

        try:
            from core.entities.primitives import SourceType
        except Exception:
            SourceType = None

        truck_pool = set(truck_drone_ids)
        depot_pool = set(depot_drone_ids)
        managed_pool = truck_pool | depot_pool

        if int(getattr(truck, "parking_slots", 0) or 0) < len(truck_drone_ids):
            truck.parking_slots = len(truck_drone_ids)

        for other_truck in trucks.values():
            existing = [did for did in getattr(other_truck, "docked_drones", []) if did not in managed_pool]
            setattr(other_truck, "docked_drones", existing)

        for item in depots.values():
            idle = [did for did in getattr(item, "idle_drones", []) if did not in truck_pool]
            if item is depot:
                idle.extend(did for did in depot_drone_ids if did not in idle)
            else:
                idle = [did for did in idle if did not in depot_pool]
            setattr(item, "idle_drones", list(dict.fromkeys(idle)))

        truck.docked_drones = list(truck_drone_ids)
        truck_loc = truck.get_location(current_time) if hasattr(truck, "get_location") else getattr(truck, "current_loc", None)
        depot_loc = getattr(depot, "location", None)

        for drone_id in truck_drone_ids:
            drone = drones.get(drone_id)
            if drone is None:
                continue
            if SourceType is not None:
                drone.home_type = SourceType.TRUCK
            drone.home_id = truck_id
            drone.transport_truck_id = truck_id
            drone.waiting_recovery_station_id = ""
            if truck_loc is not None and not self._drone_is_flying(drone):
                drone.current_loc = truck_loc

        for drone_id in depot_drone_ids:
            drone = drones.get(drone_id)
            if drone is None:
                continue
            if SourceType is not None:
                drone.home_type = SourceType.DEPOT
            drone.home_id = depot_id
            drone.transport_truck_id = None
            drone.waiting_recovery_station_id = ""
            if depot_loc is not None and not self._drone_is_flying(drone):
                drone.current_loc = depot_loc

        logger.info(
            "[GA-MMCE] 初始运行态无人机装载已同步: truck=%s drones=%s depot=%s drones=%s",
            truck_id,
            truck_drone_ids,
            depot_id,
            depot_drone_ids,
        )

    def _existing_drone_ids(self, configured_ids: Any, drones: dict[str, Any]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for drone_id in configured_ids or ():
            normalized = str(drone_id).strip()
            if normalized and normalized in drones and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    def _drone_is_flying(self, drone: Any) -> bool:
        status = getattr(drone, "status", None)
        is_flying = getattr(status, "is_flying", None)
        if is_flying is not None:
            return bool(is_flying)
        name = getattr(status, "value", status)
        return str(name).upper() in {"FLYING", "FLYING_TO_PICKUP", "DELIVERING", "RETURNING_TO_DEPOT"}

    def _should_log_generation(self, generation: int) -> bool:
        interval = max(1, int(self.config.log_interval or 1))
        return (
            generation == 0
            or generation == self.config.generations - 1
            or generation % interval == 0
        )

    def _record_generation(self, generation: int, population: list[Individual], started: float) -> None:
        if not self.config.diagnostics_enabled:
            self._log_generation(generation, population)
            return
        row = self._build_generation_row(generation, population, started)
        self._evolution_rows.append(row)
        self._log_generation_details(row)

    def _build_generation_row(
        self,
        generation: int,
        population: list[Individual],
        started: float,
    ) -> dict[str, Any]:
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        fitnesses = [float(ind.fitness) for ind in sorted_pop]
        best = sorted_pop[0]
        best_result = getattr(best, "decoded_result", None)
        mode_counts = self._population_mode_counts(sorted_pop)
        individual_counts = self._population_individual_mode_counts(sorted_pop)
        best_mode_counts = Counter(self._gene_mode(gene) for gene in best.assignment)
        aggregate_decode = self._aggregate_decode_diagnostics(sorted_pop)
        cost_breakdown = getattr(best_result, "cost_breakdown", {}) if best_result is not None else {}
        penalties = getattr(best_result, "penalties", {}) if best_result is not None else {}
        row = {
            "gen": generation,
            "best": fitnesses[0],
            "avg": sum(fitnesses) / len(fitnesses),
            "min_fitness": fitnesses[0],
            "median": self._median(fitnesses),
            "worst": fitnesses[-1],
            "feasible_count": sum(1 for ind in sorted_pop if not ind.penalties),
            "hard_feasible_count": sum(
                1
                for ind in sorted_pop
                if getattr(getattr(ind, "decoded_result", None), "feasible", False)
                and not getattr(getattr(ind, "decoded_result", None), "penalties", {})
            ),
            "soft_penalty_count": sum(1 for ind in sorted_pop if bool(ind.penalties)),
            "population_size": len(sorted_pop),
            "A_count": mode_counts.get("A", 0),
            "B_count": mode_counts.get("B", 0),
            "C_count": mode_counts.get("C", 0),
            "individuals_with_B": individual_counts["with_B"],
            "individuals_with_C": individual_counts["with_C"],
            "individuals_all_A": individual_counts["all_A"],
            "b_gene_count_in_population": mode_counts.get("B", 0),
            "b_individual_count": individual_counts["with_B"],
            "b_success": aggregate_decode["b_success"],
            "b_decoded_success_count": aggregate_decode["b_success"],
            "b_candidate_accepted_count": aggregate_decode["b_success"],
            "b_infeasible": aggregate_decode["b_infeasible"],
            "b_repaired": aggregate_decode["b_repaired"],
            "b_failure_reasons": aggregate_decode["b_failure_reasons"],
            "best_B_candidate_score": aggregate_decode["best_B_candidate_score"],
            "avg_B_candidate_score": aggregate_decode["avg_B_candidate_score"],
            "orders_where_B_feasible": aggregate_decode["orders_where_B_feasible"],
            "c_gene_count_in_population": mode_counts.get("C", 0),
            "c_success": aggregate_decode["c_success"],
            "c_decoded_success_count": aggregate_decode["c_success"],
            "c_candidate_accepted_count": aggregate_decode["c_success"],
            "c_infeasible": aggregate_decode["c_infeasible"],
            "c_repaired": aggregate_decode["c_repaired"],
            "c_failure_reasons": aggregate_decode["c_failure_reasons"],
            "best_A_count": best_mode_counts.get("A", 0),
            "best_B_count": best_mode_counts.get("B", 0),
            "best_C_count": best_mode_counts.get("C", 0),
            "truck_distance": cost_breakdown.get("raw_truck_distance", 0.0),
            "uav_distance": cost_breakdown.get("raw_uav_distance", 0.0),
            "energy": cost_breakdown.get("energy_cost", 0.0),
            "penalty": sum(float(value) for value in penalties.values()) if isinstance(penalties, dict) else 0.0,
            "repair_penalty": cost_breakdown.get("repair_penalty", 0.0),
            "station_queue_penalty": cost_breakdown.get("station_queue_penalty", 0.0),
            "infeasible_penalty": cost_breakdown.get("infeasible_penalty", 0.0),
            "unserved_penalty": cost_breakdown.get("unserved_penalty", 0.0),
            "elapsed": time.time() - started,
            "seconds_per_generation": self._last_generation_seconds,
            "last_improvement_gen": self._early_stop_info.get("last_improvement_gen", -1),
            "no_improvement_count": self._early_stop_info.get("no_improvement_count", 0),
            "early_stop_triggered": self._early_stop_info.get("early_stop_triggered", False),
            "early_stop_reason": self._early_stop_info.get("early_stop_reason", ""),
            "mutation_b_added": self._mutation_stats["b_added"],
            "mutation_b_removed": self._mutation_stats["b_removed"],
        }
        if best_result is not None:
            for key, value in cost_breakdown.items():
                row[key] = value
        return row

    def _log_generation_details(self, row: dict[str, Any]) -> None:
        if not self._should_log_generation(int(row["gen"])):
            return
        self._debug_write(
            "ga_evolution "
            f"gen={row['gen']} best={float(row['best']):.3f} "
            f"avg={float(row['avg']):.3f} median={float(row['median']):.3f} "
            f"worst={float(row['worst']):.3f} "
            f"feasible={row['feasible_count']}/{row['population_size']} "
            f"hard_feasible={row['hard_feasible_count']} "
            f"soft_penalty={row['soft_penalty_count']} "
            f"elapsed={float(row['elapsed']):.3f}s "
            f"sec_per_gen={float(row['seconds_per_generation']):.3f}"
        )
        self._debug_write(
            "ga_modes "
            f"gen={row['gen']} "
            f"population_gene_mode_counts={{'A': {row['A_count']}, 'B': {row['B_count']}, 'C': {row['C_count']}}} "
            f"population_individual_mode_counts={{'individuals_with_B': {row['individuals_with_B']}, "
            f"'individuals_with_C': {row['individuals_with_C']}, 'individuals_all_A': {row['individuals_all_A']}}} "
            f"best_individual_modes={{'A': {row['best_A_count']}, 'B': {row['best_B_count']}, 'C': {row['best_C_count']}}}"
        )
        self._debug_write(
            "ga_b_diagnostics "
            f"gen={row['gen']} b_gene_count_in_population={row['b_gene_count_in_population']} "
            f"b_individual_count={row['b_individual_count']} "
            f"b_decoded_success_count={row['b_decoded_success_count']} "
            f"b_candidate_accepted_count={row['b_candidate_accepted_count']} "
            f"b_repaired_count={row['b_repaired']} "
            f"b_infeasible_count={row['b_infeasible']} "
            f"b_failure_reasons={row['b_failure_reasons']} "
            f"best_B_candidate_score={row['best_B_candidate_score']} "
            f"avg_B_candidate_score={row['avg_B_candidate_score']} "
            f"orders_where_B_feasible={row['orders_where_B_feasible']}"
        )
        self._debug_write(
            "ga_c_diagnostics "
            f"gen={row['gen']} c_gene_count_in_population={row['c_gene_count_in_population']} "
            f"c_decoded_success_count={row['c_decoded_success_count']} "
            f"c_candidate_accepted_count={row['c_candidate_accepted_count']} "
            f"c_repaired_count={row['c_repaired']} "
            f"c_infeasible_count={row['c_infeasible']} "
            f"c_failure_reasons={row['c_failure_reasons']}"
        )
        self._debug_write(
            "ga_cost_breakdown "
            f"gen={row['gen']} "
            f"raw_truck_distance={row.get('raw_truck_distance', 0.0)} "
            f"raw_uav_distance={row.get('raw_uav_distance', 0.0)} "
            f"truck_distance_cost={row.get('truck_distance_cost', 0.0)} "
            f"uav_distance_cost={row.get('uav_distance_cost', 0.0)} "
            f"energy_cost={row.get('energy_cost', 0.0)} "
            f"plan_completion_time={row.get('plan_completion_time', 0.0)} "
            f"time_cost={row.get('time_cost', 0.0)} "
            f"waiting_cost={row.get('waiting_cost', 0.0)} "
            f"air_ground_reward={row.get('air_ground_reward', 0.0)} "
            f"closure_route_cost={row.get('closure_route_cost', 0.0)} "
            f"repair_penalty={row.get('repair_penalty', 0.0)} "
            f"station_queue_penalty={row.get('station_queue_penalty', 0.0)} "
            f"unserved_penalty={row.get('unserved_penalty', 0.0)} "
            f"total_fitness={row.get('total_fitness', row['best'])}"
        )
        self._debug_write(
            "ga_early_stop "
            f"gen={row['gen']} last_improvement_gen={row['last_improvement_gen']} "
            f"no_improvement_count={row['no_improvement_count']} "
            f"early_stop_triggered={row['early_stop_triggered']} "
            f"early_stop_reason={row['early_stop_reason']}"
        )

    def _write_evolution_outputs(self) -> None:
        if not self.config.diagnostics_enabled:
            return
        log_dir = self._diagnostics_dir()
        label = self._active_diagnostics_label or "static"
        try:
            if self.config.save_evolution_csv and self._evolution_rows:
                write_evolution_csv(self._evolution_rows, log_dir / f"ga_evolution_{label}.csv")
            if self.config.save_evolution_csv and self._b_precheck_by_order:
                write_mode_precheck_csv(self._b_precheck_by_order, log_dir / f"ga_mode_precheck_{label}.csv")
            if self.config.save_evolution_plots:
                plot_dir = log_dir if label == "static" else log_dir / label
                write_evolution_plots(self._evolution_rows, plot_dir)
        except Exception as exc:
            self._debug_write(f"diagnostics_output_failed reason={exc}")

    def _diagnostics_dir(self) -> Path:
        configured = Path(str(self.config.diagnostics_dir or "logs"))
        return configured if configured.is_absolute() else PROJECT_ROOT / configured

    def _population_mode_counts(self, population: list[Individual]) -> Counter:
        counts: Counter = Counter()
        for ind in population:
            counts.update(self._gene_mode(gene) for gene in ind.assignment)
        return counts

    def _population_individual_mode_counts(self, population: list[Individual]) -> dict[str, int]:
        result = {"with_B": 0, "with_C": 0, "all_A": 0}
        for ind in population:
            modes = {self._gene_mode(gene) for gene in ind.assignment}
            if "B" in modes:
                result["with_B"] += 1
            if "C" in modes:
                result["with_C"] += 1
            if modes == {"A"}:
                result["all_A"] += 1
        return result

    def _aggregate_decode_diagnostics(self, population: list[Individual]) -> dict[str, Any]:
        aggregate = {
            "b_success": 0,
            "b_infeasible": 0,
            "b_repaired": 0,
            "b_failure_reasons": {},
            "best_B_candidate_score": math.inf,
            "avg_B_candidate_score": math.inf,
            "orders_where_B_feasible": [],
            "c_success": 0,
            "c_infeasible": 0,
            "c_repaired": 0,
            "c_failure_reasons": {},
        }
        b_scores: list[float] = []
        b_orders: set[str] = set()
        for ind in population:
            result = getattr(ind, "decoded_result", None)
            diagnostics = getattr(result, "diagnostics", {}) if result is not None else {}
            aggregate["b_success"] += int(diagnostics.get("b_decoded_success_count", 0) or 0)
            aggregate["b_infeasible"] += int(diagnostics.get("b_infeasible_count", 0) or 0)
            aggregate["b_repaired"] += int(diagnostics.get("b_repaired_count", 0) or 0)
            aggregate["c_success"] += int(diagnostics.get("c_decoded_success_count", 0) or 0)
            aggregate["c_infeasible"] += int(diagnostics.get("c_infeasible_count", 0) or 0)
            aggregate["c_repaired"] += int(diagnostics.get("c_repaired_count", 0) or 0)
            self._merge_counts(aggregate["b_failure_reasons"], diagnostics.get("b_failure_reasons", {}))
            self._merge_counts(aggregate["c_failure_reasons"], diagnostics.get("c_failure_reasons", {}))
            score = float(diagnostics.get("best_B_candidate_score", math.inf) or math.inf)
            if math.isfinite(score):
                b_scores.append(score)
            for order_id in diagnostics.get("orders_where_B_feasible", []) or []:
                b_orders.add(str(order_id))
        if b_scores:
            aggregate["best_B_candidate_score"] = min(b_scores)
            aggregate["avg_B_candidate_score"] = sum(b_scores) / len(b_scores)
        aggregate["orders_where_B_feasible"] = sorted(b_orders)
        return aggregate

    def _merge_counts(self, target: dict[str, int], source: Any) -> None:
        if not isinstance(source, dict):
            return
        for key, value in source.items():
            target[str(key)] = target.get(str(key), 0) + int(value)

    def _median(self, values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    def _gene_mode(self, gene: str) -> str:
        if gene == "A":
            return "A"
        if gene.startswith("B_"):
            return "B"
        if gene.startswith("C_"):
            return "C"
        return "?"

    def _empty_mutation_stats(self) -> dict[str, Any]:
        return {"assignment_mutations": 0, "b_added": 0, "b_removed": 0, "by_transition": {}}

    def _accumulate_mutation_stats(self, mutation_stats: Any) -> None:
        self._mutation_stats["assignment_mutations"] += int(getattr(mutation_stats, "assignment_mutations", 0) or 0)
        self._mutation_stats["b_added"] += int(getattr(mutation_stats, "b_added", 0) or 0)
        self._mutation_stats["b_removed"] += int(getattr(mutation_stats, "b_removed", 0) or 0)
        for key, value in getattr(mutation_stats, "by_transition", {}).items():
            bucket = self._mutation_stats["by_transition"]
            bucket[key] = bucket.get(key, 0) + int(value)

    def _build_b_final_summary(self, best: Individual | None) -> dict[str, Any]:
        best_mode_counts = Counter(self._gene_mode(gene) for gene in best.assignment) if best is not None else Counter()
        b_failure_reasons: dict[str, int] = {}
        total_b_gene_count = 0
        total_b_success = 0
        total_b_repaired = 0
        total_b_infeasible = 0

        for row in self._evolution_rows:
            total_b_gene_count += int(row.get("B_count", 0) or 0)
            total_b_success += int(row.get("b_success", 0) or 0)
            total_b_repaired += int(row.get("b_repaired", 0) or 0)
            total_b_infeasible += int(row.get("b_infeasible", 0) or 0)
            self._merge_counts(b_failure_reasons, row.get("b_failure_reasons", {}))

        feasible_orders = [
            order_id
            for order_id, row in sorted(self._b_precheck_by_order.items())
            if bool((row.get("B") or {}).get("feasible", False))
        ]
        main_failure_reason = self._main_failure_reason(b_failure_reasons)
        return {
            "b_entered_population": total_b_gene_count > 0,
            "b_decode_success_total": total_b_success,
            "b_decode_success": total_b_success > 0,
            "b_repaired_total": total_b_repaired,
            "b_infeasible_total": total_b_infeasible,
            "b_selected_in_best": int(best_mode_counts.get("B", 0)),
            "b_failure_reasons": b_failure_reasons,
            "main_failure_reason": main_failure_reason,
            "b_feasible_orders": feasible_orders,
            "b_feasible_order_costs": self._b_feasible_order_costs(feasible_orders),
            "b_not_selected_reason": self._explain_b_not_selected(
                selected_count=int(best_mode_counts.get("B", 0)),
                entered=total_b_gene_count > 0,
                decoded=total_b_success > 0,
                feasible_orders=feasible_orders,
                main_failure_reason=main_failure_reason,
            ),
        }

    def _main_failure_reason(self, reasons: dict[str, int]) -> str:
        if not reasons:
            return ""
        return max(reasons.items(), key=lambda item: item[1])[0]

    def _b_feasible_order_costs(self, order_ids: list[str]) -> dict[str, dict[str, Any]]:
        costs: dict[str, dict[str, Any]] = {}
        for order_id in order_ids:
            row = self._b_precheck_by_order.get(order_id, {})
            costs[order_id] = {
                "A_score": (row.get("A") or {}).get("total_score"),
                "B_score": (row.get("B") or {}).get("total_score"),
                "C_score": (row.get("C") or {}).get("total_score"),
                "best_local_mode": self._best_local_mode_from_precheck(row),
            }
        return costs

    def _best_local_mode_from_precheck(self, row: dict[str, Any]) -> str:
        best_mode = ""
        best_score = math.inf
        for mode in ("A", "B", "C"):
            data = row.get(mode) or {}
            if not data.get("feasible"):
                continue
            try:
                score = float(data.get("total_score", math.inf))
            except (TypeError, ValueError):
                continue
            if math.isfinite(score) and score < best_score:
                best_score = score
                best_mode = mode
        return best_mode

    def _explain_b_not_selected(
        self,
        selected_count: int,
        entered: bool,
        decoded: bool,
        feasible_orders: list[str],
        main_failure_reason: str,
    ) -> str:
        if selected_count > 0:
            return "B selected in best individual"
        if not entered:
            return "B never entered population"
        if not decoded:
            return f"B entered population but did not decode successfully; main failure={main_failure_reason or 'unknown'}"
        if not feasible_orders:
            return f"B decoded attempts were infeasible; main failure={main_failure_reason or 'unknown'}"
        dominated = all(
            self._best_local_mode_from_precheck(self._b_precheck_by_order.get(order_id, {})) != "B"
            for order_id in feasible_orders
        )
        if dominated:
            if main_failure_reason:
                return f"B feasible on limited orders but dominated by A/C local cost; main infeasible reason={main_failure_reason}"
            return "B feasible on limited orders but dominated by A/C local cost"
        return "B feasible locally but not selected by global sequence fitness"

    def _log_generation(self, generation: int, population: list[Individual]) -> None:
        if not self.config.verbose or not self._should_log_generation(generation):
            return

        best = min(population, key=lambda ind: ind.fitness)
        avg = sum(float(ind.fitness) for ind in population) / max(1, len(population))
        feasible_count = sum(1 for ind in population if not ind.penalties)
        logger.info(
            "[GA-MMCE] gen=%s best=%.2f avg=%.2f feasible=%s penalties=%s",
            generation,
            best.fitness,
            avg,
            feasible_count,
            best.penalties,
        )

    def _annotate_plan(
        self,
        plan: DispatchPlan,
        context: Any,
        started: float,
        dispatch_type: str,
    ) -> None:
        modes: dict[str, int] = {}
        for allocation in plan.allocations:
            modes[allocation.mode] = modes.get(allocation.mode, 0) + 1

        summary = plan.summary
        summary["total_orders"] = len(context.order_ids)
        summary["feasible"] = sum(1 for allocation in plan.allocations if allocation.feasible)
        unserved_ids = summary.get("unserved_order_ids", []) or []
        summary["ga_feasible"] = (
            summary["feasible"] == summary["total_orders"]
            and not unserved_ids
        )
        summary["modes"] = modes
        summary["dispatch_type"] = dispatch_type
        summary["solver"] = "ga_mmce"
        summary["runtime_seconds"] = time.time() - started
        summary["best_fitness"] = float(plan.cost_total or 0.0)
        full_breakdown = dict(summary.get("cost_breakdown", {}) or {})
        summary["cost_breakdown_detail"] = full_breakdown
        summary["cost_breakdown"] = {
            "dist": sum(float(allocation.cost_dist or 0.0) for allocation in plan.allocations),
            "energy": sum(float(allocation.cost_energy or 0.0) for allocation in plan.allocations),
            "penalty": float(summary.get("total_penalty_cost", 0.0) or 0.0),
        }
        summary["early_stopping"] = dict(self._early_stop_info)
        summary["b_candidate_precheck"] = dict(self._b_precheck_by_order)

    def _attach_solve_result(
        self,
        plan: DispatchPlan,
        best: Individual | None,
        started: float,
    ) -> None:
        elapsed = time.time() - started
        ga_feasible = bool(plan.summary.get("ga_feasible", plan.summary.get("feasible", False)))
        if int(plan.summary.get("total_orders", 0) or 0) == 0 and not plan.summary.get("unserved_order_ids"):
            ga_feasible = True
        plan.summary["best_individual"] = self._individual_summary(best)
        plan.summary["best_plan"] = "attached"
        plan.summary["ga_feasible"] = ga_feasible
        plan.summary["actual_generations"] = int(self._actual_generations)
        plan.summary["elapsed_seconds"] = elapsed
        plan.summary["early_stop_triggered"] = bool(self._early_stop_info.get("early_stop_triggered", False))
        plan.summary["time_budget_hit"] = bool(self._time_budget_hit)
        plan.summary.setdefault("fallback_used", False)

        try:
            setattr(plan, "best_individual", copy.deepcopy(best))
            setattr(plan, "best_plan", plan)
            setattr(plan, "ga_feasible", ga_feasible)
            setattr(plan, "actual_generations", int(self._actual_generations))
            setattr(plan, "elapsed_seconds", elapsed)
            setattr(plan, "early_stop_triggered", bool(self._early_stop_info.get("early_stop_triggered", False)))
            setattr(plan, "time_budget_hit", bool(self._time_budget_hit))
            setattr(plan, "fallback_used", bool(plan.summary.get("fallback_used", False)))
        except Exception:
            pass

    def _individual_summary(self, individual: Individual | None) -> dict[str, Any] | None:
        if individual is None:
            return None
        return {
            "sequence": list(individual.sequence),
            "assignment": list(individual.assignment),
            "rendezvous": copy.deepcopy(individual.rendezvous),
            "fitness": float(getattr(individual, "fitness", math.inf)),
        }

    def _empty_plan(self, dispatch_type: str = "ga_mmce", reason: str = "") -> DispatchPlan:
        summary = {
            "total_orders": 0,
            "feasible": 0,
            "modes": {},
            "dispatch_type": dispatch_type,
            "solver": "ga_mmce",
            "cost_breakdown": {"dist": 0.0, "energy": 0.0, "penalty": 0.0},
        }
        if reason:
            summary["reason"] = reason
        return DispatchPlan(allocations=[], cost_total=0.0, summary=summary)

    def _make_greedy_helper(self, entity_mgr: Any) -> Any:
        try:
            from ..greedy_mmce import GreedyMMCE
        except Exception:
            try:
                from greedy_mmce import GreedyMMCE
            except Exception:
                from solver.greedy_mmce import GreedyMMCE
        return GreedyMMCE(entity_mgr)

    def _make_dynamic_greedy_helper(self, entity_mgr: Any) -> Any:
        try:
            from ..greedy_mmce_bi import GreedyMMCEBackboneInsertion
        except Exception:
            try:
                from greedy_mmce_bi import GreedyMMCEBackboneInsertion
            except Exception:
                from solver.greedy_mmce_bi import GreedyMMCEBackboneInsertion
        return GreedyMMCEBackboneInsertion(entity_mgr)

    def _debug_run_start(self, state: Any, context: Any, dispatch_type: str) -> None:
        self._debug_write("")
        self._debug_write("=" * 100)
        self._debug_write(
            "RUN START "
            f"dispatch_type={dispatch_type} "
            f"orders={len(context.order_ids)} "
            f"current_time={float(getattr(state, 'current_time', 0.0) or 0.0):.2f}"
        )
        self._debug_write(f"order_ids={context.order_ids}")
        self._debug_write(f"truck_ids={context.truck_ids} depot_ids={context.depot_ids} station_ids={context.station_ids}")
        self._debug_write(f"truck_drone_ids={context.truck_drone_ids}")
        self._debug_write(f"depot_drone_ids={context.depot_drone_ids}")
        self._debug_write(f"all_drone_ids={context.all_drone_ids}")
        self._debug_write(f"gene_pool={context.gene_pool}")
        self._debug_write(f"support_node_ids={context.support_node_ids}")
        self._debug_write(
            "ga_config "
            f"max_generations={self.config.generations} "
            f"population_size={self.config.population_size} "
            f"min_generations={self.config.min_generations} "
            f"early_stopping_patience={self.config.early_stopping_patience} "
            f"improvement_tolerance={self.config.improvement_tolerance} "
            f"log_interval={self.config.log_interval} "
            f"mode_mutation_probabilities={self._mutation_mode_probabilities()} "
            f"balanced_initialization={self.config.use_balanced_initialization} "
            f"b_seeded_initialization={self.config.use_b_seeded_initialization}"
        )

    def _debug_population_snapshot(self, label: str, population: list[Individual]) -> None:
        if not population:
            self._debug_write(f"{label}: population empty")
            return

        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        best = sorted_pop[0]
        avg = sum(float(ind.fitness) for ind in sorted_pop) / len(sorted_pop)
        feasible_individuals = sum(1 for ind in sorted_pop if not ind.penalties)
        mode_counts = self._population_mode_counts(sorted_pop)
        individual_counts = self._population_individual_mode_counts(sorted_pop)
        self._debug_write(
            f"{label}: best={float(best.fitness):.3f} avg={avg:.3f} "
            f"feasible_individuals={feasible_individuals}/{len(sorted_pop)} "
            f"best_penalties={best.penalties}"
        )
        self._debug_write(
            f"{label}: mode_counts={{'A': {mode_counts.get('A', 0)}, "
            f"'B': {mode_counts.get('B', 0)}, 'C': {mode_counts.get('C', 0)}}} "
            f"individual_counts={individual_counts}"
        )

        result = getattr(best, "decoded_result", None)
        if result is not None:
            self._debug_write(
                f"{label}: decode feasible={result.feasible} "
                f"objective={result.objective:.3f} "
                f"penalties={result.penalties} "
                f"penalty_counts={result.penalty_counts} "
                f"unserved={result.unserved_order_ids}"
            )
            self._debug_write(f"{label}: metrics={result.metrics}")
        elif self._debug_eval_errors:
            self._debug_write(f"{label}: eval_errors={self._debug_eval_errors}")

    def _debug_run_end(
        self,
        plan: DispatchPlan,
        best: Individual | None,
        context: Any,
        started: float,
    ) -> None:
        self._debug_write(
            "RUN END "
            f"elapsed={time.time() - started:.3f}s "
            f"plan_cost={float(plan.cost_total or 0.0):.3f} "
            f"summary={plan.summary}"
        )
        self._debug_b_final_diag(plan.summary.get("b_mode_final_diag", {}))
        if best is not None:
            self._debug_write(f"best_sequence={best.sequence}")
            self._debug_write(f"best_assignment={best.assignment}")
            self._debug_write(f"best_rendezvous={best.rendezvous}")
            result = getattr(best, "decoded_result", None)
            if result is not None:
                self._debug_write(f"best_penalty_counts={result.penalty_counts}")
                self._debug_write(f"best_unserved_order_ids={result.unserved_order_ids}")
                self._debug_write(f"best_candidate_count={len(result.candidates)}")
                for candidate in result.candidates[:20]:
                    self._debug_write(
                        "candidate "
                        f"order={candidate.order_id} mode={candidate.mode} "
                        f"truck={candidate.truck_id} drone={candidate.drone_id} "
                        f"launch={candidate.launch_node_id} recover={candidate.recover_node_id} "
                        f"score={candidate.score_total:.3f} "
                        f"completion={candidate.completion_time:.1f} "
                        f"reward={float(getattr(candidate, 'mode_reward', 0.0) or 0.0):.1f} "
                        f"truck_dist={candidate.truck_distance:.1f} "
                        f"uav_dist={candidate.uav_distance:.1f}"
                    )
        if self._debug_eval_errors:
            self._debug_write(f"evaluation_exceptions={self._debug_eval_errors}")
        self._debug_write("=" * 100)

    def _debug_b_final_diag(self, summary: Any) -> None:
        if not isinstance(summary, dict) or not summary:
            return
        self._debug_write(
            "B_MODE_FINAL_DIAG: "
            f"entered_population={summary.get('b_entered_population')} "
            f"decode_success={summary.get('b_decode_success')} "
            f"decode_success_total={summary.get('b_decode_success_total')} "
            f"repaired_total={summary.get('b_repaired_total')} "
            f"selected_in_best={summary.get('b_selected_in_best')} "
            f"main_failure_reason={summary.get('main_failure_reason')} "
            f"feasible_orders={summary.get('b_feasible_orders')} "
            f"reason={summary.get('b_not_selected_reason')}"
        )
        self._debug_write(
            "B_MODE_FINAL_DIAG_COSTS: "
            f"b_feasible_order_costs={summary.get('b_feasible_order_costs')}"
        )

    def _debug_write(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(f"{timestamp} {message}\n")
        except Exception:
            logger.debug("[GA-MMCE] failed to write debug log", exc_info=True)
