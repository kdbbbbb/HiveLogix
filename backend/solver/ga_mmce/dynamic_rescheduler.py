from __future__ import annotations

import copy
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from .adapters import build_ga_context, clone_state_for_decode
from .chromosome import Individual
from .config import DYNAMIC_GA_CONFIG, GAConfig, make_ga_config
from .decoder import AllocationResult, DispatchPlan
from .operators import make_random_rendezvous_for_gene, mutate
from .population import enforce_fixed_tail


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAST_DYNAMIC_REPLAN_STATS: dict[str, Any] = {}


@dataclass
class OrderBuckets:
    completed: dict[str, Any]
    locked: dict[str, Any]
    pending: dict[str, Any]
    new: dict[str, Any]


DYNAMIC_REPLAN_FIELDS = [
    "event_time",
    "new_order_ids",
    "completed_ids",
    "locked_ids",
    "pending_count",
    "reoptimized_order_count",
    "frozen_future_order_count",
    "warm_start_count",
    "archive_seed_count",
    "archive_seed_B_count",
    "archive_seed_C_count",
    "warm_start_with_B",
    "warm_start_with_C",
    "force_b_repair_seed_count",
    "dynamic_archive_size",
    "population_size",
    "max_generations",
    "actual_generations",
    "elapsed_seconds",
    "time_budget_hit",
    "early_stop_triggered",
    "fallback_used",
    "fallback_level",
    "best_fitness",
    "fast_incumbent_found",
    "fast_incumbent_selected",
    "fast_incumbent_eval_count",
    "fast_incumbent_cost",
    "fast_incumbent_A_count",
    "fast_incumbent_B_count",
    "fast_incumbent_C_count",
    "old_assignment_changes",
    "old_rendezvous_changes",
    "old_sequence_inversions",
    "bc_drop_count",
    "previous_bc_count",
    "candidate_bc_count",
    "archive_bc_available",
    "all_a_rejected",
    "new_order_pending_count",
    "plan_selection_reason",
    "ga_route_patch_enabled",
    "ga_route_patch_replace_order_ids",
    "ga_route_patch_b_wait_order_ids",
    "final_A_count",
    "final_B_count",
    "final_C_count",
    "unserved_order_ids",
]


def reschedule_on_event(state: Any, new_orders: Any, event_time: float) -> DispatchPlan:
    """Main dynamic GA entrypoint.

    The function advances or snapshots existing runtime state when such hooks
    exist, freezes active work through GA-only fields, then runs a small,
    warm-started GA with fallback layers.
    """
    started = time.time()
    advanced_state = _advance_or_snapshot_state(state, event_time)
    solver = _resolve_solver(advanced_state)
    previous_best = copy.deepcopy(getattr(solver, "last_best_individual", None))
    previous_decode = getattr(solver, "last_best_decode_result", None)
    previous_plan = getattr(previous_decode, "plan", None) if previous_decode is not None else None

    incoming_new = _normalize_order_mapping(new_orders)
    buckets = _classify_orders(advanced_state, incoming_new, event_time)
    dynamic_config = make_ga_config(DYNAMIC_GA_CONFIG, base=getattr(solver, "config", None))
    reoptimized_ids, frozen_future_ids = _select_reoptimization_window(
        previous_best,
        buckets.pending,
        buckets.new,
        dynamic_config,
        event_time,
    )

    planning_orders = _ordered_order_mapping(
        reoptimized_ids + frozen_future_ids,
        {**buckets.pending, **buckets.new},
    )
    if not planning_orders:
        plan = DispatchPlan(
            allocations=[],
            cost_total=0.0,
            summary={
                "total_orders": 0,
                "feasible": 0,
                "modes": {},
                "dispatch_type": "dynamic_replan",
                "solver": "ga_mmce",
                "ga_feasible": True,
            },
        )
        _finalize_and_log(
            plan,
            started,
            event_time,
            buckets,
            reoptimized_ids,
            frozen_future_ids,
            warm_start_count=0,
            config=dynamic_config,
            fallback_used=False,
            fallback_level="none",
        )
        return plan

    snapshot = _build_dynamic_snapshot(
        advanced_state,
        planning_orders,
        event_time,
        completed_ids=set(buckets.completed),
        locked_ids=set(buckets.locked),
    )

    initial_context = build_ga_context(snapshot, dynamic_config, mode="dynamic")
    hard_preserve = bool(getattr(dynamic_config, "dynamic_preserve_previous_best", True))
    fixed_tail_order_ids = [] if hard_preserve else list(frozen_future_ids)
    fixed_tail_gene_by_order = (
        {}
        if hard_preserve
        else _tail_gene_map(previous_best, frozen_future_ids, initial_context.gene_pool)
    )
    setattr(snapshot, "_ga_fixed_tail_order_ids", list(fixed_tail_order_ids))
    setattr(snapshot, "_ga_fixed_tail_gene_by_order", fixed_tail_gene_by_order)
    setattr(snapshot, "_ga_reoptimized_order_ids", list(reoptimized_ids))

    context = build_ga_context(snapshot, dynamic_config, mode="dynamic")
    previous_best_seeds = build_warm_start_population(
        previous_best=previous_best,
        completed_ids=set(buckets.completed),
        locked_ids=set(buckets.locked),
        new_order_ids=list(buckets.new),
        gene_pool=context.gene_pool,
        depot_ids=context.depot_ids,
        station_ids=context.station_ids,
        order_ids=list(planning_orders),
        reoptimized_order_ids=reoptimized_ids,
        frozen_future_order_ids=fixed_tail_order_ids,
        fixed_tail_gene_by_order=fixed_tail_gene_by_order,
        orders=planning_orders,
        allow_c_recover_station=dynamic_config.allow_depot_drone_recover_at_station,
        mutation_count=int(dynamic_config.warm_start_mutations or 0),
    )
    archive_seeds = build_archive_warm_starts(
        solver=solver,
        previous_best=previous_best,
        completed_ids=set(buckets.completed),
        locked_ids=set(buckets.locked),
        new_order_ids=list(buckets.new),
        gene_pool=context.gene_pool,
        depot_ids=context.depot_ids,
        station_ids=context.station_ids,
        order_ids=list(planning_orders),
        orders=planning_orders,
        allow_c_recover_station=dynamic_config.allow_depot_drone_recover_at_station,
        max_count=int(getattr(dynamic_config, "dynamic_archive_seed_count", 16) or 0),
    )
    archive_restore_seeds = build_bc_restore_warm_starts(
        solver=solver,
        previous_best=previous_best,
        completed_ids=set(buckets.completed),
        locked_ids=set(buckets.locked),
        new_order_ids=list(buckets.new),
        gene_pool=context.gene_pool,
        depot_ids=context.depot_ids,
        station_ids=context.station_ids,
        order_ids=list(planning_orders),
        orders=planning_orders,
        allow_c_recover_station=dynamic_config.allow_depot_drone_recover_at_station,
        max_count=int(getattr(dynamic_config, "dynamic_archive_seed_count", 16) or 0),
    )
    fast_incumbent_plan, fast_incumbent_diag, fast_incumbent_seeds = _fast_incumbent_plan(
        solver=solver,
        snapshot=snapshot,
        previous_best=previous_best,
        previous_plan=previous_plan,
        seed_candidates=archive_restore_seeds + archive_seeds + previous_best_seeds,
        config=dynamic_config,
        planning_orders=planning_orders,
        new_order_ids=list(buckets.new),
        context=context,
    )
    warm_starts = _merge_warm_starts(
        fast_incumbent_seeds + archive_restore_seeds + archive_seeds + previous_best_seeds,
        max_count=int(getattr(dynamic_config, "dynamic_warm_start_limit", 24) or 24),
    )
    archive_all_seeds = archive_restore_seeds + archive_seeds
    archive_bc_available = _any_seed_has_bc(archive_all_seeds, planning_orders)
    archive_seed_B_count = _seed_mode_count(archive_all_seeds, "B")
    archive_seed_C_count = _seed_mode_count(archive_all_seeds, "C")
    warm_start_with_B = _seed_with_mode_count(warm_starts, "B")
    warm_start_with_C = _seed_with_mode_count(warm_starts, "C")

    fast_diag = _stability_diagnostics(
        previous_best=previous_best,
        plan=fast_incumbent_plan,
        planning_orders=planning_orders,
        new_order_ids=set(buckets.new),
    )
    ga_skipped_for_fast = _fast_incumbent_can_skip_ga(
        fast_incumbent_plan,
        fast_diag,
        planning_orders,
        dynamic_config,
    )
    if ga_skipped_for_fast:
        ga_plan = fast_incumbent_plan
        _remember_fast_incumbent_on_solver(solver, fast_incumbent_seeds, fast_incumbent_plan, context)
    else:
        ga_plan = solver.solve(
            snapshot,
            warm_start=warm_starts,
            config=dynamic_config,
            time_budget_seconds=dynamic_config.max_runtime_seconds,
            dispatch_type="dynamic_replan",
        )
    ga_feasible = _plan_is_feasible_for_orders(ga_plan, planning_orders)

    fallback_used = False
    fallback_level = "none"
    warm_start_feasible = False
    greedy_insert_feasible = False
    ga_diag = _stability_diagnostics(
        previous_best=previous_best,
        plan=ga_plan,
        planning_orders=planning_orders,
        new_order_ids=set(buckets.new),
    )
    fast_diag = _stability_diagnostics(
        previous_best=previous_best,
        plan=fast_incumbent_plan,
        planning_orders=planning_orders,
        new_order_ids=set(buckets.new),
    )
    _attach_archive_stability_fields(ga_diag, archive_bc_available)
    _attach_archive_stability_fields(fast_diag, archive_bc_available)
    final_plan = ga_plan
    selection_diag = dict(ga_diag)
    plan_selection_reason = "ga"
    all_a_rejected = False

    if ga_skipped_for_fast and fast_incumbent_plan is not None:
        final_plan = fast_incumbent_plan
        selection_diag = dict(fast_diag)
        fallback_level = "fast_incumbent"
        plan_selection_reason = "fast_incumbent_safe_skip_ga"
    elif fast_incumbent_plan is not None:
        if _ga_can_replace_fast_incumbent(ga_diag, fast_diag, dynamic_config):
            final_plan = ga_plan
            selection_diag = dict(ga_diag)
            plan_selection_reason = "ga_replaced_fast_incumbent"
        else:
            all_a_rejected = bool(ga_diag.get("all_a_rejected", False))
            final_plan = fast_incumbent_plan
            selection_diag = dict(fast_diag)
            fallback_level = "fast_incumbent"
            plan_selection_reason = "fast_incumbent_preserved_old_structure"
    elif not ga_feasible and bool(getattr(dynamic_config, "dynamic_allow_new_order_pending", True)) and previous_plan is not None:
        final_plan = _previous_plan_fallback(
            previous_plan,
            planning_orders,
            new_order_ids=set(buckets.new),
            reason="dynamic_new_order_pending",
        )
        selection_diag = _stability_diagnostics(
            previous_best=previous_best,
            plan=final_plan,
            planning_orders=planning_orders,
            new_order_ids=set(buckets.new),
        )
        fallback_used = True
        fallback_level = "previous_plan_unserved_new"
        plan_selection_reason = "previous_pending"
    elif not ga_feasible:
        warm_plan = _best_feasible_warm_start_plan(
            solver,
            snapshot,
            warm_starts,
            dynamic_config,
            planning_orders,
        )
        warm_start_feasible = warm_plan is not None
        if warm_plan is not None:
            final_plan = warm_plan
            selection_diag = _stability_diagnostics(
                previous_best=previous_best,
                plan=final_plan,
                planning_orders=planning_orders,
                new_order_ids=set(buckets.new),
            )
            fallback_used = True
            fallback_level = "warm_start_repaired"
            plan_selection_reason = "warm_start_repaired"
        else:
            greedy_plan = _greedy_replan_fallback(solver, snapshot, planning_orders, event_time)
            greedy_insert_feasible = greedy_plan is not None and _plan_is_feasible_for_orders(greedy_plan, planning_orders)
            if greedy_insert_feasible and not bool(getattr(dynamic_config, "dynamic_allow_new_order_pending", True)):
                final_plan = greedy_plan
                selection_diag = _stability_diagnostics(
                    previous_best=previous_best,
                    plan=final_plan,
                    planning_orders=planning_orders,
                    new_order_ids=set(buckets.new),
                )
                fallback_used = True
                fallback_level = "greedy_insertion"
                plan_selection_reason = "greedy_insertion"
            else:
                final_plan = _previous_plan_fallback(
                    previous_plan,
                    planning_orders,
                    new_order_ids=set(buckets.new),
                    reason="dynamic_ga_failed",
                )
                selection_diag = _stability_diagnostics(
                    previous_best=previous_best,
                    plan=final_plan,
                    planning_orders=planning_orders,
                    new_order_ids=set(buckets.new),
                )
                fallback_used = True
                fallback_level = "previous_plan_unserved_new"
                plan_selection_reason = "previous_pending"

    unserved_order_ids = list(_unserved_order_ids(final_plan, planning_orders))
    selection_diag.update(
        {
            "fast_incumbent_found": fast_incumbent_plan is not None,
            "fast_incumbent_selected": final_plan is fast_incumbent_plan and fast_incumbent_plan is not None,
            "fast_incumbent_eval_count": int(fast_incumbent_diag.get("fast_incumbent_eval_count", 0) or 0),
            "fast_incumbent_cost": float(fast_incumbent_diag.get("fast_incumbent_cost", 0.0) or 0.0),
            "fast_incumbent_A_count": int(fast_incumbent_diag.get("fast_incumbent_A_count", 0) or 0),
            "fast_incumbent_B_count": int(fast_incumbent_diag.get("fast_incumbent_B_count", 0) or 0),
            "fast_incumbent_C_count": int(fast_incumbent_diag.get("fast_incumbent_C_count", 0) or 0),
            "archive_seed_count": len(archive_all_seeds),
            "archive_seed_B_count": archive_seed_B_count,
            "archive_seed_C_count": archive_seed_C_count,
            "warm_start_with_B": warm_start_with_B,
            "warm_start_with_C": warm_start_with_C,
            "force_b_repair_seed_count": int(getattr(solver, "_force_b_repair_seed_count", 0) or 0),
            "dynamic_archive_size": len(getattr(solver, "dynamic_archive", []) or []),
            "archive_bc_available": bool(archive_bc_available),
            "all_a_rejected": bool(all_a_rejected or selection_diag.get("all_a_rejected", False)),
            "plan_selection_reason": plan_selection_reason,
        }
    )
    _apply_ga_route_patch(
        final_plan,
        reoptimized_ids=reoptimized_ids,
        new_order_ids=list(buckets.new),
        frozen_future_ids=frozen_future_ids,
        locked_ids=set(buckets.locked),
        enabled=hard_preserve,
    )
    if final_plan is fast_incumbent_plan and fast_incumbent_plan is not None:
        _remember_fast_incumbent_on_solver(solver, fast_incumbent_seeds, fast_incumbent_plan, context)
    _annotate_dynamic_summary(
        final_plan,
        event_time=event_time,
        buckets=buckets,
        reoptimized_ids=reoptimized_ids,
        frozen_future_ids=frozen_future_ids,
        warm_start_count=len(warm_starts),
        config=dynamic_config,
        ga_plan=ga_plan,
        ga_feasible=ga_feasible,
        warm_start_feasible=warm_start_feasible,
        greedy_insert_feasible=greedy_insert_feasible,
        fallback_used=fallback_used,
        fallback_level=fallback_level,
        unserved_order_ids=unserved_order_ids,
        elapsed_seconds=time.time() - started,
        selection_diag=selection_diag,
    )
    _debug_dynamic_stability(selection_diag)
    _write_dynamic_replan_csv(final_plan.summary)
    _store_last_stats(final_plan.summary)
    return final_plan


def build_warm_start(
    previous_best: Individual | None,
    completed_ids: Iterable[str],
    locked_ids: Iterable[str],
    new_order_ids: Iterable[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    allow_c_recover_station: bool = True,
) -> Individual:
    excluded = {str(oid) for oid in completed_ids} | {str(oid) for oid in locked_ids}
    support_node_ids = list(depot_ids) + list(station_ids)

    seq: list[str] = []
    assignment: list[str] = []
    rendezvous = []

    if previous_best is not None:
        for oid, gene, rv in zip(
            previous_best.sequence,
            previous_best.assignment,
            previous_best.rendezvous,
        ):
            if oid in excluded:
                continue
            seq.append(str(oid))
            assignment.append(str(gene))
            rendezvous.append(copy.deepcopy(rv))

    for oid in new_order_ids:
        gene = _preferred_gene(gene_pool, ("C", "B", "A"))
        seq.append(str(oid))
        assignment.append(gene)
        rendezvous.append(
            make_random_rendezvous_for_gene(
                gene,
                depot_ids,
                station_ids,
                allow_c_recover_station,
            )
        )

    ind = Individual(seq, assignment, rendezvous)
    ind.validate()
    return ind


def build_warm_start_population(
    previous_best: Individual | None,
    completed_ids: Iterable[str],
    locked_ids: Iterable[str],
    new_order_ids: Iterable[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    order_ids: list[str],
    reoptimized_order_ids: list[str],
    frozen_future_order_ids: list[str],
    fixed_tail_gene_by_order: dict[str, tuple[str, dict[str, str] | None]] | None = None,
    orders: dict[str, Any] | None = None,
    allow_c_recover_station: bool = True,
    mutation_count: int = 0,
) -> list[Individual]:
    support_node_ids = list(depot_ids) + list(station_ids)
    excluded = {str(oid) for oid in completed_ids} | {str(oid) for oid in locked_ids}
    new_ids = [str(oid) for oid in new_order_ids if str(oid) in set(order_ids)]
    reopt_set = set(reoptimized_order_ids)
    fixed_tail_gene_by_order = fixed_tail_gene_by_order or {}
    population: list[Individual] = []

    base_order_ids = [oid for oid in order_ids if oid not in set(new_ids)]
    base = _previous_remaining_individual(
        previous_best,
        base_order_ids,
        excluded,
        gene_pool,
        support_node_ids,
        allow_c_recover_station,
    )
    if base is not None:
        population.append(_with_new_orders_appended(base, new_ids, gene_pool, depot_ids, station_ids, allow_c_recover_station))
        population.append(_with_new_orders_nearest(base, new_ids, gene_pool, depot_ids, station_ids, orders, allow_c_recover_station))

    for mode in ("A", "B", "C"):
        gene = _first_gene_for_mode(gene_pool, mode)
        if gene is not None:
            population.append(
                _new_orders_with_mode(
                    base,
                    order_ids,
                    new_ids,
                    gene,
                    depot_ids,
                    station_ids,
                    allow_c_recover_station,
                )
            )

    mutation_base = population[0] if population else _truck_only(order_ids)
    for _ in range(max(1, mutation_count)):
        mutated = copy.deepcopy(mutation_base)
        mutable_indices = [i for i, oid in enumerate(mutated.sequence) if oid in reopt_set]
        if len(mutable_indices) >= 2:
            i, j = random.sample(mutable_indices, 2)
            mutated.sequence[i], mutated.sequence[j] = mutated.sequence[j], mutated.sequence[i]
            mutated.assignment[i], mutated.assignment[j] = mutated.assignment[j], mutated.assignment[i]
            mutated.rendezvous[i], mutated.rendezvous[j] = mutated.rendezvous[j], mutated.rendezvous[i]
        try:
            mutate(
                mutated,
                gene_pool,
                support_node_ids,
                p_seq=0.0,
                p_assign=0.08,
                p_rendezvous=0.08,
                allow_c_recover_station=allow_c_recover_station,
            )
        except Exception:
            continue
        population.append(mutated)

    rv_perturbed = copy.deepcopy(mutation_base)
    for idx, gene in enumerate(rv_perturbed.assignment):
        if rv_perturbed.sequence[idx] not in reopt_set or gene == "A":
            continue
        rv_perturbed.rendezvous[idx] = make_random_rendezvous_for_gene(
            gene,
            depot_ids,
            station_ids,
            allow_c_recover_station,
        )
        break
    population.append(rv_perturbed)

    repaired: list[Individual] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str]] = set()
    for ind in population:
        try:
            _repair_to_order_ids(
                ind,
                order_ids,
                gene_pool,
                depot_ids,
                station_ids,
                allow_c_recover_station,
            )
            _repair_rendezvous_to_context(ind, gene_pool, depot_ids, station_ids, allow_c_recover_station)
            enforce_fixed_tail(ind, frozen_future_order_ids, fixed_tail_gene_by_order)
            _repair_rendezvous_to_context(ind, gene_pool, depot_ids, station_ids, allow_c_recover_station)
            key = (tuple(ind.sequence), tuple(ind.assignment), repr(ind.rendezvous))
            if key in seen:
                continue
            seen.add(key)
            _validate_seed_with_context(ind, gene_pool, depot_ids, station_ids)
            repaired.append(ind)
        except Exception:
            continue
    return repaired


def build_archive_warm_starts(
    solver: Any,
    previous_best: Individual | None,
    completed_ids: Iterable[str],
    locked_ids: Iterable[str],
    new_order_ids: Iterable[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    order_ids: list[str],
    orders: dict[str, Any] | None = None,
    allow_c_recover_station: bool = True,
    max_count: int = 16,
) -> list[Individual]:
    if max_count <= 0:
        return []

    sources: list[Individual] = []
    sources.extend(copy.deepcopy(getattr(solver, "dynamic_archive", []) or []))
    sources.extend(copy.deepcopy(getattr(solver, "last_population", []) or []))
    if not sources:
        return []

    excluded = {str(oid) for oid in completed_ids} | {str(oid) for oid in locked_ids}
    new_ids = [str(oid) for oid in new_order_ids if str(oid) in set(order_ids)]
    base_order_ids = [oid for oid in order_ids if oid not in set(new_ids)]
    candidates: list[Individual] = []
    for source in _sort_archive_sources(sources, previous_best, base_order_ids):
        base = _previous_remaining_individual(
            source,
            base_order_ids,
            excluded,
            gene_pool,
            list(depot_ids) + list(station_ids),
            allow_c_recover_station,
        )
        if base is None:
            continue
        candidates.append(_with_new_orders_appended(base, new_ids, gene_pool, depot_ids, station_ids, allow_c_recover_station))
        candidates.append(_with_new_orders_nearest(base, new_ids, gene_pool, depot_ids, station_ids, orders, allow_c_recover_station))
        candidates.append(_with_new_orders_deadline(base, new_ids, gene_pool, depot_ids, station_ids, orders, allow_c_recover_station))

    return _dedupe_and_validate_seeds(
        candidates,
        order_ids=order_ids,
        gene_pool=gene_pool,
        depot_ids=depot_ids,
        station_ids=station_ids,
        allow_c_recover_station=allow_c_recover_station,
        max_count=max_count,
    )


def build_bc_restore_warm_starts(
    solver: Any,
    previous_best: Individual | None,
    completed_ids: Iterable[str],
    locked_ids: Iterable[str],
    new_order_ids: Iterable[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    order_ids: list[str],
    orders: dict[str, Any] | None = None,
    allow_c_recover_station: bool = True,
    max_count: int = 16,
) -> list[Individual]:
    """Recover historical B/C genes without reordering old orders."""
    if previous_best is None or max_count <= 0:
        return []

    sources: list[Individual] = []
    sources.extend(copy.deepcopy(getattr(solver, "dynamic_archive", []) or []))
    sources.extend(copy.deepcopy(getattr(solver, "last_population", []) or []))
    if not sources:
        return []

    excluded = {str(oid) for oid in completed_ids} | {str(oid) for oid in locked_ids}
    new_ids = [str(oid) for oid in new_order_ids if str(oid) in set(order_ids)]
    new_set = set(new_ids)
    base_order_ids = [oid for oid in order_ids if oid not in new_set]
    support_node_ids = list(depot_ids) + list(station_ids)
    base = _previous_remaining_individual(
        previous_best,
        base_order_ids,
        excluded,
        gene_pool,
        support_node_ids,
        allow_c_recover_station,
    )
    if base is None:
        return []

    candidates: list[Individual] = []
    for source in _sort_archive_sources(sources, previous_best, base_order_ids):
        restored = copy.deepcopy(base)
        archive_by_order = {
            str(oid): (str(gene), copy.deepcopy(rv))
            for oid, gene, rv in zip(source.sequence, source.assignment, source.rendezvous)
        }
        changed = False
        for idx, order_id in enumerate(restored.sequence):
            if order_id in new_set:
                continue
            gene, rv = archive_by_order.get(order_id, ("A", None))
            if _gene_mode_family(gene) not in {"B", "C"}:
                continue
            if not _gene_and_rendezvous_legal(gene, rv, gene_pool, depot_ids, station_ids):
                continue
            restored.assignment[idx] = gene
            restored.rendezvous[idx] = copy.deepcopy(rv)
            changed = True
        if not changed:
            continue
        candidates.append(
            _with_new_orders_appended(
                restored,
                new_ids,
                gene_pool,
                depot_ids,
                station_ids,
                allow_c_recover_station,
            )
        )
        candidates.append(
            _with_new_orders_nearest(
                restored,
                new_ids,
                gene_pool,
                depot_ids,
                station_ids,
                orders,
                allow_c_recover_station,
            )
        )
        if len(candidates) >= max_count * 2:
            break

    return _dedupe_and_validate_seeds(
        candidates,
        order_ids=order_ids,
        gene_pool=gene_pool,
        depot_ids=depot_ids,
        station_ids=station_ids,
        allow_c_recover_station=allow_c_recover_station,
        max_count=max_count,
    )


def _sort_archive_sources(
    sources: list[Individual],
    previous_best: Individual | None,
    order_ids: list[str],
) -> list[Individual]:
    order_set = set(str(oid) for oid in order_ids)
    previous_modes = {
        str(oid): _gene_mode_family(str(gene))
        for oid, gene in zip(getattr(previous_best, "sequence", []) or [], getattr(previous_best, "assignment", []) or [])
        if str(oid) in order_set
    } if previous_best is not None else {}
    previous_rank = {
        str(oid): idx
        for idx, oid in enumerate(getattr(previous_best, "sequence", []) or [])
        if str(oid) in order_set
    } if previous_best is not None else {}

    def source_key(source: Individual) -> tuple[int, int, int, float]:
        assignment_changes = 0
        source_modes = {
            str(oid): _gene_mode_family(str(gene))
            for oid, gene in zip(source.sequence, source.assignment)
            if str(oid) in order_set
        }
        for oid, mode in previous_modes.items():
            if source_modes.get(oid, mode) != mode:
                assignment_changes += 1
        sequence = [str(oid) for oid in source.sequence if str(oid) in order_set]
        inversions = _sequence_inversions(sequence, previous_rank)
        bc_count = _individual_bc_count(source, order_set)
        return (
            assignment_changes,
            inversions,
            -bc_count,
            float(getattr(source, "fitness", math.inf)),
        )

    return sorted(sources, key=source_key)


def _merge_warm_starts(seeds: list[Individual], max_count: int) -> list[Individual]:
    merged: list[Individual] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str]] = set()
    for seed in seeds:
        key = (tuple(seed.sequence), tuple(seed.assignment), repr(seed.rendezvous))
        if key in seen:
            continue
        seen.add(key)
        merged.append(seed)
        if len(merged) >= max_count:
            break
    return merged


def _dedupe_and_validate_seeds(
    seeds: list[Individual],
    order_ids: list[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    allow_c_recover_station: bool,
    max_count: int,
) -> list[Individual]:
    repaired: list[Individual] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str]] = set()
    for seed in seeds:
        try:
            candidate = copy.deepcopy(seed)
            _repair_to_order_ids(
                candidate,
                order_ids,
                gene_pool,
                depot_ids,
                station_ids,
                allow_c_recover_station,
            )
            _repair_rendezvous_to_context(candidate, gene_pool, depot_ids, station_ids, allow_c_recover_station)
            _validate_seed_with_context(candidate, gene_pool, depot_ids, station_ids)
            key = (tuple(candidate.sequence), tuple(candidate.assignment), repr(candidate.rendezvous))
            if key in seen:
                continue
            seen.add(key)
            repaired.append(candidate)
            if len(repaired) >= max_count:
                break
        except Exception:
            continue
    return repaired


def _repair_rendezvous_to_context(
    ind: Individual,
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    allow_c_recover_station: bool,
) -> None:
    support_set = set(list(depot_ids) + list(station_ids))
    for idx, gene in enumerate(list(ind.assignment)):
        if gene not in gene_pool:
            ind.assignment[idx] = "A"
            ind.rendezvous[idx] = None
            continue
        if gene == "A":
            ind.rendezvous[idx] = None
            continue
        rv = ind.rendezvous[idx] if isinstance(ind.rendezvous[idx], dict) else {}
        launch = str(rv.get("launch", "") or "")
        recover = str(rv.get("recover", "") or "")
        if launch not in support_set or recover not in support_set or (gene.startswith("C_") and not _is_depot_node_id(launch)):
            ind.rendezvous[idx] = make_random_rendezvous_for_gene(
                gene,
                depot_ids,
                station_ids,
                allow_c_recover_station,
            )


def _validate_seed_with_context(
    ind: Individual,
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
) -> None:
    truck_drone_ids = [_gene_drone_id(gene) for gene in gene_pool if gene.startswith("B_")]
    depot_drone_ids = [_gene_drone_id(gene) for gene in gene_pool if gene.startswith("C_")]
    valid_drone_ids = truck_drone_ids + depot_drone_ids
    ind.validate_with_context(
        truck_drone_ids=truck_drone_ids,
        depot_drone_ids=depot_drone_ids,
        valid_drone_ids=valid_drone_ids,
        support_node_ids=list(depot_ids) + list(station_ids),
    )


def _gene_and_rendezvous_legal(
    gene: str,
    rendezvous: Any,
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
) -> bool:
    if gene not in set(gene_pool):
        return False
    if _gene_mode_family(gene) not in {"B", "C"}:
        return False
    if not isinstance(rendezvous, dict):
        return False
    launch = str(rendezvous.get("launch", "") or "")
    recover = str(rendezvous.get("recover", "") or "")
    support = set(list(depot_ids) + list(station_ids))
    if launch not in support or recover not in support:
        return False
    if str(gene).startswith("C_") and launch not in set(depot_ids):
        return False
    return True


def _sequence_inversions(sequence: list[str], previous_rank: dict[str, int]) -> int:
    inversions = 0
    for i in range(len(sequence)):
        for j in range(i + 1, len(sequence)):
            if previous_rank.get(sequence[i], 10**9) > previous_rank.get(sequence[j], 10**9):
                inversions += 1
    return inversions


def _individual_bc_count(individual: Individual, order_ids: set[str] | None = None) -> int:
    order_ids = set(order_ids or [])
    count = 0
    for oid, gene in zip(individual.sequence, individual.assignment):
        if order_ids and str(oid) not in order_ids:
            continue
        if _gene_mode_family(str(gene)) in {"B", "C"}:
            count += 1
    return count


def _any_seed_has_bc(seeds: list[Individual], planning_orders: dict[str, Any]) -> bool:
    order_set = set(str(oid) for oid in planning_orders)
    return any(_individual_bc_count(seed, order_set) > 0 for seed in seeds)


def _seed_mode_count(seeds: list[Individual], mode: str) -> int:
    return sum(1 for seed in seeds for gene in seed.assignment if _gene_mode_family(str(gene)) == mode)


def _seed_with_mode_count(seeds: list[Individual], mode: str) -> int:
    return sum(1 for seed in seeds if any(_gene_mode_family(str(gene)) == mode for gene in seed.assignment))


def _gene_drone_id(gene: str) -> str:
    return str(gene).partition("_")[2]


def _advance_or_snapshot_state(state: Any, event_time: float) -> Any:
    for target in (state, _entity_mgr(state)):
        if target is None:
            continue
        for name in ("snapshot_at", "advance_state_to", "apply_plan_until"):
            fn = getattr(target, name, None)
            if not callable(fn):
                continue
            try:
                advanced = fn(event_time)
            except TypeError:
                continue
            if advanced is not None:
                state = advanced
                break
    _write_field(state, "current_time", float(event_time))
    return state


def _build_dynamic_snapshot(
    state: Any,
    planning_orders: dict[str, Any],
    event_time: float,
    completed_ids: set[str],
    locked_ids: set[str],
) -> Any:
    base = SimpleNamespace(
        entity_mgr=_entity_mgr(state),
        orders=dict(planning_orders),
        current_time=float(event_time),
        bbox=_read_field(state, "bbox"),
        scene_id=_read_field(state, "scene_id"),
        _ga_context_mode="dynamic",
    )
    snapshot = clone_state_for_decode(base)
    _write_field(snapshot, "current_time", float(event_time))
    _write_field(snapshot, "orders", dict(planning_orders))
    _write_field(snapshot, "_ga_context_mode", "dynamic")
    _write_field(snapshot, "completed_order_ids", set(completed_ids))
    _write_field(snapshot, "locked_order_ids", set(locked_ids))
    _freeze_runtime_resources(snapshot, event_time, locked_ids)
    return snapshot


def _freeze_runtime_resources(snapshot: Any, event_time: float, locked_order_ids: set[str]) -> None:
    mgr = _entity_mgr(snapshot)
    locked_drone_ids: set[str] = set()
    busy_drone_ids: set[str] = set()

    for drone_id, drone in (_mapping(mgr, "drones") or {}).items():
        carrying = str(_read_field(drone, "carrying_order_id", "") or "")
        is_busy = bool(carrying) or _drone_is_flying(drone) or _has_pending_route(drone) or bool(_read_field(drone, "waiting_recovery_station_id"))
        if carrying in locked_order_ids or is_busy:
            busy_drone_ids.add(str(drone_id))
            locked_drone_ids.add(str(drone_id))
            available_time, available_pos = _estimate_drone_available(snapshot, drone, event_time)
            _write_field(drone, "_ga_available_time", available_time)
            _write_field(drone, "_ga_time", available_time)
            if available_pos is not None:
                _write_field(drone, "_ga_position", available_pos)
            host_type, host_node_id, truck_id = _future_drone_host(snapshot, drone, str(drone_id))
            if host_type and host_node_id:
                _write_field(drone, "_ga_force_available", True)
                _write_field(drone, "_ga_host_type", host_type)
                _write_field(drone, "_ga_host_node_id", host_node_id)
                _write_field(drone, "_ga_transport_truck_id", truck_id if host_type == "TRUCK" else None)
                _write_field(drone, "_ga_waiting_station_id", host_node_id if host_type == "STATION" else None)
                _add_drone_to_future_host(snapshot, str(drone_id), host_type, host_node_id, truck_id)

    for truck in (_mapping(mgr, "trucks") or {}).values():
        freeze_time, freeze_pos = _estimate_truck_available(truck, event_time, locked_order_ids)
        _write_field(truck, "_ga_time", freeze_time)
        if freeze_pos is not None:
            _write_field(truck, "_ga_position", freeze_pos)

    _write_field(snapshot, "locked_drone_ids", locked_drone_ids)
    _write_field(snapshot, "busy_drone_ids", busy_drone_ids)
    _write_field(snapshot, "running_drone_ids", locked_drone_ids)


def _future_drone_host(snapshot: Any, drone: Any, drone_id: str) -> tuple[str, str, str | None]:
    route_host = _pending_route_recovery_host(snapshot, drone)
    if route_host is not None:
        host_type, host_node_id = route_host
        return host_type, host_node_id, None

    truck_id = str(_read_field(drone, "_ga_transport_truck_id", _read_field(drone, "transport_truck_id", "")) or "")
    if truck_id:
        return "TRUCK", truck_id, truck_id

    mgr = _entity_mgr(snapshot)
    for tid, truck in (_mapping(mgr, "trucks") or {}).items():
        docked = list(_read_field(truck, "_ga_docked_drones", []) or [])
        docked.extend(_read_field(truck, "docked_drones", []) or [])
        if drone_id in docked:
            return "TRUCK", str(tid), str(tid)

    station_id = str(_read_field(drone, "_ga_waiting_station_id", _read_field(drone, "waiting_recovery_station_id", "")) or "")
    if station_id:
        return "STATION", station_id, None

    home_id = str(_read_field(drone, "home_id", "") or "")
    if home_id and home_id in (_mapping(mgr, "depots") or {}):
        return "DEPOT", home_id, None
    if home_id and home_id in (_mapping(mgr, "stations") or {}):
        return "STATION", home_id, None

    depot_ids = list((_mapping(mgr, "depots") or {}).keys())
    if depot_ids:
        return "DEPOT", str(depot_ids[0]), None
    return "", "", None


def _pending_route_recovery_host(snapshot: Any, drone: Any) -> tuple[str, str] | None:
    route_plan = list(_read_field(drone, "route_plan", []) or [])
    if not route_plan:
        return None

    start_idx = int(_read_field(drone, "current_waypoint_index", 0) or 0)
    start_idx = max(0, min(start_idx, len(route_plan)))
    for wp in route_plan[start_idx:]:
        action_name = _waypoint_action_name(_read_field(wp, "action"))
        if action_name not in {"DOCK_TRUCK", "DOCK_DEPOT"}:
            continue
        node_id = str(_read_field(wp, "target_entity_id", "") or "")
        if not node_id:
            node_id = _nearest_support_node_id(snapshot, _read_field(wp, "loc"))
        host_type = _node_host_type(snapshot, node_id)
        if host_type in {"DEPOT", "STATION"}:
            return host_type, node_id
    return None


def _waypoint_action_name(action: Any) -> str:
    if action is None:
        return ""
    if hasattr(action, "value"):
        action = action.value
    return str(action).strip().upper()


def _node_host_type(snapshot: Any, node_id: str) -> str:
    if not node_id:
        return ""
    mgr = _entity_mgr(snapshot)
    if node_id in (_mapping(mgr, "stations") or {}):
        return "STATION"
    if node_id in (_mapping(mgr, "depots") or {}) or _is_depot_node_id(node_id):
        return "DEPOT"
    return ""


def _nearest_support_node_id(snapshot: Any, pos: Any) -> str:
    if pos is None:
        return ""
    mgr = _entity_mgr(snapshot)
    best_id = ""
    best_dist = math.inf
    for mapping_name in ("stations", "depots"):
        for node_id, node in (_mapping(mgr, mapping_name) or {}).items():
            dist = _distance(pos, _read_field(node, "location"))
            if dist < best_dist:
                best_dist = dist
                best_id = str(node_id)
    return best_id if best_dist <= 60.0 else ""


def _is_depot_node_id(node_id: str) -> bool:
    normalized = str(node_id).strip().upper()
    return normalized == "DEPOT" or normalized.startswith("DEPOT") or normalized.startswith("DEP-")


def _add_drone_to_future_host(snapshot: Any, drone_id: str, host_type: str, host_node_id: str, truck_id: str | None) -> None:
    mgr = _entity_mgr(snapshot)
    if host_type == "TRUCK" and truck_id:
        truck = (_mapping(mgr, "trucks") or {}).get(truck_id)
        if truck is not None:
            _append_unique_field(truck, "_ga_docked_drones", drone_id)
        return
    if host_type == "DEPOT":
        depot = (_mapping(mgr, "depots") or {}).get(host_node_id)
        if depot is not None:
            _append_unique_field(depot, "_ga_idle_drones", drone_id)
        return
    if host_type == "STATION":
        station = (_mapping(mgr, "stations") or {}).get(host_node_id)
        if station is not None:
            _append_unique_field(station, "_ga_waiting_drones", drone_id)


def _append_unique_field(record: Any, field_name: str, value: str) -> None:
    items = list(_read_field(record, field_name, []) or [])
    if value not in items:
        items.append(value)
    _write_field(record, field_name, items)


def _classify_orders(state: Any, new_orders: dict[str, Any], event_time: float) -> OrderBuckets:
    known = _all_known_orders(state)
    known.update(new_orders)
    new_ids = set(new_orders)
    completed: dict[str, Any] = {}
    locked: dict[str, Any] = {}
    pending: dict[str, Any] = {}
    new: dict[str, Any] = {}

    running_ids = _running_order_ids(state)
    for oid, order in known.items():
        status = _status_name(_read_field(order, "status"))
        if status in {"COMPLETED", "REJECTED", "CANCELLED", "CANCELED"}:
            completed[oid] = order
            continue
        if oid in running_ids or status in {"PICKED_UP", "DELIVERING"}:
            locked[oid] = order
            continue
        if oid in new_ids:
            new[oid] = order
        else:
            pending[oid] = order

    return OrderBuckets(completed=completed, locked=locked, pending=pending, new=new)


def _select_reoptimization_window(
    previous_best: Individual | None,
    pending_orders: dict[str, Any],
    new_orders: dict[str, Any],
    config: GAConfig,
    event_time: float,
) -> tuple[list[str], list[str]]:
    new_ids = list(new_orders)
    pending_ids = list(pending_orders)
    if bool(getattr(config, "dynamic_preserve_previous_best", True)):
        neighbor_count = max(0, int(getattr(config, "dynamic_neighbor_reopt_count", 0) or 0))
        selected_pending = _select_neighbor_pending_orders(
            previous_best=previous_best,
            pending_orders=pending_orders,
            new_orders=new_orders,
            count=neighbor_count,
            event_time=event_time,
        )
        frozen_future = [oid for oid in pending_ids if oid not in set(selected_pending)]
        return list(dict.fromkeys(new_ids + selected_pending)), frozen_future

    previous_rank = {oid: i for i, oid in enumerate(previous_best.sequence)} if previous_best is not None else {}

    def sort_key(order_id: str) -> tuple[float, float, str]:
        order = pending_orders.get(order_id)
        rank = previous_rank.get(order_id, 10**9)
        deadline = _safe_float(_read_field(order, "deadline", math.inf), math.inf)
        return (rank, deadline - event_time, order_id)

    ordered_pending = sorted(pending_ids, key=sort_key)
    k = int(config.reopt_window_size or 0)
    if k <= 0:
        selected_pending = ordered_pending
    else:
        selected_pending = ordered_pending[:k]
    frozen_future = [oid for oid in ordered_pending if oid not in set(selected_pending)]
    reoptimized = list(dict.fromkeys(new_ids + selected_pending))
    return reoptimized, frozen_future


def _select_neighbor_pending_orders(
    previous_best: Individual | None,
    pending_orders: dict[str, Any],
    new_orders: dict[str, Any],
    count: int,
    event_time: float,
) -> list[str]:
    if count <= 0 or not pending_orders:
        return []
    previous_rank = {oid: i for i, oid in enumerate(previous_best.sequence)} if previous_best is not None else {}
    new_positions = [_read_field(order, "delivery_loc") for order in new_orders.values()]

    def sort_key(order_id: str) -> tuple[float, float, float, str]:
        order = pending_orders.get(order_id)
        pos = _read_field(order, "delivery_loc")
        nearest_new = min((_distance(pos, new_pos) for new_pos in new_positions), default=math.inf)
        rank = float(previous_rank.get(order_id, 10**9))
        deadline = _safe_float(_read_field(order, "deadline", math.inf), math.inf)
        return (nearest_new, rank, deadline - event_time, order_id)

    return sorted(pending_orders, key=sort_key)[:count]


def _tail_gene_map(
    previous_best: Individual | None,
    frozen_future_order_ids: list[str],
    gene_pool: list[str],
) -> dict[str, tuple[str, dict[str, str] | None]]:
    allowed = set(gene_pool)
    result: dict[str, tuple[str, dict[str, str] | None]] = {}
    if previous_best is not None:
        for oid, gene, rv in zip(previous_best.sequence, previous_best.assignment, previous_best.rendezvous):
            if oid in frozen_future_order_ids and gene in allowed:
                result[oid] = (gene, copy.deepcopy(rv))
    for oid in frozen_future_order_ids:
        result.setdefault(oid, ("A", None))
    return result


def _fast_incumbent_plan(
    solver: Any,
    snapshot: Any,
    previous_best: Individual | None,
    previous_plan: Any,
    seed_candidates: list[Individual],
    config: GAConfig,
    planning_orders: dict[str, Any],
    new_order_ids: list[str],
    context: Any,
) -> tuple[DispatchPlan | None, dict[str, Any], list[Individual]]:
    max_eval = max(0, int(getattr(config, "fast_incumbent_eval_count", 8) or 0))
    budget_seconds = max(0.0, float(getattr(config, "fast_incumbent_budget_seconds", 0.3) or 0.0))
    diag: dict[str, Any] = {
        "fast_incumbent_eval_count": 0,
        "fast_incumbent_cost": 0.0,
        "fast_incumbent_A_count": 0,
        "fast_incumbent_B_count": 0,
        "fast_incumbent_C_count": 0,
    }
    if not bool(getattr(config, "fast_incumbent_enabled", True)) or max_eval <= 0:
        return None, diag, []

    fast_seeds = _build_fast_incumbent_seeds(
        seed_candidates=seed_candidates,
        new_order_ids=new_order_ids,
        gene_pool=context.gene_pool,
        depot_ids=context.depot_ids,
        station_ids=context.station_ids,
        order_ids=list(planning_orders),
        allow_c_recover_station=config.allow_depot_drone_recover_at_station,
        max_count=max_eval * 3,
    )
    if not fast_seeds:
        if bool(getattr(config, "dynamic_allow_new_order_pending", True)) and previous_plan is not None:
            pending_plan = _previous_plan_fallback(
                previous_plan,
                planning_orders,
                new_order_ids=set(new_order_ids),
                reason="dynamic_new_order_pending",
            )
            _fill_fast_incumbent_diag(diag, pending_plan, eval_count=0)
            return pending_plan, diag, []
        return None, diag, []

    previous_config = solver.config
    previous_evaluator_config = solver.evaluator.config
    previous_decoder_config = solver.decoder.config
    solver.config = config
    solver.evaluator.config = config
    solver.decoder.config = config

    started = time.time()
    best_seed: Individual | None = None
    best_plan: DispatchPlan | None = None
    best_key: tuple[float, float, float, float, float, float, float] | None = None
    eval_count = 0
    try:
        for seed in fast_seeds:
            if eval_count >= max_eval:
                break
            if budget_seconds > 0.0 and time.time() - started >= budget_seconds:
                break
            candidate = copy.deepcopy(seed)
            try:
                enforce_fixed_tail(candidate, context.fixed_tail_order_ids, context.fixed_tail_gene_by_order)
                result = solver._evaluate_individual(candidate, snapshot, context)
            except Exception:
                continue
            eval_count += 1
            plan = getattr(candidate, "decoded_plan", None)
            if result is None or plan is None or not _old_orders_feasible(plan, planning_orders, set(new_order_ids)):
                continue
            stability = _stability_diagnostics(
                previous_best=previous_best,
                plan=plan,
                planning_orders=planning_orders,
                new_order_ids=set(new_order_ids),
            )
            key = (
                float(stability.get("old_assignment_changes", 0)),
                float(stability.get("old_rendezvous_changes", 0)),
                float(stability.get("old_sequence_inversions", 0)),
                float(stability.get("bc_drop_count", 0)),
                float(stability.get("new_order_pending_count", 0)),
                -float(stability.get("candidate_bc_count", 0)),
                float(getattr(candidate, "fitness", math.inf)),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_seed = candidate
                best_plan = plan
    finally:
        solver.config = previous_config
        solver.evaluator.config = previous_evaluator_config
        solver.decoder.config = previous_decoder_config

    if best_plan is None and bool(getattr(config, "dynamic_allow_new_order_pending", True)) and previous_plan is not None:
        best_plan = _previous_plan_fallback(
            previous_plan,
            planning_orders,
            new_order_ids=set(new_order_ids),
            reason="dynamic_new_order_pending",
        )
    if best_plan is not None:
        _fill_fast_incumbent_diag(diag, best_plan, eval_count=eval_count)
        best_plan.summary.setdefault("fallback_used", False)
        best_plan.summary["fallback_level"] = "fast_incumbent"
    return best_plan, diag, [best_seed] if best_seed is not None else fast_seeds[:max_eval]


def _build_fast_incumbent_seeds(
    seed_candidates: list[Individual],
    new_order_ids: list[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    order_ids: list[str],
    allow_c_recover_station: bool,
    max_count: int,
) -> list[Individual]:
    if max_count <= 0:
        return []
    candidates: list[Individual] = []
    new_set = set(str(oid) for oid in new_order_ids)
    preferred_genes = [
        gene
        for gene in (
            _first_gene_for_mode(gene_pool, "C"),
            _first_gene_for_mode(gene_pool, "B"),
            _first_gene_for_mode(gene_pool, "A"),
        )
        if gene is not None
    ]
    for seed in seed_candidates:
        candidates.append(copy.deepcopy(seed))
        for gene in preferred_genes:
            forced = copy.deepcopy(seed)
            for idx, oid in enumerate(forced.sequence):
                if oid not in new_set:
                    continue
                forced.assignment[idx] = gene
                forced.rendezvous[idx] = make_random_rendezvous_for_gene(
                    gene,
                    depot_ids,
                    station_ids,
                    allow_c_recover_station,
                )
            candidates.append(forced)
            if len(candidates) >= max_count * 2:
                break
        if len(candidates) >= max_count * 2:
            break

    return _dedupe_and_validate_seeds(
        candidates,
        order_ids=order_ids,
        gene_pool=gene_pool,
        depot_ids=depot_ids,
        station_ids=station_ids,
        allow_c_recover_station=allow_c_recover_station,
        max_count=max_count,
    )


def _fill_fast_incumbent_diag(diag: dict[str, Any], plan: DispatchPlan, eval_count: int) -> None:
    modes = _mode_counts(getattr(plan, "allocations", []) or [])
    diag.update(
        {
            "fast_incumbent_eval_count": int(eval_count),
            "fast_incumbent_cost": float(getattr(plan, "cost_total", 0.0) or 0.0),
            "fast_incumbent_A_count": int(modes.get("A", 0)),
            "fast_incumbent_B_count": int(modes.get("B", 0)),
            "fast_incumbent_C_count": int(modes.get("C", 0)),
        }
    )


def _old_orders_feasible(plan: Any, planning_orders: dict[str, Any], new_order_ids: set[str]) -> bool:
    old_ids = set(planning_orders) - set(new_order_ids)
    feasible_allocs = {
        str(_read_field(alloc, "order_id", ""))
        for alloc in getattr(plan, "allocations", []) or []
        if bool(_read_field(alloc, "feasible", False))
    }
    return old_ids <= feasible_allocs


def _fast_incumbent_can_skip_ga(
    plan: DispatchPlan | None,
    diag: dict[str, Any],
    planning_orders: dict[str, Any],
    config: GAConfig,
) -> bool:
    if plan is None or not bool(getattr(config, "dynamic_skip_ga_when_fast_incumbent_safe", True)):
        return False
    return (
        _plan_is_feasible_for_orders(plan, planning_orders)
        and int(diag.get("old_assignment_changes", 0) or 0) == 0
        and int(diag.get("old_rendezvous_changes", 0) or 0) == 0
        and int(diag.get("old_sequence_inversions", 0) or 0) == 0
        and int(diag.get("bc_drop_count", 0) or 0) == 0
        and int(diag.get("new_order_pending_count", 0) or 0) == 0
    )


def _remember_fast_incumbent_on_solver(
    solver: Any,
    seeds: list[Individual],
    plan: DispatchPlan | None,
    context: Any,
) -> None:
    if not seeds:
        return
    best = copy.deepcopy(seeds[0])
    best.decoded_plan = plan
    setattr(best, "decoded_result", SimpleNamespace(plan=plan))
    try:
        solver.last_best_individual = copy.deepcopy(best)
        solver.last_best_decode_result = SimpleNamespace(plan=plan)
        remember = getattr(solver, "_remember_dynamic_population", None)
        if callable(remember):
            remember([best], context, source="fast_incumbent")
    except Exception:
        return


def _stability_diagnostics(
    previous_best: Individual | None,
    plan: Any,
    planning_orders: dict[str, Any],
    new_order_ids: set[str],
) -> dict[str, Any]:
    candidate_modes = _mode_counts(getattr(plan, "allocations", []) or []) if plan is not None else {}
    candidate_bc_count = int(candidate_modes.get("B", 0) or 0) + int(candidate_modes.get("C", 0) or 0)
    if previous_best is None:
        return {
            "old_assignment_changes": 0,
            "old_rendezvous_changes": 0,
            "old_sequence_inversions": 0,
            "bc_drop_count": 0,
            "previous_bc_count": 0,
            "candidate_bc_count": candidate_bc_count,
            "new_order_pending_count": len(_unserved_order_ids(plan, planning_orders)) if plan is not None else len(new_order_ids),
        }

    planning_set = set(planning_orders)
    old_ids = [oid for oid in previous_best.sequence if oid in planning_set and oid not in new_order_ids]
    old_id_set = set(old_ids)
    previous_mode = {
        oid: _gene_mode_family(gene)
        for oid, gene in zip(previous_best.sequence, previous_best.assignment)
        if oid in old_id_set
    }
    previous_rv = {
        oid: copy.deepcopy(rv)
        for oid, rv in zip(previous_best.sequence, previous_best.rendezvous)
        if oid in old_id_set
    }
    allocs = {
        str(_read_field(alloc, "order_id", "")): alloc
        for alloc in getattr(plan, "allocations", []) or []
    } if plan is not None else {}

    assignment_changes = 0
    rendezvous_changes = 0
    previous_bc_count = 0
    current_bc_count = 0
    for oid in old_ids:
        prev_mode = previous_mode.get(oid, "A")
        if prev_mode in {"B", "C"}:
            previous_bc_count += 1
        alloc = allocs.get(oid)
        plan_mode = _allocation_mode_family(alloc)
        if plan_mode in {"B", "C"}:
            current_bc_count += 1
        if plan_mode != prev_mode:
            assignment_changes += 1
            continue
        if prev_mode in {"B", "C"} and _rendezvous_changed(previous_rv.get(oid), alloc, prev_mode):
            rendezvous_changes += 1

    plan_old_order = [
        str(_read_field(alloc, "order_id", ""))
        for alloc in getattr(plan, "allocations", []) or []
        if str(_read_field(alloc, "order_id", "")) in old_id_set
    ] if plan is not None else []
    previous_rank = {oid: idx for idx, oid in enumerate(old_ids)}
    inversions = 0
    for i in range(len(plan_old_order)):
        for j in range(i + 1, len(plan_old_order)):
            if previous_rank.get(plan_old_order[i], 10**9) > previous_rank.get(plan_old_order[j], 10**9):
                inversions += 1

    unserved_new = [
        oid
        for oid in new_order_ids
        if oid not in allocs or not bool(_read_field(allocs.get(oid), "feasible", False))
    ]
    return {
        "old_assignment_changes": int(assignment_changes),
        "old_rendezvous_changes": int(rendezvous_changes),
        "old_sequence_inversions": int(inversions),
        "bc_drop_count": int(max(0, previous_bc_count - current_bc_count)),
        "previous_bc_count": int(previous_bc_count),
        "candidate_bc_count": int(candidate_bc_count),
        "new_order_pending_count": int(len(unserved_new)),
    }


def _attach_archive_stability_fields(diag: dict[str, Any], archive_bc_available: bool) -> None:
    previous_bc_count = int(diag.get("previous_bc_count", 0) or 0)
    candidate_bc_count = int(diag.get("candidate_bc_count", 0) or 0)
    bc_drop_count = int(diag.get("bc_drop_count", 0) or 0)
    all_a_rejected = bool((archive_bc_available or previous_bc_count > 0) and (candidate_bc_count == 0 or bc_drop_count > 0))
    diag["archive_bc_available"] = bool(archive_bc_available)
    diag["all_a_rejected"] = all_a_rejected


def _ga_can_replace_fast_incumbent(ga_diag: dict[str, Any], fast_diag: dict[str, Any], config: GAConfig) -> bool:
    if not bool(getattr(config, "dynamic_preserve_previous_best", True)):
        return int(ga_diag.get("new_order_pending_count", 0) or 0) <= int(fast_diag.get("new_order_pending_count", 0) or 0)
    if bool(ga_diag.get("all_a_rejected", False)):
        return False
    return (
        int(ga_diag.get("old_assignment_changes", 0) or 0) == 0
        and int(ga_diag.get("old_rendezvous_changes", 0) or 0) == 0
        and int(ga_diag.get("old_sequence_inversions", 0) or 0) == 0
        and int(ga_diag.get("bc_drop_count", 0) or 0) == 0
        and int(ga_diag.get("new_order_pending_count", 0) or 0) <= int(fast_diag.get("new_order_pending_count", 0) or 0)
    )


def _gene_mode_family(gene: str) -> str:
    if gene == "A":
        return "A"
    if str(gene).startswith("B_"):
        return "B"
    if str(gene).startswith("C_"):
        return "C"
    return "?"


def _allocation_mode_family(alloc: Any) -> str:
    if alloc is None or not bool(_read_field(alloc, "feasible", False)):
        return "UNSERVED"
    mode = str(_read_field(alloc, "mode", "") or "")
    if mode.startswith("B"):
        return "B"
    if mode == "C":
        return "C"
    if mode == "A":
        return "A"
    return mode or "UNSERVED"


def _rendezvous_changed(previous_rv: Any, alloc: Any, mode: str) -> bool:
    if not isinstance(previous_rv, dict) or alloc is None:
        return True
    previous_launch = str(previous_rv.get("launch", "") or "")
    previous_recover = str(previous_rv.get("recover", "") or "")
    current_launch = str(_read_field(alloc, "launch_station_id", "") or "")
    current_recover = str(_read_field(alloc, "recovery_station_id", "") or "")
    if mode == "B":
        return bool(previous_launch and previous_launch != current_launch) or bool(previous_recover and previous_recover != current_recover)
    if mode == "C":
        return bool(previous_recover and previous_recover != current_recover)
    return False


def _best_feasible_warm_start_plan(
    solver: Any,
    snapshot: Any,
    warm_starts: list[Individual],
    config: GAConfig,
    planning_orders: dict[str, Any],
) -> DispatchPlan | None:
    if not warm_starts:
        return None
    previous_config = solver.config
    previous_evaluator_config = solver.evaluator.config
    previous_decoder_config = solver.decoder.config
    solver.config = config
    solver.evaluator.config = config
    solver.decoder.config = config
    try:
        context = build_ga_context(snapshot, config, mode="dynamic")
        best_plan = None
        best_fitness = math.inf
        for seed in warm_starts:
            candidate = copy.deepcopy(seed)
            try:
                enforce_fixed_tail(candidate, context.fixed_tail_order_ids, context.fixed_tail_gene_by_order)
                result = solver._evaluate_individual(candidate, snapshot, context)
            except Exception:
                continue
            plan = getattr(candidate, "decoded_plan", None)
            if result is None or plan is None or not _plan_is_feasible_for_orders(plan, planning_orders):
                continue
            fitness = float(getattr(candidate, "fitness", math.inf))
            if fitness < best_fitness:
                best_fitness = fitness
                best_plan = plan
        if best_plan is not None:
            best_plan.summary["fallback_used"] = True
            best_plan.summary["fallback_level"] = "warm_start_repaired"
        return best_plan
    finally:
        solver.config = previous_config
        solver.evaluator.config = previous_evaluator_config
        solver.decoder.config = previous_decoder_config


def _greedy_replan_fallback(
    solver: Any,
    snapshot: Any,
    planning_orders: dict[str, Any],
    event_time: float,
) -> DispatchPlan | None:
    bbox = _read_field(snapshot, "bbox")
    if not bbox:
        return None
    try:
        return solver.greedy_helper.dispatch_replan_current_state(
            planning_orders,
            float(event_time),
            bbox,
            scene_id=_read_field(snapshot, "scene_id"),
        )
    except Exception:
        return None


def _previous_plan_fallback(
    previous_plan: Any,
    planning_orders: dict[str, Any],
    new_order_ids: set[str],
    reason: str,
) -> DispatchPlan:
    allocations: list[AllocationResult] = []
    previous_allocs = {
        str(_read_field(alloc, "order_id", "")): copy.deepcopy(alloc)
        for alloc in getattr(previous_plan, "allocations", []) or []
    }
    for order_id in planning_orders:
        alloc = previous_allocs.get(order_id)
        if alloc is not None and order_id not in new_order_ids:
            allocations.append(alloc)
            continue
        allocations.append(
            AllocationResult(
                order_id=order_id,
                vehicle_id="",
                mode="UNSERVED",
                distance=0.0,
                feasible=False,
                reason=reason if order_id in new_order_ids else "previous_plan_missing",
            )
        )

    truck_routes = copy.deepcopy(getattr(previous_plan, "truck_routes", {}) or {})
    drone_routes = copy.deepcopy(getattr(previous_plan, "drone_routes", {}) or {})
    feasible_count = sum(1 for alloc in allocations if bool(_read_field(alloc, "feasible", False)))
    return DispatchPlan(
        allocations=allocations,
        cost_total=float(getattr(previous_plan, "cost_total", 0.0) or 0.0) if previous_plan is not None else 0.0,
        summary={
            "total_orders": len(planning_orders),
            "feasible": feasible_count,
            "modes": _mode_counts(allocations),
            "dispatch_type": "dynamic_replan",
            "solver": "ga_mmce",
            "ga_feasible": False,
            "new_orders_unserved": sorted(new_order_ids),
        },
        truck_routes=truck_routes,
        drone_routes=drone_routes,
    )


def _apply_ga_route_patch(
    plan: DispatchPlan,
    reoptimized_ids: list[str],
    new_order_ids: list[str],
    frozen_future_ids: list[str],
    locked_ids: set[str],
    enabled: bool,
) -> None:
    frozen_set = {str(order_id) for order_id in frozen_future_ids if str(order_id)}
    locked_set = {str(order_id) for order_id in locked_ids if str(order_id)}
    allowed_ids = list(dict.fromkeys(
        str(order_id)
        for order_id in list(reoptimized_ids) + list(new_order_ids)
        if str(order_id)
    ))
    feasible_allowed_ids = {
        str(getattr(alloc, "order_id", "") or "")
        for alloc in getattr(plan, "allocations", []) or []
        if bool(getattr(alloc, "feasible", False))
        and str(getattr(alloc, "order_id", "") or "") in set(allowed_ids)
    }
    replace_ids = [
        order_id
        for order_id in allowed_ids
        if order_id in feasible_allowed_ids
        and order_id not in locked_set
        and order_id not in frozen_set
    ]
    replace_seen = set(replace_ids)
    forced_b_wait_ids: list[str] = []
    allowed_set = set(allowed_ids)
    for alloc in getattr(plan, "allocations", []) or []:
        order_id = str(getattr(alloc, "order_id", "") or "")
        if not order_id or order_id not in allowed_set:
            continue
        if order_id in locked_set or order_id in frozen_set:
            continue
        if not bool(getattr(alloc, "feasible", False)):
            continue
        mode = str(getattr(alloc, "mode", "") or "")
        if mode not in {"B", "B_WAIT"}:
            continue
        if order_id not in replace_seen:
            replace_ids.append(order_id)
            replace_seen.add(order_id)
        forced_b_wait_ids.append(order_id)
    replace_set = set(replace_ids)
    plan.summary["ga_route_patch_enabled"] = bool(enabled)
    plan.summary["ga_route_patch_replace_order_ids"] = list(replace_ids)
    plan.summary["ga_route_patch_b_wait_order_ids"] = list(dict.fromkeys(forced_b_wait_ids))
    plan.summary["ga_route_patch_anchor_by_order"] = _route_patch_anchor_by_order(plan, replace_set)
    plan.summary["ga_route_patch"] = {
        "enabled": bool(enabled),
        "replace_order_ids": list(replace_ids),
        "b_wait_order_ids": list(dict.fromkeys(forced_b_wait_ids)),
        "anchor_by_order": copy.deepcopy(plan.summary["ga_route_patch_anchor_by_order"]),
    }
    if not enabled or not replace_set:
        return

    for route in (getattr(plan, "truck_routes", {}) or {}).values():
        original_nodes = list(getattr(route, "nodes", []) or [])
        if not original_nodes:
            continue
        filtered_nodes = []
        for idx, node in enumerate(original_nodes):
            order_id = str(getattr(node, "order_id", "") or "")
            if idx == 0:
                filtered_nodes.append(node)
                continue
            if order_id in replace_set:
                filtered_nodes.append(node)
        route.nodes = filtered_nodes
        try:
            route.geometry = [node.position for node in filtered_nodes if getattr(node, "position", None) is not None]
        except Exception:
            route.geometry = []


def _route_patch_anchor_by_order(plan: DispatchPlan, replace_ids: set[str]) -> dict[str, dict[str, str]]:
    anchors: dict[str, dict[str, str]] = {}
    ordered_alloc_ids = [
        str(_read_field(alloc, "order_id", "") or "")
        for alloc in getattr(plan, "allocations", []) or []
    ]
    for idx, order_id in enumerate(ordered_alloc_ids):
        if order_id not in replace_ids:
            continue
        before = ""
        after = ""
        for prev in reversed(ordered_alloc_ids[:idx]):
            if prev and prev not in replace_ids:
                after = prev
                break
        for nxt in ordered_alloc_ids[idx + 1:]:
            if nxt and nxt not in replace_ids:
                before = nxt
                break
        anchors[order_id] = {
            "after_order_id": after,
            "before_order_id": before,
        }
    return anchors


def _annotate_dynamic_summary(
    plan: DispatchPlan,
    event_time: float,
    buckets: OrderBuckets,
    reoptimized_ids: list[str],
    frozen_future_ids: list[str],
    warm_start_count: int,
    config: GAConfig,
    ga_plan: DispatchPlan,
    ga_feasible: bool,
    warm_start_feasible: bool,
    greedy_insert_feasible: bool,
    fallback_used: bool,
    fallback_level: str,
    unserved_order_ids: list[str],
    elapsed_seconds: float,
    selection_diag: dict[str, Any] | None = None,
) -> None:
    modes = _mode_counts(plan.allocations)
    ga_summary = getattr(ga_plan, "summary", {}) or {}
    selection_diag = dict(selection_diag or {})
    total_orders = len(reoptimized_ids) + len(frozen_future_ids)
    feasible_count = sum(1 for allocation in plan.allocations if bool(_read_field(allocation, "feasible", False)))
    final_feasible = feasible_count == total_orders and not unserved_order_ids
    plan.summary.update(
        {
            "total_orders": int(total_orders),
            "feasible": int(feasible_count),
            "event_time": float(event_time),
            "new_order_ids": list(buckets.new),
            "completed_ids": list(buckets.completed),
            "locked_ids": list(buckets.locked),
            "pending_count": len(buckets.pending),
            "reoptimized_order_count": len(reoptimized_ids),
            "frozen_future_order_count": len(frozen_future_ids),
            "reoptimized_order_ids": list(reoptimized_ids),
            "frozen_future_order_ids": list(frozen_future_ids),
            "warm_start_count": int(warm_start_count),
            "population_size": int(config.population_size),
            "max_generations": int(config.generations),
            "actual_generations": int(ga_summary.get("actual_generations", 0) or 0),
            "elapsed_seconds": float(elapsed_seconds),
            "time_budget_hit": bool(ga_summary.get("time_budget_hit", False)),
            "early_stop_triggered": bool(ga_summary.get("early_stop_triggered", False)),
            "fallback_used": bool(fallback_used),
            "fallback_level": fallback_level,
            "ga_feasible": bool(final_feasible),
            "ga_plan_feasible": bool(ga_feasible),
            "warm_start_feasible": bool(warm_start_feasible),
            "greedy_insert_feasible": bool(greedy_insert_feasible),
            "unserved_order_ids": list(unserved_order_ids),
            "modes": modes,
            "final_A_count": int(modes.get("A", 0)),
            "final_B_count": int(modes.get("B", 0)),
            "final_C_count": int(modes.get("C", 0)),
            "best_fitness": float(plan.cost_total or ga_summary.get("best_fitness", 0.0) or 0.0),
            "fast_incumbent_found": bool(selection_diag.get("fast_incumbent_found", False)),
            "fast_incumbent_selected": bool(selection_diag.get("fast_incumbent_selected", False)),
            "fast_incumbent_eval_count": int(selection_diag.get("fast_incumbent_eval_count", 0) or 0),
            "fast_incumbent_cost": float(selection_diag.get("fast_incumbent_cost", 0.0) or 0.0),
            "fast_incumbent_A_count": int(selection_diag.get("fast_incumbent_A_count", 0) or 0),
            "fast_incumbent_B_count": int(selection_diag.get("fast_incumbent_B_count", 0) or 0),
            "fast_incumbent_C_count": int(selection_diag.get("fast_incumbent_C_count", 0) or 0),
            "archive_seed_count": int(selection_diag.get("archive_seed_count", 0) or 0),
            "archive_seed_B_count": int(selection_diag.get("archive_seed_B_count", 0) or 0),
            "archive_seed_C_count": int(selection_diag.get("archive_seed_C_count", 0) or 0),
            "warm_start_with_B": int(selection_diag.get("warm_start_with_B", 0) or 0),
            "warm_start_with_C": int(selection_diag.get("warm_start_with_C", 0) or 0),
            "force_b_repair_seed_count": int(selection_diag.get("force_b_repair_seed_count", 0) or 0),
            "dynamic_archive_size": int(selection_diag.get("dynamic_archive_size", 0) or 0),
            "old_assignment_changes": int(selection_diag.get("old_assignment_changes", 0) or 0),
            "old_rendezvous_changes": int(selection_diag.get("old_rendezvous_changes", 0) or 0),
            "old_sequence_inversions": int(selection_diag.get("old_sequence_inversions", 0) or 0),
            "bc_drop_count": int(selection_diag.get("bc_drop_count", 0) or 0),
            "previous_bc_count": int(selection_diag.get("previous_bc_count", 0) or 0),
            "candidate_bc_count": int(selection_diag.get("candidate_bc_count", 0) or 0),
            "archive_bc_available": bool(selection_diag.get("archive_bc_available", False)),
            "all_a_rejected": bool(selection_diag.get("all_a_rejected", False)),
            "new_order_pending_count": int(selection_diag.get("new_order_pending_count", 0) or 0),
            "plan_selection_reason": str(selection_diag.get("plan_selection_reason", "") or ""),
            "dispatch_type": "dynamic_replan",
            "solver": "ga_mmce",
        }
    )


def _finalize_and_log(
    plan: DispatchPlan,
    started: float,
    event_time: float,
    buckets: OrderBuckets,
    reoptimized_ids: list[str],
    frozen_future_ids: list[str],
    warm_start_count: int,
    config: GAConfig,
    fallback_used: bool,
    fallback_level: str,
) -> None:
    _annotate_dynamic_summary(
        plan,
        event_time=event_time,
        buckets=buckets,
        reoptimized_ids=reoptimized_ids,
        frozen_future_ids=frozen_future_ids,
        warm_start_count=warm_start_count,
        config=config,
        ga_plan=plan,
        ga_feasible=True,
        warm_start_feasible=False,
        greedy_insert_feasible=False,
        fallback_used=fallback_used,
        fallback_level=fallback_level,
        unserved_order_ids=[],
        elapsed_seconds=time.time() - started,
    )
    _write_dynamic_replan_csv(plan.summary)
    _store_last_stats(plan.summary)


def _write_dynamic_replan_csv(summary: dict[str, Any]) -> None:
    path = PROJECT_ROOT / "logs" / "ga_dynamic_replan.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    needs_header = not exists or _csv_header_mismatch(path, DYNAMIC_REPLAN_FIELDS)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DYNAMIC_REPLAN_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: _csv_value(summary.get(field, "")) for field in DYNAMIC_REPLAN_FIELDS})


def _csv_header_mismatch(path: Path, fields: list[str]) -> bool:
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            rows = [row for row in reader]
        if not rows:
            return True
        return all(list(row) != list(fields) for row in rows)
    except Exception:
        return True


def _store_last_stats(summary: dict[str, Any]) -> None:
    global LAST_DYNAMIC_REPLAN_STATS
    LAST_DYNAMIC_REPLAN_STATS = dict(summary)


def _debug_dynamic_stability(summary: dict[str, Any]) -> None:
    path = PROJECT_ROOT / "backend" / "solver" / "ga_mmce_debug_log"
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                "dynamic_stability "
                f"old_assignment_changes={int(summary.get('old_assignment_changes', 0) or 0)} "
                f"old_rendezvous_changes={int(summary.get('old_rendezvous_changes', 0) or 0)} "
                f"old_sequence_inversions={int(summary.get('old_sequence_inversions', 0) or 0)} "
                f"bc_drop_count={int(summary.get('bc_drop_count', 0) or 0)} "
                f"previous_bc_count={int(summary.get('previous_bc_count', 0) or 0)} "
                f"candidate_bc_count={int(summary.get('candidate_bc_count', 0) or 0)} "
                f"archive_bc_available={bool(summary.get('archive_bc_available', False))} "
                f"all_a_rejected={bool(summary.get('all_a_rejected', False))} "
                f"archive_seed_B_count={int(summary.get('archive_seed_B_count', 0) or 0)} "
                f"archive_seed_C_count={int(summary.get('archive_seed_C_count', 0) or 0)} "
                f"warm_start_with_B={int(summary.get('warm_start_with_B', 0) or 0)} "
                f"warm_start_with_C={int(summary.get('warm_start_with_C', 0) or 0)} "
                f"new_order_pending_count={int(summary.get('new_order_pending_count', 0) or 0)} "
                f"selected={summary.get('plan_selection_reason', '')}\n"
            )
    except Exception:
        return


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return repr(value)
    return value


def _resolve_solver(state: Any) -> Any:
    solver = _read_field(state, "_ga_solver") or _read_field(state, "solver")
    if solver is not None:
        return solver
    from .solver import GAMMCESolver

    return GAMMCESolver(_entity_mgr(state), config=make_ga_config(DYNAMIC_GA_CONFIG))


def _all_known_orders(state: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    direct = _read_field(state, "orders")
    if isinstance(direct, dict):
        result.update({str(k): v for k, v in direct.items()})

    order_mgr = _order_mgr(state)
    if order_mgr is not None:
        for field_name in ("pending_orders", "assigned_orders"):
            value = _read_field(order_mgr, field_name, {}) or {}
            if isinstance(value, dict):
                result.update({str(k): v for k, v in value.items()})
        for order in _read_field(order_mgr, "completed_orders", []) or []:
            oid = _order_id(order)
            if oid:
                result[oid] = order
    return result


def _running_order_ids(state: Any) -> set[str]:
    mgr = _entity_mgr(state)
    running: set[str] = set()
    for drone in (_mapping(mgr, "drones") or {}).values():
        carrying = str(_read_field(drone, "carrying_order_id", "") or "")
        if carrying:
            running.add(carrying)
    for truck in (_mapping(mgr, "trucks") or {}).values():
        for stop in getattr(truck, "_planned_route_stops", []) or []:
            oid = str(stop.get("order_id", "") or "")
            if oid and float(stop.get("arrival_time", math.inf)) <= float(_read_field(state, "current_time", 0.0) or 0.0):
                running.add(oid)
    return running


def _normalize_order_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    result = {}
    for order in value:
        oid = _order_id(order)
        if oid:
            result[oid] = order
    return result


def _ordered_order_mapping(order_ids: list[str], source: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for order_id in order_ids:
        if order_id in source:
            result[order_id] = source[order_id]
    return result


def _previous_remaining_individual(
    previous_best: Individual | None,
    order_ids: list[str],
    excluded: set[str],
    gene_pool: list[str],
    support_node_ids: list[str],
    allow_c_recover_station: bool,
) -> Individual | None:
    order_set = set(order_ids)
    gene_set = set(gene_pool)
    seq: list[str] = []
    assignment: list[str] = []
    rendezvous = []
    if previous_best is not None:
        for oid, gene, rv in zip(previous_best.sequence, previous_best.assignment, previous_best.rendezvous):
            if oid in excluded or oid not in order_set:
                continue
            seq.append(oid)
            if gene in gene_set:
                assignment.append(gene)
                rendezvous.append(copy.deepcopy(rv))
            else:
                assignment.append("A")
                rendezvous.append(None)

    for oid in order_ids:
        if oid in seq:
            continue
        gene = _preferred_gene(gene_pool, ("A",))
        seq.append(oid)
        assignment.append(gene)
        rendezvous.append(make_random_rendezvous_for_gene(gene, support_node_ids, allow_c_recover_station))
    ind = Individual(seq, assignment, rendezvous)
    ind.validate()
    return ind


def _with_new_orders_appended(
    base: Individual,
    new_ids: list[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    allow_c_recover_station: bool,
) -> Individual:
    ind = copy.deepcopy(base)
    existing = set(ind.sequence)
    for oid in new_ids:
        if oid in existing:
            continue
        gene = _preferred_gene(gene_pool, ("C", "B", "A"))
        ind.sequence.append(oid)
        ind.assignment.append(gene)
        ind.rendezvous.append(make_random_rendezvous_for_gene(gene, depot_ids, station_ids, allow_c_recover_station))
    return ind


def _with_new_orders_nearest(
    base: Individual,
    new_ids: list[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    orders: dict[str, Any] | None,
    allow_c_recover_station: bool,
) -> Individual:
    ind = copy.deepcopy(base)
    orders = orders or {}
    for oid in new_ids:
        if oid in ind.sequence:
            continue
        insert_at = len(ind.sequence)
        nearest_idx = _nearest_order_index(oid, ind.sequence, orders)
        if nearest_idx is not None:
            insert_at = nearest_idx + 1
        gene = _preferred_gene(gene_pool, ("C", "B", "A"))
        ind.sequence.insert(insert_at, oid)
        ind.assignment.insert(insert_at, gene)
        ind.rendezvous.insert(insert_at, make_random_rendezvous_for_gene(gene, depot_ids, station_ids, allow_c_recover_station))
    return ind


def _with_new_orders_deadline(
    base: Individual,
    new_ids: list[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    orders: dict[str, Any] | None,
    allow_c_recover_station: bool,
) -> Individual:
    ind = copy.deepcopy(base)
    orders = orders or {}
    ordered_new = sorted(
        [oid for oid in new_ids if oid not in set(ind.sequence)],
        key=lambda oid: (
            _safe_float(_read_field(orders.get(oid), "deadline", math.inf), math.inf),
            oid,
        ),
    )
    for oid in ordered_new:
        gene = _preferred_gene(gene_pool, ("C", "B", "A"))
        ind.sequence.append(oid)
        ind.assignment.append(gene)
        ind.rendezvous.append(make_random_rendezvous_for_gene(gene, depot_ids, station_ids, allow_c_recover_station))
    return ind


def _new_orders_with_mode(
    base: Individual | None,
    order_ids: list[str],
    new_ids: list[str],
    gene: str,
    depot_ids: list[str],
    station_ids: list[str],
    allow_c_recover_station: bool,
) -> Individual:
    ind = copy.deepcopy(base) if base is not None else _truck_only(order_ids)
    by_order = {
        oid: (g, copy.deepcopy(rv))
        for oid, g, rv in zip(ind.sequence, ind.assignment, ind.rendezvous)
    }
    for oid in new_ids:
        by_order[oid] = (
            gene,
            make_random_rendezvous_for_gene(gene, depot_ids, station_ids, allow_c_recover_station),
        )
    sequence = [oid for oid in order_ids if oid in by_order]
    assignment = [by_order[oid][0] for oid in sequence]
    rendezvous = [copy.deepcopy(by_order[oid][1]) for oid in sequence]
    return Individual(sequence, assignment, rendezvous)


def _repair_to_order_ids(
    ind: Individual,
    order_ids: list[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    allow_c_recover_station: bool,
) -> None:
    order_set = set(order_ids)
    by_order: dict[str, tuple[str, Any]] = {}
    for oid, gene, rv in zip(ind.sequence, ind.assignment, ind.rendezvous):
        if oid in order_set and oid not in by_order:
            if gene not in gene_pool:
                gene = "A"
                rv = None
            by_order[oid] = (gene, copy.deepcopy(rv))
    sequence: list[str] = []
    seen: set[str] = set()
    for oid in ind.sequence:
        if oid in order_set and oid not in seen:
            sequence.append(oid)
            seen.add(oid)
    for oid in order_ids:
        if oid in seen:
            continue
        sequence.append(oid)
        seen.add(oid)
        if oid in by_order:
            continue
        gene = _preferred_gene(gene_pool, ("A",))
        by_order[oid] = (
            gene,
            make_random_rendezvous_for_gene(gene, depot_ids, station_ids, allow_c_recover_station),
        )
    ind.sequence = sequence
    ind.assignment = [by_order[oid][0] for oid in ind.sequence]
    ind.rendezvous = [copy.deepcopy(by_order[oid][1]) for oid in ind.sequence]
    ind.validate()


def _truck_only(order_ids: list[str]) -> Individual:
    return Individual(list(order_ids), ["A"] * len(order_ids), [None] * len(order_ids))


def _preferred_gene(gene_pool: list[str], modes: Iterable[str]) -> str:
    for mode in modes:
        if mode == "A" and "A" in gene_pool:
            return "A"
        prefix = f"{mode}_"
        for gene in gene_pool:
            if gene.startswith(prefix):
                return gene
    return "A"


def _first_gene_for_mode(gene_pool: list[str], mode: str) -> str | None:
    if mode == "A":
        return "A" if "A" in gene_pool else None
    prefix = f"{mode}_"
    for gene in gene_pool:
        if gene.startswith(prefix):
            return gene
    return None


def _nearest_order_index(order_id: str, sequence: list[str], orders: dict[str, Any]) -> int | None:
    target = _read_field(orders.get(order_id), "delivery_loc")
    if target is None:
        return None
    best_idx = None
    best_dist = math.inf
    for idx, oid in enumerate(sequence):
        pos = _read_field(orders.get(oid), "delivery_loc")
        dist = _distance(target, pos)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _plan_is_feasible_for_orders(plan: Any, planning_orders: dict[str, Any]) -> bool:
    if plan is None:
        return False
    required = set(planning_orders)
    feasible_allocs = {
        str(_read_field(alloc, "order_id", ""))
        for alloc in getattr(plan, "allocations", []) or []
        if bool(_read_field(alloc, "feasible", False))
    }
    return required <= feasible_allocs


def _unserved_order_ids(plan: Any, planning_orders: dict[str, Any]) -> list[str]:
    required = set(planning_orders)
    feasible_allocs = {
        str(_read_field(alloc, "order_id", ""))
        for alloc in getattr(plan, "allocations", []) or []
        if bool(_read_field(alloc, "feasible", False))
    }
    return sorted(required - feasible_allocs)


def _mode_counts(allocations: Iterable[Any]) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "C": 0}
    for alloc in allocations:
        mode = str(_read_field(alloc, "mode", "") or "")
        if mode.startswith("B"):
            counts["B"] += 1
        elif mode == "C":
            counts["C"] += 1
        elif mode == "A":
            counts["A"] += 1
    return counts


def _estimate_truck_available(truck: Any, event_time: float, locked_order_ids: set[str]) -> tuple[float, Any]:
    current_pos = truck.get_location(event_time) if hasattr(truck, "get_location") else _read_field(truck, "current_loc")
    freeze_time = float(event_time)
    freeze_pos = current_pos
    for stop in getattr(truck, "_planned_route_stops", []) or []:
        oid = str(stop.get("order_id", "") or "")
        arrival = _safe_float(stop.get("arrival_time"), math.inf)
        departure = _safe_float(stop.get("departure_time"), arrival)
        if arrival <= event_time < departure:
            freeze_time = max(freeze_time, departure)
            freeze_pos = stop.get("position", freeze_pos)
        if oid in locked_order_ids and departure >= event_time:
            freeze_time = max(freeze_time, departure)
            freeze_pos = stop.get("position", freeze_pos)
    return freeze_time, freeze_pos


def _estimate_drone_available(snapshot: Any, drone: Any, event_time: float) -> tuple[float, Any]:
    route_plan = _read_field(drone, "route_plan", []) or []
    idx = int(_read_field(drone, "current_waypoint_index", 0) or 0)
    cur = _read_field(drone, "current_loc")
    if idx >= len(route_plan):
        return float(event_time), cur
    dist = 0.0
    service_time = _safe_float(_read_field(_entity_mgr(snapshot), "DRONE_SERVICE_TIME_ORDER", 0.0), 0.0)
    service_total = 0.0
    for wp in route_plan[idx:]:
        loc = _read_field(wp, "loc")
        dist += _distance(cur, loc)
        cur = loc
        if _waypoint_action_name(_read_field(wp, "action")) == "DELIVER":
            service_total += service_time
    speed = max(1e-6, _safe_float(_read_field(drone, "cruise_speed", 0.0), 0.0))
    return float(event_time) + dist / speed + service_total, cur


def _distance(pos_a: Any, pos_b: Any) -> float:
    if pos_a is None or pos_b is None:
        return math.inf
    if hasattr(pos_a, "distance_2d"):
        try:
            return float(pos_a.distance_2d(pos_b))
        except Exception:
            return math.inf
    if all(hasattr(pos, "x") and hasattr(pos, "y") for pos in (pos_a, pos_b)):
        dx = float(pos_a.x) - float(pos_b.x)
        dy = float(pos_a.y) - float(pos_b.y)
        return (dx * dx + dy * dy) ** 0.5
    return math.inf


def _drone_is_flying(drone: Any) -> bool:
    status = _read_field(drone, "status")
    is_flying = getattr(status, "is_flying", None)
    if is_flying is not None:
        return bool(is_flying)
    return _status_name(status) in {"FLYING_TO_PICKUP", "FLYING_TO_DELIVER", "FLYING_TO_STATION", "FLYING_TO_TRUCK", "RETURNING_TO_DEPOT"}


def _has_pending_route(drone: Any) -> bool:
    value = _read_field(drone, "has_pending_route")
    if value is not None:
        return bool(value)
    route_plan = _read_field(drone, "route_plan", []) or []
    idx = int(_read_field(drone, "current_waypoint_index", 0) or 0)
    return idx < len(route_plan)


def _entity_mgr(state: Any) -> Any:
    return _read_field(state, "entity_mgr") or _read_field(state, "entity_manager") or state


def _order_mgr(state: Any) -> Any:
    return _read_field(state, "order_mgr") or _read_field(state, "order_manager") or _read_field(_entity_mgr(state), "order_mgr")


def _mapping(state_or_mgr: Any, field_name: str) -> dict[str, Any]:
    value = _read_field(state_or_mgr, field_name)
    return value if isinstance(value, dict) else {}


def _order_id(order: Any) -> str:
    return str(_read_field(order, "order_id", "") or "").strip()


def _status_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip().upper()


def _read_field(record: Any, field_name: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def _write_field(record: Any, field_name: str, value: Any) -> None:
    if isinstance(record, dict):
        record[field_name] = value
    else:
        setattr(record, field_name, value)


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default
