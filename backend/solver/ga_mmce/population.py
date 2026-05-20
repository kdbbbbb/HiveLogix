from __future__ import annotations

import copy
import random

from .chromosome import Individual
from .operators import choose_gene_by_mode, make_random_rendezvous_for_gene


def _check_inputs(
    order_ids: list[str],
    gene_pool: list[str],
    support_node_ids: list[str],
) -> None:
    if len(set(order_ids)) != len(order_ids):
        raise ValueError("order_ids contains duplicated order ids")
    if not gene_pool:
        raise ValueError("gene_pool must not be empty")
    if "A" not in gene_pool:
        raise ValueError('gene_pool must include "A"')
    if not support_node_ids:
        raise ValueError("support_node_ids must not be empty")


def make_random_individual(
    order_ids: list[str],
    gene_pool: list[str],
    support_node_ids: list[str],
    allow_c_recover_station: bool = True,
    mode_probabilities: dict[str, float] | None = None,
) -> Individual:
    _check_inputs(order_ids, gene_pool, support_node_ids)

    seq = list(order_ids)
    random.shuffle(seq)
    assignment: list[str] = []
    rendezvous = []

    for _ in seq:
        gene = choose_gene_by_mode(gene_pool, mode_probabilities)
        assignment.append(gene)
        rendezvous.append(
            make_random_rendezvous_for_gene(
                gene,
                support_node_ids,
                allow_c_recover_station,
            )
        )

    ind = Individual(seq, assignment, rendezvous)
    ind.validate()
    return ind


def make_single_b_seed_individual(
    order_ids: list[str],
    order_id: str,
    b_gene: str,
    rendezvous: dict[str, str],
) -> Individual:
    assignment = ["A"] * len(order_ids)
    rvs = [None] * len(order_ids)
    if order_id in order_ids:
        idx = order_ids.index(order_id)
        assignment[idx] = b_gene
        rvs[idx] = dict(rendezvous)
    ind = Individual(
        sequence=list(order_ids),
        assignment=assignment,
        rendezvous=rvs,
    )
    ind.validate()
    return ind


def make_combined_b_seed_individual(
    order_ids: list[str],
    b_seed_rendezvous_by_order: dict[str, tuple[str, dict[str, str]]],
) -> Individual:
    assignment = ["A"] * len(order_ids)
    rvs = [None] * len(order_ids)
    index_by_order = {order_id: idx for idx, order_id in enumerate(order_ids)}
    for order_id, seed_data in b_seed_rendezvous_by_order.items():
        idx = index_by_order.get(order_id)
        if idx is None:
            continue
        b_gene, rv = seed_data
        assignment[idx] = b_gene
        rvs[idx] = dict(rv)
    ind = Individual(
        sequence=list(order_ids),
        assignment=assignment,
        rendezvous=rvs,
    )
    ind.validate()
    return ind


def make_truck_only_individual(order_ids: list[str]) -> Individual:
    ind = Individual(
        sequence=list(order_ids),
        assignment=["A"] * len(order_ids),
        rendezvous=[None] * len(order_ids),
    )
    ind.validate()
    return ind


def make_obl_individual(
    base: Individual,
    gene_pool: list[str],
    support_node_ids: list[str],
    allow_c_recover_station: bool = True,
) -> Individual:
    base.validate()
    _check_inputs(base.sequence, gene_pool, support_node_ids)

    seq = list(reversed(base.sequence))
    assignment: list[str] = []
    rendezvous = []

    for gene in reversed(base.assignment):
        candidates = [candidate for candidate in gene_pool if candidate != gene]
        new_gene = random.choice(candidates or gene_pool)

        assignment.append(new_gene)
        rendezvous.append(
            make_random_rendezvous_for_gene(
                new_gene,
                support_node_ids,
                allow_c_recover_station,
            )
        )

    ind = Individual(seq, assignment, rendezvous)
    ind.validate()
    return ind


def _copy_seed_if_valid(
    seed: Individual,
    order_set: set[str],
    gene_set: set[str],
    fixed_tail_order_ids: list[str] | None = None,
    fixed_tail_gene_by_order: dict[str, tuple[str, dict[str, str] | None]] | None = None,
) -> Individual | None:
    copied = copy.deepcopy(seed)
    try:
        enforce_fixed_tail(
            copied,
            fixed_tail_order_ids or [],
            fixed_tail_gene_by_order or {},
        )
        copied.validate()
    except ValueError:
        return None

    if set(copied.sequence) != order_set:
        return None
    if any(gene not in gene_set for gene in copied.assignment):
        return None
    return copied


def enforce_fixed_tail(
    ind: Individual,
    fixed_tail_order_ids: list[str] | None = None,
    fixed_tail_gene_by_order: dict[str, tuple[str, dict[str, str] | None]] | None = None,
) -> Individual:
    """Keep frozen future orders at the tail with their preserved genes."""
    tail_ids = [str(order_id) for order_id in (fixed_tail_order_ids or [])]
    if not tail_ids:
        return ind

    ind.validate()
    tail_set = set(tail_ids)
    gene_map = {
        order_id: (gene, copy.deepcopy(rv))
        for order_id, gene, rv in zip(ind.sequence, ind.assignment, ind.rendezvous)
    }
    for order_id, value in (fixed_tail_gene_by_order or {}).items():
        gene, rv = value
        gene_map[str(order_id)] = (str(gene), copy.deepcopy(rv))

    head_ids = [order_id for order_id in ind.sequence if order_id not in tail_set]
    existing_tail_ids = [order_id for order_id in tail_ids if order_id in set(ind.sequence)]
    sequence = head_ids + existing_tail_ids
    assignment: list[str] = []
    rendezvous = []
    for order_id in sequence:
        gene, rv = gene_map.get(order_id, ("A", None))
        assignment.append(gene)
        rendezvous.append(copy.deepcopy(rv))

    ind.sequence = sequence
    ind.assignment = assignment
    ind.rendezvous = rendezvous
    ind.validate()
    return ind


def initialize_population(
    order_ids: list[str],
    gene_pool: list[str],
    support_node_ids: list[str],
    pop_size: int,
    greedy_seed: Individual | None = None,
    warm_start: list[Individual] | None = None,
    use_truck_only_seed: bool = True,
    use_obl_seed: bool = True,
    allow_c_recover_station: bool = True,
    use_balanced_initialization: bool = True,
    b_seed_rendezvous_by_order: dict[str, tuple[str, dict[str, str]]] | None = None,
    mutation_mode_probabilities: dict[str, float] | None = None,
    fixed_tail_order_ids: list[str] | None = None,
    fixed_tail_gene_by_order: dict[str, tuple[str, dict[str, str] | None]] | None = None,
    prefer_bc_warm_start: bool = False,
    dynamic_fill_from_warm_start: bool = False,
    dynamic_mutable_order_ids: list[str] | None = None,
) -> list[Individual]:
    _check_inputs(order_ids, gene_pool, support_node_ids)
    if pop_size <= 0:
        return []

    order_set = set(order_ids)
    gene_set = set(gene_pool)
    population: list[Individual] = []

    if warm_start:
        if prefer_bc_warm_start:
            ranked_warm_starts = sorted(
                list(warm_start),
                key=lambda ind: (
                    -_bc_count(ind),
                    float(getattr(ind, "fitness", float("inf"))),
                ),
            )
        else:
            ranked_warm_starts = sorted(
                list(warm_start),
                key=lambda ind: float(getattr(ind, "fitness", float("inf"))),
            )
        for seed in ranked_warm_starts:
            copied = _copy_seed_if_valid(
                seed,
                order_set,
                gene_set,
                fixed_tail_order_ids,
                fixed_tail_gene_by_order,
            )
            if copied is not None:
                population.append(copied)
            if len(population) >= pop_size:
                return population[:pop_size]

    if greedy_seed is not None:
        copied = _copy_seed_if_valid(
            greedy_seed,
            order_set,
            gene_set,
            fixed_tail_order_ids,
            fixed_tail_gene_by_order,
        )
        if copied is not None:
            population.append(copied)

    if use_truck_only_seed:
        population.append(
            enforce_fixed_tail(
                make_truck_only_individual(order_ids),
                fixed_tail_order_ids,
                fixed_tail_gene_by_order,
            )
        )

    if use_obl_seed and population:
        population.append(
            enforce_fixed_tail(
                make_obl_individual(
                    population[0],
                    gene_pool,
                    support_node_ids,
                    allow_c_recover_station,
                ),
                fixed_tail_order_ids,
                fixed_tail_gene_by_order,
            )
        )

    if b_seed_rendezvous_by_order:
        valid_b_seeds = {
            order_id: (b_gene, rv)
            for order_id, (b_gene, rv) in b_seed_rendezvous_by_order.items()
            if order_id in order_set and b_gene in gene_set
        }
        if valid_b_seeds:
            population.append(
                enforce_fixed_tail(
                    make_combined_b_seed_individual(order_ids, valid_b_seeds),
                    fixed_tail_order_ids,
                    fixed_tail_gene_by_order,
                )
            )
            if len(population) >= pop_size:
                return population[:pop_size]

        for order_id in order_ids:
            seed_data = valid_b_seeds.get(order_id)
            if seed_data is None:
                continue
            b_gene, rv = seed_data
            population.append(
                enforce_fixed_tail(
                    make_single_b_seed_individual(order_ids, order_id, b_gene, rv),
                    fixed_tail_order_ids,
                    fixed_tail_gene_by_order,
                )
            )
            if len(population) >= pop_size:
                return population[:pop_size]

    if use_balanced_initialization:
        target = min(pop_size, max(len(population), int(pop_size * 0.8)))
        recipes = [
            {"A": 1.0, "B": 0.0, "C": 0.0},
            {"A": 0.55, "B": 0.0, "C": 0.45},
            {"A": 0.65, "B": 0.35, "C": 0.0},
            {"A": 0.40, "B": 0.25, "C": 0.35},
        ]
        index = 0
        while len(population) < target:
            recipe = recipes[index % len(recipes)]
            population.append(
                enforce_fixed_tail(
                    make_random_individual(
                        order_ids,
                        gene_pool,
                        support_node_ids,
                        allow_c_recover_station,
                        recipe,
                    ),
                    fixed_tail_order_ids,
                    fixed_tail_gene_by_order,
                )
            )
            index += 1

    warm_clone_bases = [copy.deepcopy(ind) for ind in population]
    mutable_order_ids = {str(order_id) for order_id in (dynamic_mutable_order_ids or [])}
    clone_index = 0
    while len(population) < pop_size:
        if dynamic_fill_from_warm_start and warm_clone_bases and mutable_order_ids:
            base = copy.deepcopy(warm_clone_bases[clone_index % len(warm_clone_bases)])
            clone_index += 1
            _mutate_mutable_orders_only(
                base,
                mutable_order_ids=mutable_order_ids,
                gene_pool=gene_pool,
                support_node_ids=support_node_ids,
                allow_c_recover_station=allow_c_recover_station,
                mode_probabilities=mutation_mode_probabilities,
            )
            population.append(
                enforce_fixed_tail(
                    base,
                    fixed_tail_order_ids,
                    fixed_tail_gene_by_order,
                )
            )
            continue
        population.append(
            enforce_fixed_tail(
                make_random_individual(
                    order_ids,
                    gene_pool,
                    support_node_ids,
                    allow_c_recover_station,
                    mutation_mode_probabilities,
                ),
                fixed_tail_order_ids,
                fixed_tail_gene_by_order,
            )
        )

    return population[:pop_size]


def _bc_count(ind: Individual) -> int:
    return sum(1 for gene in ind.assignment if str(gene).startswith("B_") or str(gene).startswith("C_"))


def _mutate_mutable_orders_only(
    ind: Individual,
    mutable_order_ids: set[str],
    gene_pool: list[str],
    support_node_ids: list[str],
    allow_c_recover_station: bool,
    mode_probabilities: dict[str, float] | None,
) -> None:
    mutable_indices = [
        idx
        for idx, order_id in enumerate(ind.sequence)
        if str(order_id) in mutable_order_ids
    ]
    if not mutable_indices:
        return
    if len(mutable_indices) >= 2 and random.random() < 0.20:
        i, j = random.sample(mutable_indices, 2)
        ind.sequence[i], ind.sequence[j] = ind.sequence[j], ind.sequence[i]
        ind.assignment[i], ind.assignment[j] = ind.assignment[j], ind.assignment[i]
        ind.rendezvous[i], ind.rendezvous[j] = ind.rendezvous[j], ind.rendezvous[i]

    for idx in mutable_indices:
        if random.random() < 0.35:
            ind.assignment[idx] = choose_gene_by_mode(gene_pool, mode_probabilities)
        if random.random() < 0.45 or ind.assignment[idx] == "A":
            ind.rendezvous[idx] = make_random_rendezvous_for_gene(
                ind.assignment[idx],
                support_node_ids,
                allow_c_recover_station,
            )
    ind.validate()
