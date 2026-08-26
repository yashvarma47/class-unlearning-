"""Plan A: MicroGA / NSGA-II search for a class-unlearning strategy.

Searches over ``(b, g, s, d_g, d_s)`` -- which layer group to touch, with which
editor and smoother operator, at what intensity -- against three objectives:

    f1 = JS(P_ref(D_f) || P(D_f))     match the reference's distribution on D_f
    f2 = L_r                           retain loss
    f3 = ||dtheta|| / ||theta_0||      edit cost

Screening runs at reduced fidelity; the fronts this produces are re-measured by
``evaluate_class_front.py`` before any number is reported.

Run::

    python experiments/run_plan_a.py --config search/plan_a_frog_smoke.yaml
    python experiments/run_plan_a.py --config search/plan_a_frog.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.evaluation import ClassEvaluator  # noqa: E402
from medus_class.evaluation.metrics import normalise_objectives  # noqa: E402
from medus_class.search import Chromosome, ChromosomeBounds, decode  # noqa: E402
from medus_class.search.nsga2 import NSGA2, NSGA2Config  # noqa: E402
from medus_class.search.population import PopulationEvaluator  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="search/plan_a_frog.yaml")
    parser.add_argument("--limit-generations", type=int, default=None,
                        help="Override the config, for a shorter run.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    search_cfg = cfg["search"]
    generations = args.limit_generations or int(search_cfg["generations"])

    print("=" * 100)
    print(f"PLAN A -- {search_cfg['name']}")
    print("=" * 100)

    build_started = time.perf_counter()
    evaluator = ClassEvaluator(cfg)
    print(f"  evaluator ready in {time.perf_counter() - build_started:.1f}s")
    print(f"  forget class     {evaluator.forget_class}")
    print(f"  device           {evaluator.device}")
    print(f"  selector         {evaluator.selection_rule}")
    print(f"  max_level        {evaluator.max_level}")
    print(f"  loader sizes     {evaluator.loaders.sizes()}")
    print(f"  reference        {Path(evaluator.reference_path).name}")

    print("\n  BASELINES (full sets)")
    print(f"    {'':<12}{'D_f acc':>10}{'D_r acc':>10}{'D_f_test':>11}{'D_r_test':>11}")
    for label, m in (("original", evaluator.original),
                     ("reference", evaluator.reference_metrics)):
        print(f"    {label:<12}{m['forget_train_acc']:>10.4f}{m['retain_train_acc']:>10.4f}"
              f"{m['forget_test_acc']:>11.4f}{m['retain_test_acc']:>11.4f}")

    bounds = ChromosomeBounds.from_registry(
        n_groups=len(evaluator.registry.names),
        implemented_only=True,
        max_level=cfg["evaluation"].get("max_level"),
    )
    print(f"\n  search space     {bounds.search_space_size():.3e}")
    print(f"  genes            {bounds.n_genes}")

    population = PopulationEvaluator(
        evaluator=evaluator,
        cache_objectives=bool(search_cfg.get("cache_objectives", True)),
    )

    normalise = bool(search_cfg.get("normalise_objectives", True))

    def evaluate_batch(chromosomes: list[Chromosome]) -> list[tuple[float, float, float]]:
        values = population.evaluate(chromosomes)
        return normalise_objectives(values) if normalise else values

    config = NSGA2Config(
        population_size=int(search_cfg["population_size"]),
        generations=generations,
        crossover_probability=float(search_cfg["crossover_probability"]),
        mutation_probability=search_cfg.get("mutation_probability"),
        p_active=float(search_cfg.get("p_active", 0.5)),
        seed=int(search_cfg.get("seed", 42)),
        normalise_objectives=normalise,
    )
    algorithm = NSGA2(bounds=bounds, config=config, evaluate=evaluate_batch)

    results_dir = resolve_path(search_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 100)
    print(f"  {'gen':>4}{'evaluated':>11}{'cached':>8}{'failed':>8}"
          f"{'best f1':>10}{'best f2':>10}{'best f3':>10}{'fronts':>18}")
    print("-" * 100)

    def on_generation(index: int, chromosomes, objectives) -> None:
        columns = list(zip(*objectives)) if objectives else ((), (), ())
        best = [min(c) if c else float("nan") for c in columns]
        print(f"  {index:>4}{population.evaluated:>11}{population.cache_hits:>8}"
              f"{population.failures:>8}{best[0]:>10.5f}{best[1]:>10.5f}"
              f"{best[2]:>10.5f}", flush=True)

    started = time.perf_counter()
    result = algorithm.run(on_generation=on_generation)
    elapsed = time.perf_counter() - started

    # --- outputs ----------------------------------------------------------
    front = result.pareto_front()
    print("\n" + "=" * 100)
    print("SEARCH COMPLETE")
    print("=" * 100)
    print(f"  wall clock       {elapsed/60:.1f} min")
    print(f"  evaluated        {population.evaluated}")
    print(f"  cache hits       {population.cache_hits}")
    print(f"  failures         {population.failures}")
    print(f"  Pareto front     {len(front)} members")

    rows: list[dict[str, Any]] = []
    for position, index in enumerate(front):
        chromosome = result.population[index]
        strategy = decode(chromosome, evaluator.registry.names)
        f1, f2, f3 = result.objectives[index]
        rows.append({
            "front_position": position,
            "obj1_js": f1,
            "obj2_retain_loss": f2,
            "obj3_edit_cost": f3,
            "crowding_distance": result.distance[index],
            "n_actions": len(strategy.actions),
            "operators": "|".join(sorted({a.operator_name.split("@")[0]
                                          for a in strategy.actions})),
            "strategy": " ".join(
                f"{a.group_name}:{a.operator_name}" for a in strategy.actions),
            "chromosome": " ".join(str(g) for g in chromosome.to_vector()),
        })

    front_path = results_dir / "pareto_front.csv"
    with front_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    history_path = results_dir / "evaluation_history.csv"
    records = [r.to_dict() if hasattr(r, "to_dict") else r
               for r in population.records]
    if records:
        with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    (results_dir / "summary.json").write_text(json.dumps({
        "name": search_cfg["name"],
        "config": args.config,
        "generations": generations,
        "population_size": config.population_size,
        "evaluated": population.evaluated,
        "cache_hits": population.cache_hits,
        "failures": population.failures,
        "elapsed_seconds": round(elapsed, 1),
        "forget_class": evaluator.forget_class,
        "reference_checkpoint": evaluator.reference_path,
        "original": evaluator.original,
        "reference": evaluator.reference_metrics,
        "front_size": len(front),
    }, indent=2), encoding="utf-8")

    print(f"\n  wrote {front_path.relative_to(PROJECT_ROOT)}")
    print(f"  Re-measure this front at full fidelity before reporting anything:")
    print(f"    python experiments/evaluate_class_front.py --front "
          f"{front_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
