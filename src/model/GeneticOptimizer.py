import random
import re
import time
from typing import Dict, List, Tuple, Set, Optional, Callable

class GeneticOptimizer:
    def __init__(
        self,
        domains: Dict[str, List[Dict]],
        budget_limit: float,
        compatibility_conflicts: Set[Tuple[Tuple[str, str], Tuple[str, str]]],
        fitness_mode: str = "quality_price",
        population_size: int = 50,
        generations: int = 10000,
        mutation_rate: float = 0.1,
        elite_ratio: float = 0.1,
        timeout: float = 5.0  # segundos
    ):
        self.domains = domains
        self.budget_limit = budget_limit
        self.compatibility_conflicts = compatibility_conflicts
        self.fitness_mode = fitness_mode
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.timeout = timeout
        self.component_types = sorted(domains.keys())

    def run(self) -> Optional[Dict[str, Dict]]:
        print("start")
        population = self._initialize_population()
        best = None
        best_score = float("-inf")
        start_time = time.time()

        for generation in range(self.generations):
            if time.time() - start_time > self.timeout:
                break

            scored = [(ind, self._fitness(ind)) for ind in population]
            scored = [s for s in scored if s[1] is not None]
            if not scored:
                continue

            scored.sort(key=lambda x: x[1], reverse=True)
            best, best_score = scored[0]

            elites = [ind for ind, _ in scored[:int(self.elite_ratio * self.population_size)]]
            new_population = elites[:]

            while len(new_population) < self.population_size:
                p1, p2 = self._select_parents(scored)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

        return best

    def _initialize_population(self) -> List[Dict[str, Dict]]:
        population = []
        attempts = 0
        while len(population) < self.population_size and attempts < self.population_size * 10:
            build = {
                comp: random.choice(self.domains[comp]) for comp in self.component_types
            }
            if self._is_valid(build):
                population.append(build)
            attempts += 1
        return population

    def _fitness(self, build: Dict[str, Dict]) -> Optional[float]:
        if not self._is_valid(build):
            return None
        perf = self._estimate_build_performance(build)
        price = sum(float(c.get("price", c.get("Price", 1e9))) for c in build.values())
        if self.fitness_mode == "quality_price":
            return perf / price if price > 0 else 0
        elif self.fitness_mode == "performance":
            return perf
        else:
            return None

    def _is_valid(self, build: Dict[str, Dict]) -> bool:
        names = [(t, c.get("Model_Name", "Unknown")) for t, c in build.items()]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if (names[i], names[j]) in self.compatibility_conflicts:
                    return False
        total = sum(float(c.get("price", c.get("Price", 1e9))) for c in build.values())
        return total <= self.budget_limit

    def _select_parents(self, scored_population: List[Tuple[Dict[str, Dict], float]]) -> Tuple[Dict, Dict]:
        def tournament():
            k = 3
            return max(random.sample(scored_population, k), key=lambda x: x[1])[0]
        return tournament(), tournament()

    def _crossover(self, p1: Dict[str, Dict], p2: Dict[str, Dict]) -> Dict[str, Dict]:
        child = {}
        for comp in self.component_types:
            child[comp] = random.choice([p1[comp], p2[comp]])
        return child

    def _mutate(self, build: Dict[str, Dict]) -> Dict[str, Dict]:
        new_build = build.copy()
        for comp in self.component_types:
            if random.random() < self.mutation_rate:
                new_build[comp] = random.choice(self.domains[comp])
        return new_build

    def _estimate_build_performance(self, build: Dict[str, Dict]) -> float:
        score = 0.0
        for comp in build.values():
            score += self._estimate_component_perf(comp)
        return score

    def _estimate_component_perf(self, comp: Dict) -> float:
        perf = 0.0
        if comp.get("Type") == "CPU":
            perf += comp.get("score", 0) + 0.5 * comp.get("multicore_score", 0)
        elif comp.get("Type") == "GPU":
            perf += 1.5 * comp.get("multicore_score", 0)
        else:
            perf = 1.0
        return perf
    
    def _estimate_individual_perf(self, comp: Dict) -> float:
        perf = 0.0
        if comp.get("Type") == "CPU":
            perf += comp.get("score", 0) + 0.5 * comp.get("multicore_score", 0)
        elif comp.get("Type") == "GPU":
            perf *= 1.5
        else:
            perf = 1

        rank = self._extract_best_seller_rank(comp)
        if rank < float("inf"):
            perf *= 1 + (100 - min(rank, 100)) / 500
        return perf
    
    def _extract_best_seller_rank(self, comp: Dict) -> float:
        raw = comp.get("_Best Seller Ranking", "")
        match = re.search(r"#(\d+)", raw)
        if match:
            return int(match.group(1))
