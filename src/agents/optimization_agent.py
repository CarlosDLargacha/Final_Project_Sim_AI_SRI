from typing import Dict, List, Any, Tuple, Set, Optional
from blackboard import Blackboard, EventType
from agents.decorators import agent_error_handler
from agents.compatibility_agent import ComponentType, CompatibilityIssue
from model.GeneticOptimizer import GeneticOptimizer
import copy
import re

class OptimizationAgent:
    def __init__(self, blackboard: Blackboard):
        self.blackboard = blackboard

        self.blackboard.subscribe(
            EventType.COMPATIBILITY_CHECKED,
            self.optimize
        )

    @agent_error_handler
    def optimize(self):
        proposals: Dict[str, List[Dict]] = self.blackboard.get_consolidated_components() or {}
        requirements = self.blackboard.get("user_requirements") or {}
        issues: List[CompatibilityIssue] = self.blackboard.get("compatibility_issues") or []

        if not proposals:
            return

        domains = { k : [] for k in proposals}
        url_set = set()
        for k, v in proposals.items():
            for comp in v:
                meta = comp["metadata"]
                if meta.get('URL') not in url_set:
                    price = meta.get("price", meta.get("Price", 1e9))
                    if isinstance(price, str):
                        price = price.replace(',', '')
                    meta["Price"] = float(price)

                    if k == ComponentType.CPU.value:
                        meta['score'] = comp['score']['score']
                        meta['multicore_score'] = comp['score']['multicore_score']

                    domains[k].append(meta)
                    url_set.add(meta.get('URL'))

        reduced_domains = self._ac3(domains, issues)
        if any(len(v) == 0 for v in reduced_domains.values()):
            print("[OptimizationAgent] AC-3 detectó inconsistencia: no hay combinaciones válidas")
            self.blackboard.update("optimized_configs", [], agent_id="optimization_agent")
            return

        max_budget = requirements.budget.get("max", float("inf"))
        conflict_set = self._build_conflict_set(issues)

        builds = []

        cheapest = self._find_cheapest_build(reduced_domains, max_budget, conflict_set)
        if cheapest:
            builds.append(self._package_build(cheapest, label="Build Más Económica"))

        optimizer = GeneticOptimizer(
            domains=reduced_domains,
            budget_limit=max_budget,
            compatibility_conflicts=conflict_set,
            fitness_mode='performance'
        )

        performance = optimizer.run()
        if performance:
            builds.append(self._package_build(performance, label="Build Con Mejor Rendimiento"))

        self.blackboard.update(
            section="optimized_configs",
            data=builds,
            agent_id="optimization_agent",
            notify=True
        )

        print("[OptimizationAgent] Build creadas")

    def _find_best_price_perf_build(
        self,
        domains: Dict[str, List[Dict]],
        max_budget: float,
        compatibility_conflicts: Set[Tuple[Tuple[str, str], Tuple[str, str]]]
    ) -> Optional[Dict[str, Dict]]:
        max_perf = {
            k: max((self._estimate_individual_perf(c) for c in comps), default=0) for k, comps in domains.items()
        }

        min_price = {
            k: min((float(c.get("Price", 1e9)) for c in comps), default=1e9) for k, comps in domains.items()
        }

        # Reordenar variables por impacto (mayor rendimiento primero)
        variables = sorted(domains.keys(), key=lambda k: -max_perf[k])

        domains_sorted = {
            k: sorted(v, key=self._estimate_comp_performance) for k, v in domains.items()
        }

        best_score = 0
        best_build = None
        num_chop = [0, 0]
        def backtrack(assignment, perf_so_far, price_so_far):
            nonlocal best_score, best_build, num_chop

            if len(assignment) == len(variables):
                if price_so_far > max_budget:
                    return
                score = perf_so_far / price_so_far if price_so_far > 0 else 0
                if score > best_score:
                    best_score = score
                    best_build = assignment.copy()
                return

            var = variables[len(assignment)]
            remaining_vars = variables[len(assignment)+1:]
            max_remaining_perf = sum(max_perf[v] for v in remaining_vars)
            min_remaining_price = sum(min_price[v] for v in remaining_vars)

            for value in domains_sorted[var]:
                model_name = value.get('Model_Name', 'Unknown')
                if not model_name:
                    continue

                conflict = False
                for prev_type, prev_comp in assignment.items():
                    prev_name = prev_comp.get('Model_Name', 'Unknown')
                    if ((var, model_name), (prev_type, prev_name)) in compatibility_conflicts:
                        conflict = True
                        break
                if conflict:
                    num_chop[0] += 1
                    continue

                # Forward checking: asegurar consistencia futura
                skip_branch = False
                for next_var in remaining_vars:
                    compatible = any(
                        ((next_var, c.get("Model_Name", "Unknown")), (var, model_name)) not in compatibility_conflicts
                        for c in domains_sorted[next_var]
                    )
                    if not compatible:
                        skip_branch = True
                        break
                if skip_branch:
                    continue

                comp_price = float(value.get("Price", 1e9))
                comp_perf = self._estimate_comp_performance(value)

                price_est = price_so_far + comp_price + min_remaining_price
                perf_est = perf_so_far + comp_perf + max_remaining_perf
                upper_bound = perf_est / price_est if price_est > 0 else 0

                if upper_bound < best_score:
                    num_chop[1] += 1
                    continue

                assignment[var] = value
                backtrack(assignment, perf_so_far + comp_perf, price_so_far + comp_price)
                del assignment[var]

        backtrack({}, 0, 0)
        print(num_chop)
        return best_build

    def _find_cheapest_build(
        self,
        domains: Dict[str, List[Dict]],
        max_budget: float,
        compatibility_conflicts: Set[Tuple[Tuple[str, str], Tuple[str, str]]]
    ) -> Optional[Dict[str, Dict]]:
        variables = sorted(domains.keys())
        domains_sorted = {
            k: sorted(v, key=lambda c: float(c.get("Price", 1e9))) for k, v in domains.items()
        }

        def backtrack_cheapest(assignment):
            if len(assignment) == len(variables):
                if self._is_valid(assignment, max_budget):
                    return assignment.copy()
                return None

            var = variables[len(assignment)]
            for value in domains_sorted[var]:
                model_name = value.get('Model_Name', 'Unknown')
                if not model_name:
                    continue

                conflict = False
                for prev_type, prev_comp in assignment.items():
                    prev_name = prev_comp.get('Model_Name', 'Unknown')
                    if ((var, model_name), (prev_type, prev_name)) in compatibility_conflicts:
                        conflict = True
                        break
                if conflict:
                    continue

                assignment[var] = value
                result = backtrack_cheapest(assignment)
                if result:
                    return result
                del assignment[var]

            return None

        return backtrack_cheapest({})

    def _package_build(self, build: Dict[str, Dict], label: str) -> Dict:
        total_price = sum(float(comp.get("price", comp.get("Price", 0))) for comp in build.values())
        return {
            "components": build,
            "total_price": round(total_price, 2),
            "performance_rating": self._estimate_build_performance(build),
            "compatibility_warnings": [],
            "upgrade_paths": {},
            "label": label
        }

    def _ac3(self, domains: Dict[str, List[Dict]], issues: List[CompatibilityIssue]) -> Dict[str, List[Dict]]:
        queue: List[Tuple[str, str]] = []
        variables = list(domains.keys())

        for i, a in enumerate(variables):
            for b in variables[i+1:]:
                queue.append((a, b))
                queue.append((b, a))

        conflicts = set()
        for issue in issues:
            a = issue.component_a.model_name
            b = issue.component_b.model_name
            type_a = issue.component_a.type.value
            type_b = issue.component_b.type.value
            conflicts.add(((type_a, a), (type_b, b)))
            conflicts.add(((type_b, b), (type_a, a)))

        domains = copy.deepcopy(domains)

        def revise(Xi: str, Xj: str) -> bool:
            revised = False
            new_domain = []
            for x in domains[Xi]:
                name_x = x.get('Model_Name', 'Unknown')
                consistent = any(((Xi, name_x), (Xj, y.get('Model_Name', 'Unknown'))) not in conflicts for y in domains[Xj])
                if consistent:
                    new_domain.append(x)
                else:
                    revised = True
            domains[Xi] = new_domain
            return revised

        while queue:
            Xi, Xj = queue.pop(0)
            if revise(Xi, Xj):
                if len(domains[Xi]) == 0:
                    return domains
                for Xk in variables:
                    if Xk != Xi and Xk != Xj:
                        queue.append((Xk, Xi))

        return domains

    def _build_conflict_set(self, issues: List[CompatibilityIssue]) -> Set[Tuple[Tuple[str, str], Tuple[str, str]]]:
        conflict_set = set()
        for issue in issues:
            a = (issue.component_a.type.value, issue.component_a.model_name)
            b = (issue.component_b.type.value, issue.component_b.model_name)
            conflict_set.add((a, b))
            conflict_set.add((b, a))
        return conflict_set

    def _is_valid(self, build: Dict[str, Dict], max_budget: float) -> bool:
        total_price = 0
        for comp in build.values():
            try:
                price = float(comp.get("price", comp.get("Price", "0")))
                total_price += price
            except Exception:
                return False
        return total_price <= max_budget

    def _estimate_build_performance(self, build: Dict[str, Dict]) -> float:
        score = 0.0
        for comp in build.values():
            score += self._estimate_comp_performance(comp)
        return score

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

    def _estimate_comp_performance(self, comp):
        perf = self._estimate_individual_perf(comp)
        price = float(comp.get("Price", 1e9))
        return -perf / price if price > 0 else float("inf")

    def _build_score(self, build: Dict[str, Dict]) -> float:
        perf = self._estimate_build_performance(build)
        price = sum(float(comp.get("price", comp.get("Price", 0))) for comp in build.values())
        return perf / price if price > 0 else 0

    def _extract_best_seller_rank(self, comp: Dict) -> float:
        raw = comp.get("_Best Seller Ranking", "")
        match = re.search(r"#(\d+)", raw)
        if match:
            return int(match.group(1))
        return float("inf")
