from typing import Dict, List, Any, Tuple, Set, Optional
from blackboard import Blackboard, EventType
from agents.decorators import agent_error_handler
from agents.compatibility_agent import ComponentType, CompatibilityIssue
import copy

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

        # domains: Dict[str, List[Dict]] = {
        #     k: [comp["metadata"] for comp in v] for k, v in proposals.items()
        # }
        
        domains = { k : [] for k in proposals}
        url_set = set()
        for k, v in proposals.items():
            for comp in v:
                meta = comp["metadata"]
                if meta.get('URL') not in url_set:
                    
                    price = meta.get("price", meta.get("Price", 1e9))
                    if isinstance(price, str):
                        price = price.replace(',', '.')
                    meta["Price"] = float(price)
                    
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

        # TODO: Agregar _find_best_price_perf_build y _find_best_performance_build

        self.blackboard.update(
            section="optimized_configs",
            data=builds,
            agent_id="optimization_agent",
            notify=True
        )

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
                model_name = value.get('Model_Name', value.get('Model_Model', value.get('Model - Model', 'Unknown')))
                if not model_name:
                    continue

                conflict = False
                for prev_type, prev_comp in assignment.items():
                    prev_name = prev_comp.get('Model_Name', prev_comp.get('Model_Model', prev_comp.get('Model - Model', 'Unknown')))
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
            "performance_rating": self._estimate_performance(build),
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
                name_x = x.get('Model_Name', x.get('Model_Model', x.get('Model - Model', 'Unknown')))
                consistent = any(((Xi, name_x), (Xj, y.get('Model_Name', y.get('Model_Model', y.get('Model - Model', 'Unknown'))))) not in conflicts for y in domains[Xj])
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

    def _estimate_performance(self, build: Dict[str, Dict]) -> float:
        score = 0.0
        if "CPU" in build:
            s = build["CPU"].get("score", {})
            score += s.get("score", 0) + 0.5 * s.get("multicore_score", 0)
        if "GPU" in build:
            s = build["GPU"].get("score", {})
            score += 1.5 * s.get("multicore_score", 0)
        return score

    def _build_score(self, build: Dict[str, Dict]) -> float:
        perf = self._estimate_performance(build)
        price = sum(float(comp.get("price", comp.get("Price", 0))) for comp in build.values())
        return perf / price if price > 0 else 0
