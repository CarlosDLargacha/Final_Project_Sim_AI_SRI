from typing import Dict, List, Any, Tuple
from blackboard import Blackboard, EventType
from agents.decorators import agent_error_handler
from agents.compatibility_agent import ComponentType, CompatibilityIssue
from itertools import product
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

        # Construir dominios originales (component_type -> lista de metadatas)
        domains: Dict[str, List[Dict]] = {
            k: [comp["metadata"] for comp in v] for k, v in proposals.items()
        }

        # Aplicar AC-3 para reducir dominios
        reduced_domains = self._ac3(domains, issues)

        if any(len(v) == 0 for v in reduced_domains.values()):
            print("[OptimizationAgent] AC-3 detectó inconsistencia: no hay combinaciones válidas")
            self.blackboard.update("optimized_configs", [], agent_id="optimization_agent")
            return

        # Generar combinaciones solo con dominios filtrados
        keys = sorted(reduced_domains.keys())
        domain_lists = [reduced_domains[k] for k in keys]

        max_budget = requirements.budget.get("max", float("inf"))
        valid_builds = []
        for combo in product(*domain_lists):
            build = {k: v for k, v in zip(keys, combo)}
            if self._is_valid(build, max_budget):
                valid_builds.append(build)

        print(f"[OptimizationAgent] {len(valid_builds)} builds válidas encontradas")

        # Rankear builds
        ranked_builds = sorted(valid_builds, key=self._build_score, reverse=True)

        final_builds = []
        for build in ranked_builds[:3]:
            total_price = sum(float(comp.get("price", comp.get("Price", 0))) for comp in build.values())
            final_builds.append({
                "components": build,
                "total_price": round(total_price, 2),
                "performance_rating": self._estimate_performance(build),
                "compatibility_warnings": [],
                "upgrade_paths": {}
            })

        self.blackboard.update(
            section="optimized_configs",
            data=final_builds,
            agent_id="optimization_agent",
            notify=True
        )

    def _ac3(self, domains: Dict[str, List[Dict]], issues: List[CompatibilityIssue]) -> Dict[str, List[Dict]]:
        """Aplica el algoritmo AC-3 usando las incompatibilidades para filtrar dominios"""
        queue: List[Tuple[str, str]] = []
        variables = list(domains.keys())

        # Crear cola de arcos para todos los pares de variables
        for i, a in enumerate(variables):
            for b in variables[i+1:]:
                queue.append((a, b))
                queue.append((b, a))

        # Indexar conflictos entre pares (compatibility_issues)
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
            i = 0
            for x in domains[Xi]:
                name_x = x.get('Model_Name', x.get('Model_Model', x.get('Model - Model', 'Unknown')))
                # Existe al menos un y en D_j tal que (Xi=x, Xj=y) NO sea conflicto
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
                    return domains  # inconsistencia detectada
                for Xk in variables:
                    if Xk != Xi and Xk != Xj:
                        queue.append((Xk, Xi))

        return domains

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
