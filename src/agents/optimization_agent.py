from typing import Dict, List, Any
from blackboard import Blackboard, EventType
from agents.decorators import agent_error_handler
from itertools import product

class OptimizationAgent:
    def __init__(self, blackboard: Blackboard):
        """
        Agente encargado de optimizar las configuraciones de componentes propuestos
        :param blackboard: Instancia del Blackboard para acceder a datos compartidos
        """
        
        self.blackboard = blackboard

        # Suscribirse al evento de compatibilidad completada
        self.blackboard.subscribe(
            EventType.COMPATIBILITY_CHECKED,
            self.optimize
        )

    @agent_error_handler
    def optimize(self):
        """
        Optimiza las configuraciones de componentes propuestos basándose en las restricciones del usuario y problemas de compatibilidad. Este agente combina las propuestas de múltiples agentes especializados y genera configuraciones óptimas.
        """

        # Obtener datos del blackboard
        proposals: Dict[str, List[Dict]] = self.blackboard.get_consolidated_components() or {}
        requirements = self.blackboard.get("user_requirements") or {}
        compatibility_issues = self.blackboard.get("compatibility_issues") or []

        # Generar combinaciones posibles
        domains = [proposals[k] for k in sorted(proposals.keys())]
        keys = sorted(proposals.keys())  # ['CPU', 'GPU', ...]

        max_budget = requirements.budget.get("max", float("inf"))

        valid_builds = []
        for combo in product(*domains):
            build = {k: v for k, v in zip(keys, combo)}
            if self._is_valid(build, max_budget, compatibility_issues):
                valid_builds.append(build)

        print(f"[OptimizationAgent] {len(valid_builds)} builds válidas encontradas")

        # Rankear
        ranked_builds = sorted(valid_builds, key=self._build_score, reverse=True)

        # Empaquetar resultados
        final_builds = []
        for build in ranked_builds[:3]:
            total_price = sum(float(comp.get("price", comp.get("Price", 0))) for comp in build.values())
            final_builds.append({
                "components": build,
                "total_price": round(total_price, 2),
                "performance_rating": self._estimate_performance(build),
                "compatibility_warnings": [],  # Se puede completar luego
                "upgrade_paths": {}
            })

        # Guardar resultado
        self.blackboard.update(
            section="optimized_configs",
            data=final_builds,
            agent_id="optimization_agent",
            notify=True
        )

    def _is_valid(self, build: Dict[str, Dict], max_budget: float, compatibility_issues: List[Dict]) -> bool:
        """Verifica si una build cumple con las restricciones duras"""
        # Verificar presupuesto total
        total_price = 0
        for comp in build.values():
            try:
                price = float(comp.get("price", comp.get("Price", "0")))
                total_price += price
            except Exception:
                return False

        if total_price > max_budget:
            return False

        # TODO: Verificar contra compatibility_issues
        # Por ahora asumimos que todo es compatible
        return True

    def _estimate_performance(self, build: Dict[str, Dict]) -> float:
        """Estimación básica de rendimiento combinado"""
        # Esto se puede mejorar mucho
        score = 0.0
        if "CPU" in build:
            s = build["CPU"].get("score", {})
            score += s.get("score", 0) + 0.5 * s.get("multicore_score", 0)
        if "GPU" in build:
            s = build["GPU"].get("score", {})
            score += 1.5 * s.get("multicore_score", 0)
        return score

    def _build_score(self, build: Dict[str, Dict]) -> float:
        """Heurística de ranking: rendimiento/precio"""
        perf = self._estimate_performance(build)
        price = sum(float(comp.get("price", comp.get("Price", 0))) for comp in build.values())
        return perf / price if price > 0 else 0
