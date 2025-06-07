from dataclasses import dataclass
from typing import List, Dict

@dataclass
class HardwareComponent:
    """Estructura para representar componentes de hardware"""
    id: str
    model: str
    specs: Dict[str, any]
    price: float
    performance_score: float
    compatibility: List[str]

@dataclass
class SystemRecommendation:
    """Estructura para configuraciones completas recomendadas"""
    components: Dict[str, HardwareComponent]
    total_price: float
    performance_rating: float
    compatibility_warnings: List[str]
    upgrade_paths: Dict[str, List[str]]