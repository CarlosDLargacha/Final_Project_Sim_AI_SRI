from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import time
import json

class EventType(Enum):
    """Tipos de eventos para notificaciones"""
    USER_INPUT = 0
    REQUIREMENTS_UPDATED = 1
    COMPONENTS_PROPOSED = 2
    COMPATIBILITY_CHECKED = 3
    OPTIMIZATION_DONE = 4

@dataclass
class BlackboardEntry:
    data: Any
    timestamp: float
    agent_id: str
    version: int = 1

class Blackboard:
    def __init__(self):
        # Estado estructurado del sistema
        self.state = {
            'user_input': None,
            'user_requirements': None,       # Requisitos extraídos por BDI
            'component_proposals': {},       # {agent_id: [components]}
            'compatibility_issues': [],      # Problemas detectados
            'optimized_configs': [],         # Configuraciones finales
            'knowledge_updates': {},          # Datos para actualizar RAG
            'compatibility_status': {'ready_for_optimization': False},
            'errors': []
        }
        
        # Control de concurrencia
        self.lock = threading.RLock()
        self.subscribers: Dict[EventType, List[Callable]] = {e: [] for e in EventType}
        
        # Histórico de cambios (para debugging/experimentación)
        self.audit_log = []
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """Registra un agente para recibir notificaciones"""
        with self.lock:
            self.subscribers[event_type].append(callback)
    
    def update(self, section: str, data: Any, agent_id: str, notify: bool = True):
        """Actualiza una sección del estado de manera segura"""
        with self.lock:
            # Registrar cambio
            entry = BlackboardEntry(
                data=data,
                timestamp=time.time(),
                agent_id=agent_id
            )
            
            # Secciones especiales con versionado
            if section in self.state and isinstance(self.state[section], dict):
                self.state[section][agent_id] = data
                entry.version = len(self.state[section])
            else:
                self.state[section] = data
            
            self.audit_log.append((section, entry))
            
            # Notificar según tipo de cambio
            if notify:
                event_map = {
                    'user_input': EventType.USER_INPUT,
                    'user_requirements': EventType.REQUIREMENTS_UPDATED,
                    'component_proposals': EventType.COMPONENTS_PROPOSED,
                    'compatibility_issues': EventType.COMPATIBILITY_CHECKED,
                    'optimized_configs': EventType.OPTIMIZATION_DONE
                }
                
                if section in event_map:
                    self._notify(event_map[section])
    
    def get(self, section: str, agent_id: str = None) -> Any:
        """Obtiene datos de una sección de manera segura"""
        with self.lock:
            data = self.state.get(section)
            
            if agent_id and isinstance(data, dict):
                return data.get(agent_id)
            return data
    
    def _notify(self, event_type: EventType):
        """Notifica a agentes suscritos de manera asíncrona"""
        with self.lock:
            callbacks = self.subscribers[event_type][:]
        
        # Ejecutar en hilos separados para no bloquear
        for callback in callbacks:
            threading.Thread(target=callback, daemon=True).start()
    
    def get_consolidated_components(self, min_agents: int = 3) -> Dict[str, List]:
        """
        Combina propuestas de múltiples agentes especializados
        :param min_agents: Mínimo de agentes que deben haber contribuido
        :return: {component_type: [components]}
        """
        with self.lock:
            proposals = self.state.get('component_proposals', {})
            
            if len(proposals) < min_agents:
                raise ValueError(f"Faltan contribuciones de agentes. Solo {len(proposals)}/{min_agents}")
            
            # Combinar propuestas eliminando duplicados
            consolidated = {}
            for agent, components in proposals.items():
                for comp in components:
                    comp_type = comp['component_type']
                    if comp_type not in consolidated:
                        consolidated[comp_type] = []
                    
                    # Evitar duplicados por ID/modelo
                    if not any(c['id'] == comp['id'] for c in consolidated[comp_type]):
                        consolidated[comp_type].append(comp)
            
            return consolidated
    
    def log_experiment_data(self, experiment_name: str):
        """Exporta datos para experimentación"""
        with self.lock:
            return {
                'name': experiment_name,
                'state': json.dumps(self.state, indent=2),
                'timeline': self.audit_log,
                'subscribers': {e.name: len(cb) for e, cb in self.subscribers.items()}
            }