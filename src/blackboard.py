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
    TRIGGER_COMPATIBILITY = 3
    COMPATIBILITY_CHECKED = 4
    OPTIMIZATION_DONE = 5
    USER_RESPONSE = 6

@dataclass
class BlackboardEntry:
    data: Any
    timestamp: float
    agent_id: str
    version: int = 1

class Blackboard:
    def __init__(self, components_agent_number = 4):
        # Estado estructurado del sistema
        self.state = {
            'user_input': None,
            'user_response': None,          # Respuesta del usuario a la propuesta
            'user_requirements': None,       # Requisitos extraídos por BDI
            'component_proposals': {},       # {agent_id: [components]}
            'compatibility_issues': [],      # Problemas detectados
            'optimized_configs': [],         # Configuraciones finales
            'knowledge_updates': {},          # Datos para actualizar RAG
            'compatibility_status': {'ready_for_compability': False},
            'errors': []
        }
        
        # Control de concurrencia
        self.lock = threading.RLock()
        self.subscribers: Dict[EventType, List[Callable]] = {e: [] for e in EventType}
        
        # Histórico de cambios (para debugging/experimentación)
        self.audit_log = []
        
        # Número de agentes que deben proponer componentes
        self.total_components_agent_proposal = components_agent_number
        self.actual_components_agent_proposal = 0
        self.subscribe(
            EventType.COMPONENTS_PROPOSED, 
            self.trigger_compability_event
        )
    
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
                    'compatibility_status': EventType.TRIGGER_COMPATIBILITY,
                    'compatibility_issues': EventType.COMPATIBILITY_CHECKED,
                    'optimized_configs': EventType.OPTIMIZATION_DONE,
                    'user_response': EventType.USER_RESPONSE
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
    
    def trigger_compability_event(self):
        self.actual_components_agent_proposal += 1
        
        if self.actual_components_agent_proposal >= self.total_components_agent_proposal:
            print(self.actual_components_agent_proposal, " OK")
            self.update(
                section='compatibility_status', 
                data={'ready_for_compability': True}, 
                agent_id='blackboard', 
                notify=True
            )
    
    def get_consolidated_components(self) -> Dict[str, List]:
        """
        Combina propuestas de múltiples agentes especializados
        :param min_agents: Mínimo de agentes que deben haber contribuido
        :return: {component_type: [components]}
        """
        with self.lock:
            proposals = self.state.get('component_proposals', {})
            
            if self.actual_components_agent_proposal < self.total_components_agent_proposal:
                #raise ValueError(f"Faltan contribuciones de agentes. Solo {len(proposals)}/{min_agents}")
                return []
            
            # Combinar propuestas eliminando duplicados
            consolidated = {}
            for agent, components in proposals.items():
                for comp_type , comp in components.items():
                    #comp_type = comp['Component_Type']
                    if comp_type not in consolidated:
                        consolidated[comp_type] = []
                    
                    for c in comp:
                        consolidated[comp_type].append(c)
            
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
            
    def reset(self):
        # Estado estructurado del sistema
        self.state = {
            'user_input': None,
            'user_response': None,          # Respuesta del usuario a la propuesta
            'user_requirements': None,       # Requisitos extraídos por BDI
            'component_proposals': {},       # {agent_id: [components]}
            'compatibility_issues': [],      # Problemas detectados
            'optimized_configs': [],         # Configuraciones finales
            'knowledge_updates': {},          # Datos para actualizar RAG
            'compatibility_status': {'ready_for_compability': False},
            'errors': []
        }
        
        # Histórico de cambios (para debugging/experimentación)
        self.audit_log = []
        self.actual_components_agent_proposal = 0