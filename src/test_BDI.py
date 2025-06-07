import pytest
from unittest.mock import MagicMock
from agents.BDI_agent import BDIAgent, HardwareRequirements, UseCase
import json

# Fixture para inicializar el agente con un mock LLM
@pytest.fixture
def bdi_agent():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '''{
        "use_case": "gaming",
        "budget": {"min": 1000, "max": 1500},
        "performance": {"resolution": "4K", "fps": 60},
        "aesthetics": {"rgb": true},
        "constraints": ["small_form_factor"]
    }'''
    return BDIAgent(mock_llm)

# Tests para la extracción de requisitos
class TestRequirementExtraction:
    def test_valid_input_parsing(self, bdi_agent):
        """Prueba extracción de input válido"""
        result = bdi_agent.extract_requirements("Quiero PC para gaming 4K $1500")
        
        assert result.use_case == UseCase.GAMING
        assert result.budget == {"min": 1000.0, "max": 1500.0}
        assert result.performance["resolution"] == "4K"
        assert result.aesthetics["rgb"] is True

    def test_currency_normalization(self, bdi_agent):
        """Prueba normalización de formatos monetarios"""
        bdi_agent._ask_llm = lambda _: '''{
            "use_case": "general",
            "budget": {"max": "$1,500.99"},
            "performance": {},
            "aesthetics": {},
            "constraints": []
        }'''
        result = bdi_agent.extract_requirements("")
        assert result.budget["max"] == 1500.99

# Tests para la generación de preguntas
class TestClarificationQuestions:
    def test_question_generation(self, bdi_agent):
        """Prueba generación de preguntas contextuales"""
        bdi_agent.current_beliefs["missing_fields"] = ["budget.max", "use_case"]
        questions = bdi_agent.generate_clarification_questions()
        
        assert "presupuesto máximo" in questions[0]
        assert "principal uso" in questions[1]

# Tests para manejo de errores
class TestErrorHandling:
    def test_malformed_llm_response(self, bdi_agent):
        """Prueba respuesta mal formada del LLM"""
        bdi_agent._ask_llm = lambda _: "Esto no es un JSON válido"
        
        with pytest.raises(ValueError) as e:
            bdi_agent.extract_requirements("input cualquiera")
        assert "Error parsing" in str(e.value)

    def test_invalid_use_case(self, bdi_agent):
        """Prueba caso de uso inválido"""
        bdi_agent._ask_llm = lambda _: '''{"use_case": "invalid"}'''
        
        with pytest.raises(ValueError):
            bdi_agent.extract_requirements("")

# Tests para actualización del estado BDI
class TestBDIState:
    def test_state_update(self, bdi_agent):
        """Prueba actualización correcta del estado BDI"""
        bdi_agent.extract_requirements("Gaming PC $1500")
        
        assert bdi_agent.user_desires["performance"]["fps"] == 60
        assert "consultar_agentes_especializados" in bdi_agent.intentions

# Tests de integración (simulados)
class TestIntegration:
    def test_full_workflow(self, bdi_agent):
        """Prueba flujo completo con mock LLM"""
        input_text = "Necesito PC para editar video 4K, máximo $2000 con RGB"
        
        # Ejecutar extracción
        requirements = bdi_agent.extract_requirements(input_text)
        
        # Verificar salida
        assert requirements.use_case == UseCase.GAMING
        assert requirements.budget["max"] == 1500.0
        assert bdi_agent.llm.generate.called