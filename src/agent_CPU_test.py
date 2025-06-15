import unittest
import json
import pandas as pd
import numpy as np
from agents.CPU_agent import CPUAgent
from agents.BDI_agent import HardwareRequirements, UseCase
from model.vectorDB import CSVToEmbeddings
from unittest.mock import MagicMock
from blackboard import *

class MockEmbeddingModel:
    """Mock del modelo de embeddings para pruebas"""
    def encode(self, texts, batch_size=32, **kwargs):
        # Simulamos embeddings aleatorios del tamaño correcto
        return np.random.rand(len(texts), 384)

class TestCPUAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Crear datos de prueba para CPUs
        cls.cpu_data = pd.DataFrame([
            {
                "URL": "https://www.newegg.com/amd-ryzen-9-7950x/p/N82E16819113771",
                "Price": "549.99",
                "Component_Type": "CPU",
                "Model_Brand": "AMD",
                "Model_Name": "Ryzen 9 7950X",
                "Details_# of Cores# of Cores": "16",
                "CPU Socket Type_CPU Socket Type": "AM5",
                "Details_Operating FrequencyOperating Frequency": "4.5 GHz",
                "Details_Thermal Design PowerThermal Design Power": "170W"
            },
            {
                "URL": "https://www.newegg.com/intel-core-i9-13900k/p/N82E16819118412",
                "Price": "589.99",
                "Component_Type": "CPU",
                "Model_Brand": "Intel",
                "Model_Name": "Core i9-13900K",
                "Details_# of Cores# of Cores": "24",
                "CPU Socket Type_CPU Socket Type": "LGA1700",
                "Details_Operating FrequencyOperating Frequency": "3.0 GHz",
                "Details_Thermal Design PowerThermal Design Power": "125W"
            },
            {
                "URL": "https://www.newegg.com/amd-ryzen-7-7800x3d/p/N82E16819113772",
                "Price": "449.99",
                "Component_Type": "CPU",
                "Model_Brand": "AMD",
                "Model_Name": "Ryzen 7 7800X3D",
                "Details_# of Cores# of Cores": "8",
                "CPU Socket Type_CPU Socket Type": "AM5",
                "Details_Operating FrequencyOperating Frequency": "4.2 GHz",
                "Details_Thermal Design PowerThermal Design Power": "120W"
            },
            {
                "URL": "https://www.newegg.com/intel-core-i5-13600k/p/N82E16819118413",
                "Price": "319.99",
                "Component_Type": "CPU",
                "Model_Brand": "Intel",
                "Model_Name": "Core i5-13600K",
                "Details_# of Cores# of Cores": "14",
                "CPU Socket Type_CPU Socket Type": "LGA1700",
                "Details_Operating FrequencyOperating Frequency": "3.5 GHz",
                "Details_Thermal Design PowerThermal Design Power": "125W"
            },
            {
                "URL": "https://www.newegg.com/amd-ryzen-5-7600/p/N82E16819113773",
                "Price": "229.99",
                "Component_Type": "CPU",
                "Model_Brand": "AMD",
                "Model_Name": "Ryzen 5 7600",
                "Details_# of Cores# of Cores": "6",
                "CPU Socket Type_CPU Socket Type": "AM5",
                "Details_Operating FrequencyOperating Frequency": "3.8 GHz",
                "Details_Thermal Design PowerThermal Design Power": "65W"
            }
        ])
        
        # Crear embeddings simulados
        cls.embeddings = np.random.rand(len(cls.cpu_data), 384)
        
        # Crear base de datos vectorial simulada CON MODELO MOCK
        cls.vector_db = {
            'embeddings': cls.embeddings,
            'metadata': cls.cpu_data.to_dict('records'),
            'model': MockEmbeddingModel(),  # ¡Ahora con modelo mockeado!
            'component_type': "CPU"
        }
        
        # Crear datos de benchmarks simulados
        cls.cpu_benchmarks = [
            {
                "id": 1,
                "name": "AMD Ryzen 9 7950X",
                "description": "16-Core, 4.5 GHz",
                "samples": 1200,
                "score": 2450,
                "multicore_score": 19800,
                "icon": "AMD",
                "family": "Ryzen 9"
            },
            {
                "id": 2,
                "name": "Intel Core i9-13900K",
                "description": "24-Core, 3.0 GHz",
                "samples": 1500,
                "score": 2400,
                "multicore_score": 21000,
                "icon": "Intel",
                "family": "Core i9"
            },
            {
                "id": 3,
                "name": "AMD Ryzen 7 7800X3D",
                "description": "8-Core, 4.2 GHz",
                "samples": 800,
                "score": 2200,
                "multicore_score": 15000,
                "icon": "AMD",
                "family": "Ryzen 7"
            },
            {
                "id": 4,
                "name": "Intel Core i5-13600K",
                "description": "14-Core, 3.5 GHz",
                "samples": 1000,
                "score": 1950,
                "multicore_score": 14000,
                "icon": "Intel",
                "family": "Core i5"
            },
            {
                "id": 5,
                "name": "AMD Ryzen 5 7600",
                "description": "6-Core, 3.8 GHz",
                "samples": 700,
                "score": 1800,
                "multicore_score": 11000,
                "icon": "AMD",
                "family": "Ryzen 5"
            }
        ]
        
        # Guardar benchmarks en un archivo temporal
        cls.benchmarks_path = "test_cpu_benchmarks.json"
        with open(cls.benchmarks_path, 'w') as f:
            json.dump(cls.cpu_benchmarks, f)
    
    def setUp(self):
        # Crear un blackboard simulado
        class MockBlackboard:
            def __init__(self):
                self.data = {}
                self.subscribers = {}
                self.use_case = UseCase
            
            def update(self, section, data, agent_id, notify=True):
                self.data[section] = data
            
            def get(self, section, agent_id=None):
                return self.data.get(section)
            
            def subscribe(self, event_type, callback):
                pass  # No necesario para pruebas básicas
        
        self.blackboard = MockBlackboard()
        
        # Crear instancia del agente de CPU
        self.cpu_agent = CPUAgent(
            vector_db=self.vector_db,
            cpu_scores_path=self.benchmarks_path,
            blackboard=self.blackboard
        )
    
    def test_gaming_requirements(self):
        """Prueba con requisitos de gaming"""
        requirements = HardwareRequirements(
            use_case=UseCase.GAMING,
            budget={"min": 1000, "max": 2000},
            performance={"resolution": "1440p", "fps": 144},
            aesthetics={},
            constraints=[]
        )
        
        self.blackboard.update('user_requirements', requirements, 'test')
        self.cpu_agent.process_requirements()
        
        proposals = self.blackboard.get('component_proposals')
        self.assertIsNotNone(proposals)
        self.assertTrue('CPU' in proposals)
        self.assertGreater(len(proposals['CPU']), 0)
    
    def test_video_editing_requirements(self):
        """Prueba con requisitos de edición de video"""
        requirements = HardwareRequirements(
            use_case=UseCase.VIDEO_EDITING,
            budget={"min": 1500, "max": 3000},
            performance={"resolution": "4K", "software": ["Premiere Pro", "After Effects"]},
            aesthetics={},
            constraints=[]
        )
        
        self.blackboard.update('user_requirements', requirements, 'test')
        self.cpu_agent.process_requirements()
        
        proposals = self.blackboard.get('component_proposals')
        self.assertIsNotNone(proposals)
        self.assertTrue('CPU' in proposals)
        self.assertGreater(len(proposals['CPU']), 0)
    
    def test_budget_constraint(self):
        """Prueba con restricción de presupuesto"""
        requirements = HardwareRequirements(
            use_case=UseCase.GAMING,
            budget={"min": 500, "max": 800},
            performance={"resolution": "1080p", "fps": 60},
            aesthetics={},
            constraints=[]
        )
        
        self.blackboard.update('user_requirements', requirements, 'test')
        self.cpu_agent.process_requirements()
        
        proposals = self.blackboard.get('component_proposals')
        self.assertIsNotNone(proposals)
        self.assertTrue('CPU' in proposals)
        
        # Verificar que los precios están dentro del presupuesto
        for cpu in proposals['CPU']:
            price = float(cpu['metadata']['Price'])
            self.assertLessEqual(price, 800 * 0.35)  # 35% del presupuesto máximo
    
    def test_small_form_factor_constraint(self):
        """Prueba con restricción de tamaño pequeño"""
        requirements = HardwareRequirements(
            use_case=UseCase.GENERAL,
            budget={"min": 800, "max": 1500},
            performance={},
            aesthetics={},
            constraints=["small_form_factor"]
        )
        
        self.blackboard.update('user_requirements', requirements, 'test')
        self.cpu_agent.process_requirements()
        
        proposals = self.blackboard.get('component_proposals')
        self.assertIsNotNone(proposals)
        self.assertTrue('CPU' in proposals)
        
        # Verificar que solo CPUs con bajo TDP son recomendadas
        for cpu in proposals['CPU']:
            tdp = cpu['metadata']['Details_Thermal Design PowerThermal Design Power']
            if tdp.endswith('W'):
                tdp_value = float(tdp[:-1])
                self.assertLessEqual(tdp_value, 65)
    
    def test_report_generation(self):
        """Prueba de generación de reporte con verificación flexible"""
        requirements = HardwareRequirements(
            use_case=UseCase.GAMING,
            budget={"min": 1200, "max": 2000},
            performance={"resolution": "1440p", "fps": 120},
            aesthetics={"rgb": True},
            constraints=[]
        )
        
        self.blackboard.update('user_requirements', requirements, 'test')
        self.cpu_agent.process_requirements()
        
        proposals = self.blackboard.get('component_proposals')
        report = self.cpu_agent.get_recommendation_report(proposals['CPU'])
        
        # Verificación flexible que SI funcionará
        lower_report = report.lower()
        self.assertTrue(
            any(version in lower_report 
                for version in [
                    "caso de uso: gaming",
                    "caso de uso:gaming",
                    "**caso de uso:** gaming",
                    "**caso de uso:**gaming"
                ]),
            f"No se encontró el caso de uso en el reporte. Reporte completo:\n{report}"
        )
        
        # Otras verificaciones importantes
        self.assertIn("Recomendaciones de CPU", report)
        self.assertIn("Precio:", report)
        self.assertIn("Single-Core", report)

if __name__ == "__main__":
    unittest.main()