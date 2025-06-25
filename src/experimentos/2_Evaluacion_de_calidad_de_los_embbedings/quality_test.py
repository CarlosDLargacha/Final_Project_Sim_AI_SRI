from typing import Dict, Tuple, List
from enum import Enum
from pydantic import BaseModel
from typing import Dict, Any, List
from embedding_evaluator import EmbeddingEvaluator
from agents.BDI_agent import UseCase, HardwareRequirements
from model.vectorDB import CSVToEmbeddings


# Diccionario de pruebas para evaluación de embeddings
quality_embedding_dict = {
    # ------------------------------------------
    # Consulta 1: Gaming básico
    # ------------------------------------------
    "Necesito una PC para gaming en 1080p con presupuesto de $1000": (
        HardwareRequirements(
            use_case=UseCase.GAMING,
            budget={"min": float('-inf'), "max": 1000},
            performance={"resolution": "1080p", "fps": 60, "software": []},
            aesthetics={"color": None, "rgb": False, "window": False},
            constraints=[],
            cpu="Intel Core i5-12400F",
            gpu="NVIDIA RTX 3060",
            storage={"prefer_ssd": True, "include_hdd": False, "capacity": "512GB", "performance": {"read_speed": "2000MB/s"}},
            ram={"capacity": "16GB", "type": "DDR4", "speed": 3200}
        ),
        {
            "CPU": ["Intel Core i5-12400F", "AMD Ryzen 5 5600X", "Intel Core i5-13400F"],
            "GPU": ["NVIDIA RTX 3060", "AMD RX 6600", "NVIDIA RTX 3060 Ti"],
            "RAM": ["Corsair Vengeance LPX 16GB DDR4 3200MHz", "G.Skill Ripjaws V 16GB DDR4 3200MHz"],
            "MB": ["MSI B660M-A", "ASUS TUF B550-PLUS"],
            "PSU": ["EVGA 600W 80+ Gold", "Corsair CX650M"],
            "Case": ["NZXT H510", "Fractal Design Focus G"]
        }
    ),
    
    # ------------------------------------------
    # Consulta 2: Workstation para edición 4K
    # ------------------------------------------
    "Workstation para edición 4K en Premiere con máximo $2500": (
        HardwareRequirements(
            use_case=UseCase.VIDEO_EDITING,
            budget={"min": float('-inf'), "max": 2500},
            performance={"resolution": "4K", "fps": 60, "software": ["Premiere Pro", "After Effects"]},
            aesthetics={"color": None, "rgb": True, "window": True},
            constraints=[],
            cpu="Intel Core i7-13700K",
            gpu="NVIDIA RTX 4070",
            storage={"prefer_ssd": True, "include_hdd": True, "capacity": "2TB", "performance": {"read_speed": "3500MB/s"}},
            ram={"capacity": "32GB", "type": "DDR5", "speed": 5600}
        ),
        {
            "CPU": ["Intel Core i7-13700K", "AMD Ryzen 9 7900X", "Intel Core i9-13900"],
            "GPU": ["NVIDIA RTX 4070", "NVIDIA RTX 4080", "NVIDIA RTX 3090"],
            "RAM": ["Corsair Dominator Platinum 32GB DDR5 5600MHz", "G.Skill Trident Z5 32GB DDR5 5600MHz"],
            "MB": ["ASUS ProArt Z790", "Gigabyte X670 AORUS Elite"],
            "PSU": ["Corsair RM850x", "Seasonic Focus GX-850"],
            "Case": ["Lian Li PC-O11 Dynamic", "Corsair iCUE 4000X"]
        }
    ),

    # ------------------------------------------
    # Consulta 3: PC económica para oficina
    # ------------------------------------------
    "PC económica para oficina y navegación web bajo $500": (
        HardwareRequirements(
            use_case=UseCase.GENERAL,
            budget={"min": float('-inf'), "max": 500},
            performance={"resolution": "1080p", "fps": 30, "software": ["Microsoft Office"]},
            aesthetics={"color": None, "rgb": False, "window": False},
            constraints=["small_form_factor"],
            cpu="AMD Ryzen 5 5600G",
            gpu="Integrada",
            storage={"prefer_ssd": True, "include_hdd": False, "capacity": "256GB", "performance": {"read_speed": "500MB/s"}},
            ram={"capacity": "8GB", "type": "DDR4", "speed": 3200}
        ),
        {
            "CPU": ["AMD Ryzen 5 5600G", "Intel Core i3-12100", "AMD Ryzen 3 5300G"],
            "RAM": ["Crucial 8GB DDR4 3200MHz", "Kingston ValueRAM 8GB DDR4 3200MHz"],
            "MB": ["ASRock B550M-ITX", "Gigabyte B660I AORUS Pro"],
            "PSU": ["EVGA 450W 80+ Bronze", "Thermaltake Smart 500W"],
            "Case": ["Cooler Master Elite 110", "Fractal Design Node 304"]
        }
    ),

    # ------------------------------------------
    # Consulta 4: Máquina para ciencia de datos
    # ------------------------------------------
    "Estación de trabajo para machine learning con GPU NVIDIA y 64GB RAM": (
        HardwareRequirements(
            use_case=UseCase.MACHINE_LEARNING,
            budget={"min": float('-inf'), "max": 3500},
            performance={"resolution": "1440p", "fps": 60, "software": ["TensorFlow", "PyTorch"]},
            aesthetics={"color": "black", "rgb": False, "window": False},
            constraints=[],
            cpu="AMD Ryzen 9 7950X",
            gpu="NVIDIA RTX 4090",
            storage={"prefer_ssd": True, "include_hdd": True, "capacity": "2TB", "performance": {"read_speed": "5000MB/s"}},
            ram={"capacity": "64GB", "type": "DDR5", "speed": 5200}
        ),
        {
            "CPU": ["AMD Ryzen 9 7950X", "Intel Core i9-13900K", "AMD Threadripper 3960X"],
            "GPU": ["NVIDIA RTX 4090", "NVIDIA RTX 3090 Ti", "NVIDIA RTX A6000"],
            "RAM": ["Corsair Vengeance 64GB DDR5 5200MHz", "G.Skill Ripjaws S5 64GB DDR5 5600MHz"],
            "MB": ["ASUS ROG Crosshair X670E Hero", "MSI MEG X670E ACE"],
            "PSU": ["Corsair HX1200", "Seasonic PRIME TX-1000"],
            "Case": ["Fractal Design Meshify 2 XL", "Lian Li Lancool III"]
        }
    ),

    # ------------------------------------------
    # Consulta 5: Build compacta para LAN parties
    # ------------------------------------------
    "PC compacta para gaming en LAN parties con RGB, máximo $1200": (
        HardwareRequirements(
            use_case=UseCase.GAMING,
            budget={"min": float('-inf'), "max": 1200},
            performance={"resolution": "1440p", "fps": 144, "software": []},
            aesthetics={"color": None, "rgb": True, "window": True},
            constraints=["small_form_factor"],
            cpu="AMD Ryzen 7 7700X",
            gpu="NVIDIA RTX 4070",
            storage={"prefer_ssd": True, "include_hdd": False, "capacity": "1TB", "performance": {"read_speed": "3500MB/s"}},
            ram={"capacity": "32GB", "type": "DDR5", "speed": 6000}
        ),
        {
            "CPU": ["AMD Ryzen 7 7700X", "Intel Core i5-13600KF"],
            "GPU": ["NVIDIA RTX 4070", "NVIDIA RTX 4060 Ti"],
            "RAM": ["G.Skill Trident Z5 RGB 32GB DDR5 6000MHz", "Corsair Dominator Platinum RGB 32GB DDR5 6200MHz"],
            "MB": ["ASUS ROG STRIX B650-I", "Gigabyte B650I AORUS Ultra"],
            "PSU": ["Cooler Master V850 SFX", "Lian Li SP750"],
            "Case": ["NZXT H1 V2", "SSUPD Meshlicious"]
        }
    )
}

# Uso del evaluador
if __name__ == "__main__":
    # 1. Inicializar el processor (debe ser tu instancia de CSVToEmbeddings)
    processor = CSVToEmbeddings()
    
    # 2. Crear evaluador
    evaluator = EmbeddingEvaluator(processor)
    
    # 3. Ejecutar evaluación (usando el quality_embedding_dict que definimos antes)
    results_df = evaluator.evaluate_all_queries(quality_embedding_dict, top_k=15)
    
    # 4. Generar reporte
    report = evaluator.generate_text_report(results_df)
    