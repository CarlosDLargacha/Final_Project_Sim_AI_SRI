
from agents.CPU_agent import CPUAgent
from agents.GPU_agent import GPUAgent
from agents.MB_agent import MotherboardAgent
from agents.storage_agent import StorageAgent
from agents.RAM_agent import RAMAgent
from agents.PSU_agent import PSUAgent
from agents.case_agent import CaseAgent
from agents.compatibility_agent import CompatibilityAgent
from agents.optimization_agent import OptimizationAgent
from blackboard import Blackboard
from model.vectorDB import CSVToEmbeddings
from compatibility_experiment import CompatibilityExperiment

test_cases = [
    # (tipo_comp1, modelo_comp1, tipo_comp2, modelo_comp2, deberían_ser_incompatibles)
    ('CPU', 'Intel Core i9-13900K', 'Motherboard', 'ASUS ROG Strix B550-F', True),
    ('CPU', 'AMD Ryzen 9 7950X', 'Motherboard', 'MSI MAG B660 Tomahawk', True),
    ('GPU', 'NVIDIA RTX 4090', 'Case', 'NZXT H210i', True),  # GPU demasiado grande para el case
    ('RAM', 'Corsair Vengeance DDR5', 'Motherboard', 'ASUS TUF Gaming B450-Plus', True),
    ('CPU', 'AMD Ryzen 7 5800X', 'Motherboard', 'MSI B550 Tomahawk', False),  # Compatibles
    ('GPU', 'AMD RX 6700 XT', 'PSU', 'Corsair RM750x', False),  # Compatibles
    # Agrega más casos según tu conocimiento de incompatibilidades
]


processor = CSVToEmbeddings()

# Inicializar Blackboard
blackboard = Blackboard(7)

cpu_db = processor.load_embeddings('CPU')
gpu_db = processor.load_embeddings('GPU')
mb_db = processor.load_embeddings('Motherboard')
hdd_db = processor.load_embeddings('HDD')
ssd_db = processor.load_embeddings('SSD')
ram_db = processor.load_embeddings('RAM')
psu_db = processor.load_embeddings('PSU')
case_db = processor.load_embeddings('case')

agents = {
    'cpu': CPUAgent(
        vector_db=cpu_db ,
        cpu_scores_path='src/data/benchmarks/CPU_benchmarks.json',
        blackboard=blackboard
    ),
    'gpu': GPUAgent(
        vector_db=gpu_db,
        gpu_benchmarks_path='src/data/benchmarks/GPU_benchmarks_v7.csv',
        blackboard=blackboard
    ),
    'mb': MotherboardAgent(
        vector_db=mb_db,
        blackboard=blackboard
    ),
    'storage': StorageAgent(
        ssd_vector_db=ssd_db,
        hdd_vector_db=hdd_db,
        blackboard=blackboard
    ),
    'ram': RAMAgent(
        vector_db=ram_db,
        blackboard=blackboard
    ),
    'psu': PSUAgent(
        vector_db=psu_db,
        blackboard=blackboard
    ),
    'case': CaseAgent(
        vector_db=case_db, 
        blackboard=blackboard
    ),
    'comp': CompatibilityAgent(
        blackboard=blackboard,
    ),
    'opt': OptimizationAgent(
        blackboard=blackboard,
    )
}



# Inicialización
experiment = CompatibilityExperiment(agents, {
    'cpu': cpu_db,
    'gpu': gpu_db,
    'motherboard': mb_db,
    'ram': ram_db,
    'psu': psu_db,
    'case': case_db,
    'ssd': ssd_db,
    'hdd': hdd_db
})

# Ejecutar pruebas
experiment.run_test_cases(test_cases)

# Obtener resultados
results_df = experiment.get_results_df()
metrics = experiment.calculate_metrics()

print("Métricas del experimento:")
print(f"Precisión: {metrics['accuracy']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
print(f"Verdaderos positivos: {metrics['true_positives']}")
print(f"Falsos negativos: {metrics['false_negatives']}")

# Guardar resultados
results_df.to_csv('compatibility_test_results.csv', index=False)