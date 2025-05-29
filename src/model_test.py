# 1. Inicializar el procesador
from model.vectorDB import CSVToEmbeddings
from model.recommender import RecommenderSystem
from model.vectorCSP import VectorCSP

path = '/home/gabo/Personal/Universidad/3er Año/2do Semetre/IA-SIM-SRI/Final_Project_Sim_AI_SRI/output/gpu_specs_combined.csv'
processor = CSVToEmbeddings()

# 2. Procesar el CSV y crear la base vectorial
vector_db = processor.process_csv(path)

# 3. Mapeo de campos para CSP
FIELD_MAPPING = {
    'price': 'precio',
    'tdp': 'Thermal Design Power',
    'vram': 'Memory Size',
    'psu_watts': 'Recommended PSU Wattage',
    'interface': 'Interface'
}

# 4. Inicializar sistema
recommender = RecommenderSystem(vector_db, VectorCSP(FIELD_MAPPING, vector_db['embeddings']))

# Consulta del usuario
# user_query = "GPU para gaming con buen rendimiento en 4K"
# # Restricciones del usuario
# constraints = {
#     'price': {'type': 'max', 'value': 800},
#     'vram': {'type': 'min', 'value': 8},
#     'tdp': {'type': 'max', 'value': 250},
#     'interface': {'type': 'in', 'value': ['PCI Express 4.0', 'PCI Express 5.0']}
# }

# user_query = "Tarjeta gráfica para diseño gráfico profesional y renderizado 3D"
# constraints = {
#     'price': {'type': 'max', 'value': 1500},       # Presupuesto máximo
#     'vram': {'type': 'min', 'value': 12},          # Mínimo de VRAM
#     'memory_type': {'type': 'eq', 'value': 'GDDR6'}, # Tipo específico de memoria
#     'ports': {'type': 'in', 'value': ['DisplayPort 1.4', 'DisplayPort 2.0']}, # Puertos requeridos, 
#     'Model - Brand': {'type': 'in', 'value': ['GIGABATY']} # Marcas preferidas
# }

user_query = "Componentes eficientes para minería de Ethereum"
constraints = {
    'hash_rate': {'type': 'min', 'value': 30},     # Tasa de hash mínima
    'power_efficiency': {'type': 'max', 'value': 0.2}, # Consumo en kW/h
    'price': {'type': 'max', 'value': 1000},
    'cooling': {'type': 'eq', 'value': 'Advanced'}, # Enfriamiento adecuado
    'availability': {'type': 'eq', 'value': True}   # Solo componentes disponibles
}

# Obtener recomendaciones
recommendations = recommender.recommend(user_query, constraints, top_k=3)

# Mostrar resultados
for i, rec in enumerate(recommendations, 1):
    print(rec)
    # print(f"{i}. {rec['Model - Model']}")
    # print(f"   Precio: ${rec['precio']}")
    # print(f"   VRAM: {rec['Memory Size']}GB")
    # print(f"   TDP: {rec['Thermal Design Power']}W")
    # print(f"   Similitud: {rec['similarity']:.2f}, Score CSP: {rec['csp_score']}")
    # print()