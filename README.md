# README: Sistema de Recomendación para Armado de PCs con Arquitectura Multiagente

## Autores
- Gabriel Andrés Pla Lasa
- Carlos Daniel Largacha Leal

## El Problema
Armar una computadora personalizada es un proceso complejo que requiere:
1. **Conocimiento técnico especializado** sobre compatibilidad entre componentes
2. **Actualización constante** ante nuevos lanzamientos de hardware
3. **Balance óptimo** entre presupuesto, rendimiento y necesidades específicas

Los usuarios sin experiencia frecuentemente:
- Compran componentes incompatibles
- Invierten mal su presupuesto
- Subestiman requisitos técnicos para sus casos de uso

Nuestro sistema resuelve estos problemas mediante inteligencia artificial avanzada.

## Requerimientos Técnicos
Para el correcto funcionamiento del sistema se necesita:

### Software
- Python 3.10+
- Gestor de paquetes Pip
- Sistema operativo: Windows 10/11, Linux (Ubuntu 22.04+), macOS Monterey+

## APIs Utilizadas
| Proveedor       | API Key Requerida | Uso en el Sistema                  |
|-----------------|-------------------|------------------------------------|
| ScraperAPI      | Sí                | Extracción de datos de Newegg      |
| OpenAI          | Opcional          | Procesamiento de lenguaje natural  |
| Google Gemini   | Opcional          | Alternativa a OpenAI               |

**Nota**: Las claves API se configuran en `.env`:
```ini
SCRAPERAPI_KEY="tu_api_key"
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="tu_api_key"
```

## Ejecución del Proyecto

### 1. Instalación de dependencias
```bash
pip install -r requirements.txt
```

### 2. Generar embeddings vectoriales
```python
from model.vectorDB import CSVToEmbeddings

processor = CSVToEmbeddings()
vector_db = processor.process_csv('data/component_specs/CPU_specs.csv')
processor.save_embeddings(vector_db)
```

### 3. Ejecutar la aplicación principal
```bash
streamlit run src/app.py
```

### 4. Uso del sistema
1. Acceder a la URL proporcionada por Streamlit (generalmente `http://localhost:8501`)
2. Ingresar requerimientos en lenguaje natural:
   ```
   "Necesito una PC para gaming en 1440p con presupuesto de $1200"
   ```
3. Esperar la recomendación completa con componentes compatibles
4. Revisar justificaciones técnicas para cada componente
