 
from typing import Dict, List
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from tqdm import tqdm

class CSVToEmbeddings:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        :param embedding_model_name: Modelo de Sentence Transformers a usar
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.text_columns = []  # Se detectarán automáticamente
        
    def _auto_detect_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Detecta automáticamente columnas con texto descriptivo"""
        text_cols = []
        for col in df.columns:
            # Columnas que contienen texto descriptivo
            if any(keyword in col.lower() for keyword in ['model', 'name', 'description', 'type', 'series']):
                text_cols.append(col)
            # O si la mayoría de valores son strings
            elif df[col].apply(lambda x: isinstance(x, str)).mean() > 0.7:
                text_cols.append(col)
        return text_cols
    
    def _clean_text(self, text: str) -> str:
        """Limpia texto eliminando caracteres especiales y normalizando"""
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'[^\w\s.-]', ' ', text)  # Elimina caracteres especiales
        text = re.sub(r'\s+', ' ', text).strip()  # Normaliza espacios
        return text
    
    def _create_dynamic_description(self, row: pd.Series, text_columns: List[str]) -> str:
        """Crea descripción automática usando las columnas de texto detectadas"""
        description_parts = []
        
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                clean_value = self._clean_text(row[col])
                if clean_value:
                    description_parts.append(f"{col.replace('_', ' ')}: {clean_value}")
        
        # Añadir algunas columnas numéricas importantes si existen
        numeric_fields = ['memory', 'clock', 'tdp', 'wattage', 'price']
        for field in numeric_fields:
            for col in row.index:
                if field in col.lower() and pd.notna(row[col]):
                    description_parts.append(f"{col}: {row[col]}")
        
        return ". ".join(description_parts)
    
    def process_csv(self, csv_path: str, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Procesa CSV y genera embeddings sin asumir columnas específicas
        
        :param csv_path: Ruta al archivo CSV
        :param batch_size: Tamaño de lote para generación de embeddings
        :return: Diccionario con embeddings y metadatos
        """
        # Leer CSV con manejo robusto de encoding
        try:
            df = pd.read_csv(csv_path)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin1')
        
        # Detectar columnas de texto automáticamente
        self.text_columns = self._auto_detect_text_columns(df)
        
        if not self.text_columns:
            raise ValueError("No se detectaron columnas de texto descriptivo en el CSV")
        
        # Generar descripciones para cada componente
        descriptions = []
        metadata = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesando filas"):
            desc = self._create_dynamic_description(row, self.text_columns)
            descriptions.append(desc)
            
            # Guardar todos los metadatos disponibles
            metadata.append({
                'id': str(row.get('id', hash(desc))),
                **{k: v for k, v in row.items() if pd.notna(v)}
            })
        
        # Generar embeddings por lotes (más eficiente para grandes datasets)
        embeddings = []
        for i in tqdm(range(0, len(descriptions), batch_size), 
                  desc="Generando embeddings"):
            batch = descriptions[i:i+batch_size]
            embeddings.append(self.embedding_model.encode(batch))
        
        embeddings = np.vstack(embeddings)
        
        return {
            'embeddings': embeddings,
            'metadata': metadata,
            'text_columns': self.text_columns, 
            'model': self.embedding_model
        }