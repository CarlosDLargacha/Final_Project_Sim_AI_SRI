import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re

class CSVToEmbeddings:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.component_types = {
            'CPU': ['Details_# of Cores# of Cores', 'CPU Socket Type_CPU Socket Type', 
                   'Details_Operating FrequencyOperating Frequency'],
            'GPU': ['Details_Memory Size', 'Details_PCI Express Version']
        }

    def _clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = re.sub(r'[^\w\s.-]', ' ', str(text))
        return re.sub(r'\s+', ' ', text).strip()

    def _create_dynamic_description(self, row: pd.Series) -> str:
        """Incluye TODOS los campos técnicos del CSV, con limpieza automática"""
        desc_parts = []
        
        # 1. Campos clave obligatorios (con manejo de errores)
        mandatory_fields = {
            'Component_Type': row.get('Component_Type', 'N/A'),
            'Model_Brand': row.get('Model_Brand', row.get('Brand', 'N/A')),
            'Model_Name': row.get('Model_Name', row.get('Name', 'N/A')),
            'Price': row.get('Price', row.get('price', 0))
        }
        
        desc_parts.append(f"Component Type: {mandatory_fields['Component_Type']}")
        desc_parts.append(f"Model: {self._clean_text(mandatory_fields['Model_Brand'])} {self._clean_text(mandatory_fields['Model_Name'])}")
        desc_parts.append(f"Price: ${mandatory_fields['Price']}")

        # 2. Incluir TODAS las columnas restantes (excepto URLs o metadatos irrelevantes)
        exclude_fields = {'URL', '_Best Seller Ranking', 'Additional Information_Date First Available'}
        
        for col_name, value in row.items():
            if col_name not in mandatory_fields and col_name not in exclude_fields:
                clean_value = self._clean_text(str(value))
                if clean_value and clean_value != 'N/A':
                    # Simplificar nombres de columnas complejos
                    clean_col_name = col_name.split('_')[-1].split('#')[-1]
                    desc_parts.append(f"{clean_col_name}: {clean_value}")
        
        return ". ".join(desc_parts)

    def process_csv(self, csv_path: str, batch_size: int = 32) -> dict:
        try:
            df = pd.read_csv(csv_path)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin1')
            
        # Verificar y estandarizar columna de URL
        url_col = next((col for col in df.columns if 'URL' in col or 'Link' in col), None)
        if url_col:
            df['URL'] = df[url_col]
        else:
            df['URL'] = None

        # Limpieza de datos
        df = df.dropna(how='all').fillna('N/A')
        
        # Generar descripciones
        descriptions = [self._create_dynamic_description(row) for _, row in df.iterrows()]
        
        # Generar embeddings por lotes
        embeddings = []
        for i in tqdm(range(0, len(descriptions), batch_size), 
                    desc="Generando embeddings"):
            batch = descriptions[i:i+batch_size]
            embeddings.append(self.embedding_model.encode(batch))
        
        return {
            'embeddings': np.vstack(embeddings),
            'metadata': df.to_dict('records'),
            'model': self.embedding_model,
            'component_type': df['Component_Type'].iloc[0]
        }