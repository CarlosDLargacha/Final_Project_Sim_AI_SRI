import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import os

# Configuración de ScraperAPI
SCRAPER_API_KEY = "494c3ddce73318b281234f1ebf7124db"
BASE_URL = f"http://api.scraperapi.com?api_key={SCRAPER_API_KEY}&url="

def get_current_price(product_url):
    """Obtiene el precio actual de un producto de Newegg usando ScraperAPI"""
    try:
        # Construir la URL para ScraperAPI
        api_url = BASE_URL + requests.utils.quote(product_url)
        
        # Hacer la solicitud con un timeout razonable
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        # Parsear el HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extraer el precio - ajusta este selector según la estructura actual de Newegg
        price_element = soup.find('li', {'class': 'price-current'})
        if price_element:
            price_text = price_element.get_text(strip=True)
            # Limpiar el texto del precio (esto puede necesitar ajustes)
            price = float(''.join(c for c in price_text if c.isdigit() or c == '.'))
            return price
        return None
    except Exception as e:
        print(f"Error al obtener precio para {product_url}: {str(e)}")
        return None

def update_prices_in_csv(input_csv, output_csv=None):
    """Actualiza los precios en un archivo CSV"""
    if output_csv is None:
        output_csv = input_csv
    
    # Leer el archivo CSV
    df = pd.read_csv(input_csv)
    
    # Verificar que las columnas necesarias existan
    if 'URL' not in df.columns or 'Price' not in df.columns:
        print(f"El archivo {input_csv} no tiene las columnas 'URL' y 'Price'")
        return
    
    # Actualizar precios
    for index, row in df.iterrows():
        if pd.notna(row['URL']):
            current_price = get_current_price(row['URL'])
            if current_price is not None:
                df.at[index, 'Price'] = current_price
                print(f"Actualizado: {row['URL']} - Precio: {current_price}")
            else:
                print(f"Manteniendo precio anterior para: {row['URL']}")
            
            # Esperar para evitar sobrecargar la API
            time.sleep(1)  # Ajusta este delay según tu plan de ScraperAPI
    
    # Guardar los resultados
    df.to_csv(output_csv, index=False)
    print(f"Archivo actualizado guardado como: {output_csv}")

def process_all_csvs(folder_path):
    """Procesa todos los archivos CSV en una carpeta"""
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            full_path = os.path.join(folder_path, filename)
            print(f"\nProcesando archivo: {filename}")
            update_prices_in_csv(full_path)

# Ejemplo de uso
if __name__ == "__main__":
    # Para un solo archivo
    # update_prices_in_csv('productos_newegg.csv', 'productos_actualizados.csv')
    
    # Para todos los archivos CSV en una carpeta
    process_all_csvs('src/data/component_specs')