import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time
from typing import Dict, List

def scrape_newegg_specs(api_key: str, product_url: str, component_type: str) -> Dict[str, Dict[str, str]]:
    """
    Extrae especificaciones para cualquier componente de Newegg.
    
    Args:
        api_key: API key de ScraperAPI.
        product_url: URL del producto.
        component_type: Tipo de componente ('gpu', 'cpu', 'motherboard', etc.).
    
    Returns:
        Dict con especificaciones organizadas por categoría.
    """
    try:
        response = requests.get(
            "https://api.scraperapi.com",
            params={
                "api_key": api_key,
                "url": product_url,
                "render": "true",
                "country_code": "us"
            },
            timeout=60
        )
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        specs_data = {}
        
        # Selector universal para tablas de specs
        spec_tables = soup.select('div.tab-pane table.table-horizontal')
        
        for table in spec_tables:
            caption = table.find('caption').get_text(strip=True) if table.find('caption') else "Specs"
            rows = table.find_all('tr')
            
            category_specs = {}
            for row in rows:
                th = row.find('th')
                td = row.find('td')
                if th and td:
                    spec_name = th.get_text(strip=True).replace(':', '')
                    spec_value = td.get_text(" ", strip=True)
                    category_specs[spec_name] = spec_value
            
            if category_specs:
                specs_data[caption] = category_specs
        
        return specs_data
    
    except Exception as e:
        print(f"Error al scrapear {product_url}: {str(e)}")
        return {}

def save_failed_links(failed_links: List[Dict[str, str]], filename: str = "failed_links.csv") -> None:
    """
    Guarda los links fallidos en un CSV con el formato original.
    
    Args:
        failed_links: Lista de diccionarios con los datos originales.
        filename: Nombre del archivo de salida.
    """
    if failed_links:
        df = pd.DataFrame(failed_links)
        df.to_csv(filename, index=False)
        print(f"Links fallidos guardados en {filename}")
    else:
        print("No hay links fallidos para guardar.")

