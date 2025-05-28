import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List
import time

def scrape_newegg_gpu_specs(api_key: str, product_url: str) -> Dict[str, Dict[str, str]]:
    """
    Extrae especificaciones técnicas de una GPU en Newegg en el formato solicitado.
    
    Args:
        api_key (str): API key de ScraperAPI.
        product_url (str): URL del producto.
    
    Returns:
        Dict: { "Model": {"Brand": "GIGABYTE", ...}, "Interface": {...}, ... }
    """
    try:
        response = requests.get(
            "http://api.scraperapi.com",
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
        
        # Encontrar todas las tablas de especificaciones
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

def save_specs_to_txt(specs_data: Dict[str, Dict[str, str]], filename: str) -> None:
    """
    Guarda las especificaciones en un archivo de texto con el formato deseado.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for category, specs in specs_data.items():
            f.write(f"{category}\n")
            for name, value in specs.items():
                f.write(f"{name}\t{value}\n")
            f.write("\n")
    print(f"¡Datos guardados en {filename}!")