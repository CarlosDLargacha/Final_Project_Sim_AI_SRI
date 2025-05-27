import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List

def scrape_amazon_gpu_listings(api_key: str, base_url: str, max_pages: int = 20) -> List[str]:
    """
    Extrae URLs de GPUs de la categoría 'Computers Graphics Cards' de Amazon.
    
    Args:
        api_key (str): Tu API key de ScraperAPI.
        base_url (str): URL base de la categoría (ej: categoría de GPUs).
        max_pages (int): Número máximo de páginas a scrapear (default: 20).
    
    Returns:
        List[str]: Lista de URLs únicas de productos.
    """
    product_links = []
    
    for page in range(1, max_pages + 1):
        print(f"Scrapeando página {page}...")
        
        try:
            response = requests.get(
                "http://api.scraperapi.com",
                params={
                    "api_key": api_key,
                    "url": f"{base_url}&page={page}",
                    "render": "true",
                    "country_code": "us"  # Cambia según región
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"Error en página {page}. Status code: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extrae URLs de productos (ajusta según HTML actual de Amazon)
            for item in soup.select('div[data-component-type="s-search-result"] a.a-link-normal[href*="/dp/"]'):
                href = item['href']
                full_url = f"https://www.amazon.com{href.split('?')[0]}"
                product_links.append(full_url)
                
        except Exception as e:
            print(f"Error en página {page}: {str(e)}")
    
    # Elimina duplicados y devuelve
    return list(set(product_links))

def save_to_csv(urls: List[str], filename: str = "amazon_gpu_links.csv") -> None:
    """Guarda las URLs en un archivo CSV."""
    pd.DataFrame(urls, columns=["url"]).to_csv(filename, index=False)
    print(f"¡Datos guardados en {filename}!")