import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict

def scrape_newegg(
    api_key: str,
    base_url: str = "https://www.newegg.com/GPUs-Video-Graphics-Cards/SubCategory/ID-48",
    max_pages: int = 1,
    delay: float = 1.5  # Evitar baneos
) -> List[Dict[str, str]]:
    """
    Scrapea GPUs de Newegg usando ScraperAPI para evitar bloqueos.
    
    Args:
        api_key (str): Tu API key de ScraperAPI.
        base_url (str): URL base de la categoría en Newegg.
        max_pages (int): Número máximo de páginas a scrapear.
        delay (float): Delay entre páginas (en segundos).
    
    Returns:
        List[Dict]: Lista de productos con título, precio, URL y marca.
    """
    all_products = []
    
    for page in range(1, max_pages + 1):
        print(f"Scrapeando página {page}...")
        url = f"{base_url}?Page={page}"
        
        try:
            # Usamos ScraperAPI aquí
            response = requests.get(
                "http://api.scraperapi.com",
                params={
                    "api_key": api_key,
                    "url": url,
                    "render": "true",  # Renderizado JS si es necesario
                    "country_code": "us"  # Geotargeting
                },
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"Error en página {page}. Status: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            items = soup.select('div.item-cell')
            
            for item in items:
                title_elem = item.select_one('a.item-title')
                price_elem = item.select_one('li.price-current strong')
                brand_elem = item.select_one('a.item-brand img')
                
                if not title_elem or not price_elem:
                    continue
                
                product_data = {
                    "title": title_elem.text.strip(),
                    "price": price_elem.text.strip(),
                    "url": title_elem['href'],
                    "brand": brand_elem['title'] if brand_elem else "N/A"
                }
                all_products.append(product_data)
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"Error en página {page}: {str(e)}")
            continue
    
    return all_products

def save_to_csv(data: List[Dict[str, str]], filename: str = "newegg_gpus.csv") -> None:
    """Guarda los datos en un CSV."""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"¡Datos guardados en {filename}!")