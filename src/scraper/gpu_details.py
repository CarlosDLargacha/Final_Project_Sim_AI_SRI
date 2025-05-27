import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, Optional
import re
from typing import List, Dict, Optional

def extract_gpu_details(api_key: str, product_url: str) -> Optional[Dict[str, str]]:
    """
    Extrae detalles completos de una página de GPU en Amazon.
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
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"Error al acceder a {product_url}. Status: {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # ===== DATOS BÁSICOS =====
        details = {
            # Información principal
            "name": soup.select_one("#productTitle").get_text(strip=True) if soup.select_one("#productTitle") else "N/A",
            "url": product_url,
            "price": soup.select_one("span.a-price span").get_text(strip=True) if soup.select_one("span.a-price span") else "N/A",
            
            # Disponibilidad
            "availability": soup.select_one("#availability span").get_text(strip=True) if soup.select_one("#availability span") else "In Stock",
            
            # Rating y reviews
            "rating": soup.select_one("span.a-icon-alt").get_text(strip=True).split()[0] if soup.select_one("span.a-icon-alt") else "N/A",
            "review_count": soup.select_one("#acrCustomerReviewText").get_text(strip=True) if soup.select_one("#acrCustomerReviewText") else "0",
            
            # Best Seller Rank
            "best_seller_rank": soup.select_one("tr.po-best_sellers_rank td.a-span9").get_text(strip=True) if soup.select_one("tr.po-best_sellers_rank") else "N/A"
        }

        # ===== ESPECIFICACIONES TÉCNICAS =====
        tech_specs = {
            # Modelo y marca
            "brand": soup.select_one("tr.po-brand td.a-span9").get_text(strip=True) if soup.select_one("tr.po-brand") else "N/A",
            "model": soup.select_one("tr.po-item_model_number td.a-span9").get_text(strip=True) if soup.select_one("tr.po-item_model_number") else "N/A",
            
            # Rendimiento
            "gpu_model": soup.select_one("tr.po-graphics_coprocessor td.a-span9").get_text(strip=True) if soup.select_one("tr.po-graphics_coprocessor") else "N/A",
            "vram": soup.select_one("tr.po-graphics_ram.size td.a-span9").get_text(strip=True) if soup.select_one("tr.po-graphics_ram.size") else "N/A",
            "clock_speed": soup.select_one("tr.po-gpu_clock_speed td.a-span9").get_text(strip=True) if soup.select_one("tr.po-gpu_clock_speed") else "N/A",
            "memory_interface": extract_spec_by_label(soup, "Memory Interface"),
            
            # Consumo
            "tdp": extract_spec_by_label(soup, "TDP", "Wattage", "Power Consumption"),
        }

        # ===== COMPATIBILIDAD =====
        compatibility = {
            "interface": soup.select_one("tr.po-video_output_interface td.a-span9").get_text(strip=True) if soup.select_one("tr.po-video_output_interface") else "N/A",
            "slot_size": extract_spec_by_label(soup, "Slot Size"),
            "length": extract_dimension(soup, "Length"),
            "width": extract_dimension(soup, "Width"),
            "height": extract_dimension(soup, "Height"),
            "ports": extract_spec_by_label(soup, "Ports", "Display Outputs"),
        }

        # ===== INFORMACIÓN ADICIONAL =====
        additional_info = {
            "asin": soup.select_one("th:contains('ASIN') + td").get_text(strip=True) if soup.select_one("th:contains('ASIN')") else "N/A",
            "date_first_available": soup.select_one("th:contains('Date First Available') + td").get_text(strip=True) if soup.select_one("th:contains('Date First Available')") else "N/A",
            "weight": soup.select_one("th:contains('Item Weight') + td").get_text(strip=True) if soup.select_one("th:contains('Item Weight')") else "N/A",
            "package_dimensions": soup.select_one("th:contains('Package Dimensions') + td").get_text(strip=True) if soup.select_one("th:contains('Package Dimensions')") else "N/A",
        }

        # Combinar todos los datos
        details.update({
            **tech_specs,
            **compatibility,
            **additional_info
        })
        
        return details
    
    except Exception as e:
        print(f"Error en {product_url}: {str(e)}")
        return None

# ===== FUNCIONES AUXILIARES =====
def extract_spec_by_label(soup: BeautifulSoup, *labels: str) -> str:
    """Busca especificaciones por múltiples posibles etiquetas."""
    for label in labels:
        elem = soup.find("th", string=re.compile(label, re.IGNORECASE))
        if elem and elem.find_next_sibling("td"):
            return elem.find_next_sibling("td").get_text(strip=True)
    return "N/A"

def extract_dimension(soup: BeautifulSoup, dimension: str) -> str:
    """Extrae dimensiones específicas (Largo/Ancho/Alto)."""
    # Busca en especificaciones técnicas
    value = extract_spec_by_label(soup, f"{dimension}", f"Card {dimension}", f"GPU {dimension}")
    
    # Si no se encuentra, intenta parsear de "Package Dimensions"
    if value == "N/A":
        package_dim = soup.select_one("th:contains('Package Dimensions') + td")
        if package_dim:
            dims = re.findall(r"[\d.]+", package_dim.get_text())
            if dims and len(dims) >= 3:
                return {
                    "Length": dims[0],
                    "Width": dims[1],
                    "Height": dims[2]
                }.get(dimension, "N/A")
    
    return value

def save_to_csv(data: List[Dict[str, str]], filename: str = "gpu_details.csv") -> None:
    """Guarda los datos en CSV con columnas organizadas."""
    df = pd.DataFrame(data)
    
    # Ordenar columnas por categorías
    columns_order = [
        # Datos básicos
        "name", "url", "price", "availability", "rating", "review_count", "best_seller_rank",
        
        # Especificaciones técnicas
        "brand", "model", "gpu_model", "vram", "clock_speed", "memory_interface", "tdp",
        
        # Compatibilidad
        "interface", "slot_size", "length", "width", "height", "ports",
        
        # Información adicional
        "asin", "date_first_available", "weight", "package_dimensions"
    ]
    
    # Mantener solo columnas que existan en los datos
    columns_order = [col for col in columns_order if col in df.columns]
    
    df[columns_order].to_csv(filename, index=False)
    print(f"¡Datos guardados en {filename}!")