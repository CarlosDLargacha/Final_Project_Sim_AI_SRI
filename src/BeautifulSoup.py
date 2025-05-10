import requests
from bs4 import BeautifulSoup
import json

def scrape_cubatravel():
    url = "http://www.cubatravel.cu/destinos"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extraer datos principales
        main_data = {
            "titulo": soup.find('h1').get_text(strip=True) if soup.find('h1') else "Sin título",
            "descripcion": soup.find('p').get_text(strip=True) if soup.find('p') else "",
        }
        
        # Extraer instalaciones gastronómicas
        secciones = []
        current_section = {}
        
        for element in soup.select('#dnn_ctr19397_HtmlModule_lblContent h2, #dnn_ctr19397_HtmlModule_lblContent h3, #dnn_ctr19397_HtmlModule_lblContent p'):
            if element.name == 'h2':
                if current_section:
                    secciones.append(current_section)
                current_section = {
                    "categoria": element.get_text(strip=True),
                    "elementos": []
                }
            elif element.name == 'h3':
                current_section['subcategoria'] = element.get_text(strip=True)
            elif element.name == 'p' and current_section:
                if element.find('strong'):
                    item = {
                        "nombre": element.find('strong').get_text(strip=True),
                        "detalles": element.get_text().split(':')[-1].strip()
                    }
                    current_section['elementos'].append(item)
        
        if current_section:
            secciones.append(current_section)
        
        # Extraer imágenes destacadas
        imagenes = [img['src'] for img in soup.select('.img-content')]
        
        # Extraer qué hacer en Cuba
        actividades = []
        for actividad in soup.select('.boxcaption-quehacer'):
            actividades.append({
                "titulo": actividad.find('h3').get_text(strip=True),
                "imagen": actividad.find('img')['src'] if actividad.find('img') else ""
            })
        
        return {
            **main_data,
            "secciones_gastronomicas": secciones,
            "imagenes_destacadas": imagenes,
            "actividades": actividades
        }
        
    except Exception as e:
        print(f"Error al scrapear: {str(e)}")
        return {}

# Ejecutar y guardar resultados
data = scrape_cubatravel()
with open('cuba_travel_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Datos extraídos: {len(data.get('secciones_gastronomicas', []))} secciones")
print(json.dumps(data, indent=2, ensure_ascii=False))