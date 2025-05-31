import pandas as pd
from src.scraper.newegg_specs import *
import os
import time

def main():
    API_KEY = "315e98b8a8ea24b3d682638fd09f54a4"
    INPUT_CSV = "data\product_links\_newegg_SSD_links.csv"  # Debe contener: title,price,url,brand
    OUTPUT_CSV = "data\component_specs\SSD_specs.csv"
    FAILED_LINKS_CSV = "output/failed_links.csv"
    
    os.makedirs("output", exist_ok=True)
    
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {INPUT_CSV}")
        return
    
    all_specs = []
    try:
        a = pd.read_csv(OUTPUT_CSV)
        all_specs = a.to_dict(orient="records")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {OUTPUT_CSV}")
    
    failed_links = []
    
    for idx, row in df.iterrows():
        url = row['url']
        price = row['price']
        print(f"\nProcesando componente {idx + 1}/{len(df)}: {url}")
        
        try:
            # Extraer el tipo de componente de la URL o columna específica
            component_type = 'SSD'
            
            specs = scrape_newegg_specs(API_KEY, url, component_type)
            
            if specs:
                flat_specs = {
                    "URL": url,
                    "Price": price,
                    "Component_Type": component_type,
                    **{f"{cat}_{k}": v for cat, details in specs.items() for k, v in details.items()}
                }
                all_specs.append(flat_specs)
                print("✓ Especificaciones extraídas")
            else:
                print("✗ No se pudieron extraer especificaciones")
                # Guardar el link fallido con todos los datos originales
                failed_links.append({
                    "title": row.get('title', ''),
                    "price": row.get('price', ''),
                    "url": url,
                    "brand": row.get('brand', '')
                })
        
        except Exception as e:
            print(f"✗ Error procesando el componente: {str(e)}")
            failed_links.append({
                "title": row.get('title', ''),
                "price": row.get('price', ''),
                "url": url,
                "brand": row.get('brand', '')
            })
        
        time.sleep(2)  # Delay para evitar bloqueos
    
    # Guardar resultados
    if all_specs:
        pd.DataFrame(all_specs).to_csv(OUTPUT_CSV, index=False)
        print(f"\nEspecificaciones guardadas en {OUTPUT_CSV}")
    
    save_failed_links(failed_links, FAILED_LINKS_CSV)

if __name__ == "__main__":
    main()