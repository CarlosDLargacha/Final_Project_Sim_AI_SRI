import pandas as pd
from src.scraper.newegg_specs import scrape_newegg_gpu_specs
import os
import time

def main():
    API_KEY = "" 
    INPUT_CSV = "newegg_gpus.csv"
    OUTPUT_CSV = "output/gpu_specs_combined.csv"
    
    # Crear directorios si no existen
    os.makedirs("output", exist_ok=True)
    
    # Leer URLs desde el CSV
    df_urls = pd.read_csv(INPUT_CSV)
    urls = df_urls['url'].tolist()
    
    all_specs = []
    
    for i, url in enumerate(urls):
        print(f"\nProcesando GPU {i+1}/{len(urls)}: {url}")
        
        try:
            # Extraer specs
            specs = scrape_newegg_gpu_specs(api_key=API_KEY, product_url=url)
            
            if specs:
                # Aplanar el diccionario y añadir URL
                flat_specs = {"URL": url}
                for category, details in specs.items():
                    for name, value in details.items():
                        flat_specs[f"{category} - {name}"] = value
                
                all_specs.append(flat_specs)
                print("✓ Datos extraídos")
            else:
                print("✗ No se encontraron specs, omitiendo...")
        
        except Exception as e:
            print(f"✗ Error procesando la URL: {str(e)}")
            continue
        
        time.sleep(2)  # Delay para evitar bloqueos
    
    # Guardar todo en un CSV
    if all_specs:
        df_specs = pd.DataFrame(all_specs)
        df_specs.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\n¡Datos guardados en {OUTPUT_CSV}!")
        print(f"Total GPUs procesadas: {len(all_specs)}/{len(urls)}")
    else:
        print("\nNo se extrajeron datos válidos.")

if __name__ == "__main__":
    main()