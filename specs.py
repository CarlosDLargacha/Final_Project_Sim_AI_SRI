from src.scraper.gpu_details import extract_gpu_details, save_to_csv
import pandas as pd
import time

def main():
    API_KEY = ""  # Reemplaza con tu API key real
    gpu_urls = pd.read_csv("output/amazon_gpu_links.csv")["url"].tolist()
    
    all_gpus_data = []
    for i, url in enumerate(gpu_urls[:5]):  # Limitar a 50 productos para prueba
        print(f"Extrayendo GPU {i+1}/{len(gpu_urls[:50])}: {url}")
        gpu_data = extract_gpu_details(api_key=API_KEY, product_url=url)
        if gpu_data:
            all_gpus_data.append(gpu_data)
        time.sleep(2)  # Evitar bloqueos
        
    if all_gpus_data:
        save_to_csv(all_gpus_data, "output/gpu_details_full.csv")
        
        # Opcional: Crear una versión simplificada del CSV
        df = pd.DataFrame(all_gpus_data)
        simplified_df = df[[
            # Datos Básicos
            "name", "brand", "price", "availability", "url", "rating", "review_count",
            # Especificaciones Clave
            "gpu_model", "vram", "clock_speed", "tdp",
            # Dimensiones
            "length", "width", "height"
        ]]
        simplified_df.to_csv("output/gpu_details_simplified.csv", index=False)
        print("¡Datos guardados en output/gpu_details_full.csv y versión simplificada!")
    else:
        print("No se extrajeron datos.")

if __name__ == "__main__":
    main()