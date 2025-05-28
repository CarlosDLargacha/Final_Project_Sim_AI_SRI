from src.scraper.newegg_scraper import scrape_newegg_gpus, save_to_csv

def main():
    API_KEY = ""  # ¡Reemplaza esto!
    
    print("Iniciando scraping de GPUs en Newegg con ScraperAPI...")
    gpu_data = scrape_newegg_gpus(api_key=API_KEY, max_pages=20)
    
    if gpu_data:
        save_to_csv(gpu_data)
        print(f"Total de GPUs extraídas: {len(gpu_data)}")
    else:
        print("No se encontraron datos.")

if __name__ == "__main__":
    main()