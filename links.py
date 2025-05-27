from src.scraper.amazon_scraper import scrape_amazon_gpu_listings, save_to_csv

def main():
    # Configuración
    API_KEY = "" 
    BASE_URL = "https://www.amazon.com/s?bbn=284822&rh=n%3A172282%2Cn%3A541966%2Cn%3A193870011%2Cn%3A284822"  # URL de categoría 'Computers Graphics Cards'
    MAX_PAGES = 20  # Límite de páginas
    
    print("Iniciando scraping de GPUs en Amazon...")
    gpu_urls = scrape_amazon_gpu_listings(api_key=API_KEY, base_url=BASE_URL, max_pages=MAX_PAGES)
    
    if gpu_urls:
        save_to_csv(gpu_urls)
        print(f"Total de URLs extraídas: {len(gpu_urls)}")
    else:
        print("No se encontraron URLs.")

if __name__ == "__main__":
    main()