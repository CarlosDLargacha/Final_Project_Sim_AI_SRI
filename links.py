from src.scraper.newegg_scraper import scrape_newegg, save_to_csv

def main():
    API_KEY = ""  # ¡Reemplaza esto!
    
    print("Iniciando scraping de CPUs en Newegg con ScraperAPI...")
    comp_data = scrape_newegg(api_key=API_KEY, max_pages=20, base_url="https://www.newegg.com/p/pl?N=100006676&page=1")
    
    if comp_data:
        save_to_csv(comp_data, filename="data\product_links\_newegg_CPU_links.csv")
        print(f"Total de componentes extraídas: {len(comp_data)}")
    else:
        print("No se encontraron datos.")

if __name__ == "__main__":
    main()