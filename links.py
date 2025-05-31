from src.scraper.newegg_scraper import scrape_newegg, save_to_csv

def main():
    API_KEY = "315e98b8a8ea24b3d682638fd09f54a4"  # ¡Reemplaza esto!
    
    print("Iniciando scraping de SSDs en Newegg con ScraperAPI...")
    comp_data = scrape_newegg(api_key=API_KEY, max_pages=20, base_url="https://www.newegg.com/Internal-SSDs/SubCategory/ID-636/Page-1")
    
    if comp_data:
        save_to_csv(comp_data, filename="data\product_links\_newegg_SSD_links.csv")
        print(f"Total de componentes extraídas: {len(comp_data)}")
    else:
        print("No se encontraron datos.")

if __name__ == "__main__":
    main()