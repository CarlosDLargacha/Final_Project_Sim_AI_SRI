from src.scrapper.amazon_scrapper import scrape_amazon_components

# URLs base para diferentes componentes
COMPONENT_URLS = {
    "gpus": "https://www.amazon.com/s?i=computers&rh=n%3A284822",
    "cpus": "https://www.amazon.com/s?i=computers&rh=n%3A229189",
    "ram": "https://www.amazon.com/s?i=computers&rh=n%3A172500",
    "motherboards": "https://www.amazon.com/s?i=computers&rh=n%3A1048424",
    "storage": "https://www.amazon.com/s?i=computers&rh=n%3A1292116011"
}

def main():
    print("Selecciona el componente a scrapear:")
    for i, (key, _) in enumerate(COMPONENT_URLS.items(), 1):
        print(f"{i}. {key.upper()}")
    
    choice = int(input("Ingresa el número: ")) - 1
    component = list(COMPONENT_URLS.keys())[choice]
    
    output_file = f"{component}_urls.json"
    base_url = COMPONENT_URLS[component]
    
    print(f"\nIniciando scrapeo de {component}...")
    results = scrape_amazon_components(
        base_url=base_url,
        output_file=output_file,
        max_pages=5  # Puedes ajustar este valor
    )
    
    print(f"\n¡Scrapeo completado! Resultados:")
    print(f"- URLs encontradas: {len(results)}")
    print(f"- Archivo generado: {output_file}")

if __name__ == "__main__":
    main()