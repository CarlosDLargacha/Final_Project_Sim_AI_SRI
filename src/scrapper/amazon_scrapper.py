import scrapy
from scrapy.crawler import CrawlerProcess
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc
import time
import random
import json
from typing import List

class AmazonComponentSpider(scrapy.Spider):
    name = 'amazon_component'
    
    def __init__(self, base_url: str, output_file: str, max_pages: int = 20):
        self.base_url = base_url
        self.output_file = output_file
        self.max_pages = max_pages
        self.results = []
        
        # Configuración de Selenium
        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--headless")
        self.driver = uc.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

    def start_requests(self):
        for page in range(1, self.max_pages + 1):
            url = f"{self.base_url}&page={page}"
            yield scrapy.Request(url, callback=self.parse, meta={'page': page})

    def parse(self, response):
        page = response.meta['page']
        self.driver.get(response.url)
        time.sleep(random.uniform(2, 5))
        
        # Scroll y espera
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        
        # Extracción de productos
        products = self.driver.find_elements(By.XPATH, '//div[@data-asin]')
        for product in products:
            asin = product.get_attribute("data-asin")
            if asin and len(asin) == 10:
                url = f'https://www.amazon.com/dp/{asin}'
                item = {'url': url, 'page': page}
                self.results.append(item)
                
                # Escribe en el archivo inmediatamente
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(f"{json.dumps(item, ensure_ascii=False)}\n")

        # Paginación
        try:
            next_btn = self.driver.find_element(By.CSS_SELECTOR, 'a.s-pagination-next')
            next_btn.click()
            time.sleep(random.uniform(3, 7))
            yield scrapy.Request(self.driver.current_url, callback=self.parse, meta={'page': page + 1})
        except:
            self.logger.info("Fin de las páginas.")

    def closed(self, reason):
        self.driver.quit()
        return self.results

def scrape_amazon_components(base_url: str, output_file: str, max_pages: int = 20) -> List[dict]:
    """Extrae URLs de componentes de PC de Amazon.
    
    Args:
        base_url: URL base de búsqueda (sin número de página)
        output_file: Nombre del archivo JSON de salida
        max_pages: Número máximo de páginas a scrapear
        
    Returns:
        Lista de diccionarios con las URLs encontradas
    """
    process = CrawlerProcess(settings={
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'INFO',
    })
    
    # Limpiar archivo existente
    open(output_file, 'w').close()
    
    crawler = process.create_crawler(AmazonComponentSpider)
    process.crawl(crawler, base_url=base_url, output_file=output_file, max_pages=max_pages)
    process.start()
    
    return crawler.spider.results