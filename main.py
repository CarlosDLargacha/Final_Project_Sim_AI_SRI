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

class AmazonGPURealTimeSpider(scrapy.Spider):
    name = 'amazon_gpu_realtime'
    custom_settings = {
        'FEEDS': {
            'gpus_en_tiempo_real.json': {
                'format': 'jsonlines',  # Formato para escritura incremental
                'encoding': 'utf8',
            }
        },
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    def __init__(self):
        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--headless")
        self.driver = uc.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

    def start_requests(self):
        base_url = 'https://www.amazon.com/s?i=computers&rh=n%3A284822&page={}'
        for page in range(1, 21):
            url = base_url.format(page)
            yield scrapy.Request(url, callback=self.parse, meta={'page': page})

    def parse(self, response):
        page = response.meta['page']
        self.driver.get(response.url)
        time.sleep(random.uniform(2, 5))
        
        # Scroll para cargar productos
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        
        # Extraer ASINs y escribir en JSON inmediatamente
        products = self.driver.find_elements(By.XPATH, '//div[@data-asin]')
        for product in products:
            asin = product.get_attribute("data-asin")
            if asin and len(asin) == 10:
                item = {'url': f'https://www.amazon.com/dp/{asin}', 'page': page}
                with open('gpus_en_tiempo_real.json', 'a', encoding='utf-8') as f:
                    f.write(f"{scrapy.utils.serialize.json.dumps(item, ensure_ascii=False)}\n")

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

# Ejecutar
process = CrawlerProcess()
process.crawl(AmazonGPURealTimeSpider)
process.start()