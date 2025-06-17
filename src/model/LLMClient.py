from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

class LLMClient(ABC):
    """Interfaz abstracta para clientes de LLM"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Método principal para generar texto"""
        pass

    @staticmethod
    def validate_key(key: str) -> bool:
        """Valida que la API key tenga formato correcto"""
        return key and isinstance(key, str) and len(key) > 20

class OpenAIClient(LLMClient):
    """Implementación para OpenAI"""
    
    def __init__(self, model: str = "gpt-4-turbo"):
        self.client = OpenAI(api_key=self._get_api_key())
        self.model = model
    
    def _get_api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not self.validate_key(key):
            raise ValueError("OpenAI API key inválida o no configurada en .env")
        return key
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error en OpenAI: {str(e)}")

class GeminiClient(LLMClient):
    """Implementación para Google Gemini"""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model_name = model
        self.client = self._initialize_client()
    
    def _get_api_key(self) -> str:
        key = os.getenv("GOOGLE_API_KEY")
        if not self.validate_key(key):
            raise ValueError("Google API key inválida o no configurada en .env")
        return key
    
    def _initialize_client(self):
        genai.configure(api_key=self._get_api_key())
        return genai.GenerativeModel(self.model_name)
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.generate_content(prompt, **kwargs)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Error en Gemini: {str(e)}")