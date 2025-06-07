import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from model.vectorDB import CSVToEmbeddings
from model.recommender import RecommenderSystem
import pandas as pd

# --- Configuración Inicial ---
load_dotenv()
st.set_page_config(
    page_title="ExpertBot de Hardware", 
    layout="wide",
    page_icon="🖥️"
)

# --- Modelos Disponibles ---
MODEL_OPTIONS = {
    "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    "google": ["gemini-1.5-flash", "gemini-pro", "gemini-1.5-pro"]
}

# --- Clientes de IA ---
@st.cache_resource
def configure_providers():
    return {
        "openai": {
            "client": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
            "default_model": "gpt-4-turbo"
        },
        "google": {
            "client": genai.configure(api_key=os.getenv("GOOGLE_API_KEY")),
            "default_model": "gemini-1.5-flash"
        }
    }

# --- Sistema de Recomendación ---
@st.cache_resource
def load_systems():
    vectorizer = CSVToEmbeddings()
    dbs = {
        'CPU': vectorizer.process_csv('data/component_specs/CPU_specs.csv'),
        'GPU': vectorizer.process_csv('data/component_specs/GPU_specs.csv')
        # Añadir más componentes según sea necesario
    }
    return RecommenderSystem(dbs)

# --- Inicialización ---
providers = configure_providers()
recommender = load_systems()

if "messages" not in st.session_state:
    st.session_state.update({
        "messages": [{
            "role": "assistant", 
            "content": "¡Hola! Soy tu experto en hardware. ¿Qué componentes necesitas hoy?"
        }],
        "provider": "openai",
        "model": MODEL_OPTIONS["openai"][0],
        "recommendations": []
    })

# --- Funciones Clave ---
def get_llm_response(user_query: str, history: list) -> str:
    """Obtiene respuesta del LLM integrando recomendaciones y enlaces"""
    try:
        # Contexto del sistema mejorado
        system_prompt = """
        Eres un experto en hardware con acceso a datos reales de componentes. Reglas estrictas:
        1. NUNCA decir "No puedo proporcionar enlaces" o similares
        2. SIEMPRE usar los enlaces exactos del sistema cuando existan
        3. Si no hay enlace, decir "Consulta disponibilidad en distribuidores"
        4. Los precios mostrados SON los actuales del sistema

        Reglas:
        - Sé técnico pero claro
        - Destaca características clave
        - Siempre menciona precio y enlace si está disponible
        - Explica compatibilidades
        """

        # Procesar con el LLM seleccionado
        if st.session_state.provider == "openai":
            messages = [{"role": "system", "content": system_prompt}]
            messages += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
            
            response = providers["openai"]["client"].chat.completions.create(
                model=st.session_state.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1200
            )
            llm_response = response.choices[0].message.content
            
        elif st.session_state.provider == "google":
            model = genai.GenerativeModel(st.session_state.model)
            chat = model.start_chat(history=[])
            
            context = system_prompt + "\nHistorial:\n" + "\n".join(
                f"{m['role']}: {m['content']}" for m in history[-4:]
            )
            response = chat.send_message(context + f"\nUsuario: {user_query}")
            llm_response = response.text

        # Extraer requisitos y obtener recomendaciones
        component_type = recommender._infer_component_type(user_query)
        if component_type:
            recs = recommender.recommend(user_query, component_type)
            st.session_state.recommendations = recs
            
            # Dentro de get_llm_response(), modifica la sección de recomendaciones:
            if recs:
                llm_response += "\n\n🔍 **Recomendaciones basadas en disponibilidad actual:**\n"
                for i, item in enumerate(recs[:3], 1):
                    response_text = (
                        f"{i}. **{item['Model_Brand']} {item['Model_Name']}**\n"
                        f"   - 💵 Precio actual: ${item['Price']}\n"
                    )
                    
                    # Añadir specs dinámicas
                    specs = {
                        'CPU': f"⚙️ {item.get('Details_# of Cores# of Cores', 'N/A')} núcleos",
                        'GPU': f"🎮 {item.get('Details_Memory Size', 'N/A')}GB VRAM"
                    }
                    response_text += f"   - {specs.get(component_type, '')}\n"
                    
                    # Manejo robusto de URLs
                    if 'URL' in item and str(item['URL']).startswith('http'):
                        response_text += f"   - 🔗 [Ver producto en Newegg]({item['URL']})\n"
                    else:
                        response_text += "   - 📞 Consultar disponibilidad con distribuidores\n"
                    
                    llm_response += response_text
        
        return llm_response

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- Interfaz de Usuario Mejorada ---
def main():
    st.title("🖥️ ExpertBot de Hardware AI")
    
    # Sidebar - Configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Selector de Proveedor
        provider = st.selectbox(
            "Proveedor de IA",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda x: "OpenAI" if x == "openai" else "Google",
            index=0 if st.session_state.provider == "openai" else 1
        )
        
        # Selector de Modelo
        if provider != st.session_state.provider:
            st.session_state.provider = provider
            st.session_state.model = MODEL_OPTIONS[provider][0]
            st.rerun()
        
        model = st.selectbox(
            "Modelo",
            options=MODEL_OPTIONS[st.session_state.provider],
            index=0
        )
        
        if model != st.session_state.model:
            st.session_state.model = model
            st.rerun()
        
        st.markdown("---")
        if st.button("🔄 Reiniciar Conversación", use_container_width=True):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Conversación reiniciada. ¿En qué puedo ayudarte?"
            }]
            st.session_state.recommendations = []
            st.rerun()
    
    # Historial de Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Mostrar recomendaciones expandibles
            if msg["role"] == "assistant" and st.session_state.recommendations:
                with st.expander("📊 Detalles técnicos y compra", expanded=False):
                    for item in st.session_state.recommendations[:3]:
                        st.subheader(f"{item['Model_Brand']} {item['Model_Name']}")
                        
                        # Tarjeta de información
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(f"**Precio:** ${item['Price']}")
                            if 'Details_# of Cores# of Cores' in item:
                                st.caption(f"**Núcleos:** {item['Details_# of Cores# of Cores']}")
                            if 'Details_Memory Size' in item:
                                st.caption(f"**VRAM:** {item['Details_Memory Size']}GB")
                        
                        with col2:
                            if 'URL' in item and pd.notna(item['URL']):
                                st.markdown(
                                    f"<a href='{item['URL']}' target='_blank' style='color: white; background-color: #FF6B00; padding: 0.5em; border-radius: 5px; text-decoration: none;'>🛒 Comprar</a>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.caption("🔴 Enlace no disponible")
    
    # Input de Usuario
    if prompt := st.chat_input("Describe tu necesidad de hardware..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner(f"Consultando {st.session_state.model}..."):
                response = get_llm_response(
                    prompt,
                    st.session_state.messages
                )
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()