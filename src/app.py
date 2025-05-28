import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Chatbot",
    page_icon="💻",
    layout="wide"
)

# Modelos disponibles por proveedor
MODEL_OPTIONS: dict[str, list[str]] = {
    "openai": [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo"
    ],
    "google": [
        "gemini-1.5-flash",
        "gemini-pro",
        "gemini-1.5-pro"
    ]
}

# Configurar clientes de IA
def configure_providers():
    """Configura los clientes para los proveedores disponibles"""
    providers = {
        "openai": {
            "client": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
            "default_model": "gpt-3.5-turbo"
        },
        "google": {
            "client": genai.configure(api_key=os.getenv("GOOGLE_API_KEY")),
            "default_model": "gemini-1.5-flash"
        }
    }
    return providers

providers = configure_providers()

# Inicializar el estado de la sesión
if "messages" not in st.session_state:
    st.session_state.update({
        "messages": [{
            "role": "assistant", 
            "content": "¡Hola! Soy tu experto en hardware. ¿En qué puedo ayudarte hoy?"
        }],
        "provider": os.getenv("DEFAULT_PROVIDER", "openai"),
        "model": MODEL_OPTIONS["openai"][0],
        "conversation_history": []
    })

def get_llm_response(user_query, history):
    """Obtiene respuesta del modelo seleccionado"""
    try:
        system_prompt = """
        Eres un experto en hardware de computadoras con estos roles:
        1. ANALISTA: Identifica necesidades técnicas del usuario
        2. TRADUCTOR: Convierte requisitos a especificaciones
        3. ASESOR: Recomienda componentes compatibles
        4. CRÍTICO: Evalúa relación calidad-precio
        
        Reglas:
        - Sé técnico pero claro
        - Pide detalles faltantes
        - Compara opciones
        - Verifica compatibilidad
        - Proporciona rangos de precios
        """
        
        if st.session_state.provider == "openai":
            messages = [{"role": "system", "content": system_prompt}]
            
            # Formatear historial para OpenAI
            for msg in history[-8:]:  # Mantener último contexto
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            response = providers["openai"]["client"].chat.completions.create(
                model=st.session_state.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
            
        elif st.session_state.provider == "google":
            # Configurar modelo Gemini
            model = genai.GenerativeModel(st.session_state.model)
            chat = model.start_chat(history=[])
            
            # Construir contexto
            prompt = system_prompt + "\n\nContexto:\n"
            for msg in history[-6:]:
                prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
            
            prompt += f"\nUsuario: {user_query}"
            response = chat.send_message(prompt)
            return response.text
            
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Interfaz de usuario
def main():
    st.title("🔧 ExpertBot de computadoras")
    
    # Sidebar para configuración avanzada
    with st.sidebar:
        st.header("Configuración de Modelo")
        
        # Selector de proveedor
        new_provider = st.selectbox(
            "Proveedor de IA",
            options=list(MODEL_OPTIONS.keys()),
            index=0 if st.session_state.provider == "openai" else 1,
            format_func=lambda x: "OpenAI" if x == "openai" else "Google"
        )
        
        # Selector de modelo específico
        if new_provider != st.session_state.provider:
            st.session_state.provider = new_provider
            st.session_state.model = MODEL_OPTIONS[new_provider][0]
            st.rerun()
        
        print(list(MODEL_OPTIONS[st.session_state.provider]))
        model_key = st.selectbox(
            "Modelo",
            options=list(MODEL_OPTIONS[st.session_state.provider]),
            index=list(MODEL_OPTIONS[st.session_state.provider]).index(st.session_state.model)
        )
        
        if model_key != st.session_state.model:
            st.session_state.model = model_key
            st.rerun()
        
        st.markdown("---")
        
        if st.button("🔄 Reiniciar conversación"):
            st.session_state.messages = [{
                "role": "assistant",
                "content": f"Conversación reiniciada. Ahora usando {st.session_state.provider.upper()}/{st.session_state.model}"
            }]
            st.rerun()
    
    # Mostrar historial de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption(f"Generado con {st.session_state.provider.upper()} - {st.session_state.model}")
    
    # Input del usuario
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