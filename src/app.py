import streamlit as st
import os
from time import sleep
from dotenv import load_dotenv
from agents.MB_agent import MotherboardAgent
from agents.compatibility_agent import CompatibilityAgent
from blackboard import Blackboard, EventType
from agents.BDI_agent import BDIAgent, HardwareRequirements
from agents.CPU_agent import CPUAgent
from agents.GPU_agent import GPUAgent
from agents.MB_agent import MotherboardAgent
from agents.storage_agent import StorageAgent
from agents.RAM_agent import RAMAgent
from agents.PSU_agent import PSUAgent
from agents.case_agent import CaseAgent
from agents.compatibility_agent import CompatibilityAgent
from agents.optimization_agent import OptimizationAgent
from agents.optimization_agent import OptimizationAgent
from model.vectorDB import CSVToEmbeddings
from model.LLMClient import OpenAIClient, GeminiClient

# --- Configuración inicial ---
load_dotenv()
st.set_page_config(page_title="ExpertBot de Hardware", layout="wide")

# --- Inicialización de modelos ---
MODEL_OPTIONS = {
    "google": ["gemini-1.5-flash", "gemini-pro", "gemini-1.5-pro"],
    "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
}

# --- Configuración de sesión ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "¡Hola! Soy tu experto en hardware. ¿Qué necesitas?"}
    ]

if "provider" not in st.session_state:
    st.session_state.provider = "openai"

if "model" not in st.session_state:
    st.session_state.model = MODEL_OPTIONS["openai"][0]

if "blackboard" not in st.session_state:
    st.session_state.blackboard = Blackboard(7)

if "user_response" not in st.session_state:
    st.session_state.user_response = None

# --- Inicializar sistema y agentes ---
def init_agents():
    st.session_state.init_agents = True

    processor = CSVToEmbeddings()
    blackboard = st.session_state.blackboard

    llm_client = OpenAIClient(model=st.session_state.model) if st.session_state.provider == "openai" else GeminiClient()

    cpu_db = processor.load_embeddings('CPU')
    gpu_db = processor.load_embeddings('GPU')
    mb_db = processor.load_embeddings('Motherboard')
    hdd_db = processor.load_embeddings('HDD')
    ssd_db = processor.load_embeddings('SSD')
    ram_db = processor.load_embeddings('RAM')
    psu_db = processor.load_embeddings('PSU')
    case_db = processor.load_embeddings('case')
    
    
    agents = {
        'bdi': BDIAgent(
            llm_client=GeminiClient(),
            blackboard=blackboard
        ),
        'cpu': CPUAgent(
            vector_db=cpu_db ,
            cpu_scores_path='src/data/benchmarks/CPU_benchmarks.json',
            blackboard=blackboard
        ),
        'gpu': GPUAgent(
            vector_db=gpu_db,
            gpu_benchmarks_path='src/data/benchmarks/GPU_benchmarks_v7.csv',
            blackboard=blackboard
        ),
        'mb': MotherboardAgent(
            vector_db=mb_db,
            blackboard=blackboard
        ),
        'storage': StorageAgent(
            ssd_vector_db=ssd_db,
            hdd_vector_db=hdd_db,
            blackboard=blackboard
        ),
        'ram': RAMAgent(
            vector_db=ram_db,
            blackboard=blackboard
        ),
        'psu': PSUAgent(
            vector_db=psu_db,
            blackboard=blackboard
        ),
        'case': CaseAgent(
            vector_db=case_db, 
            blackboard=blackboard
        ),
        'comp': CompatibilityAgent(
            blackboard=blackboard,
        ),
        'opt': OptimizationAgent(
            blackboard=blackboard,
        )
    }
    
    def handle_user_response():
        st.session_state.user_response = blackboard.get("user_response")
        
if not st.session_state.get("init_agents"):
    init_agents()

# --- Sidebar de configuración ---
with st.sidebar:
    st.header("⚙️ Configuración")

    provider = st.selectbox("Proveedor de IA", options=list(MODEL_OPTIONS.keys()), format_func=lambda x: x.capitalize())
    if provider != st.session_state.provider:
        st.session_state.provider = provider
        st.session_state.model = MODEL_OPTIONS[provider][0]
        st.rerun()

    model = st.selectbox("Modelo", options=MODEL_OPTIONS[st.session_state.provider])
    if model != st.session_state.model:
        st.session_state.model = model
        st.rerun()

    st.markdown("---")
    if st.button("🔄 Reiniciar conversación", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Conversación reiniciada. ¿Qué necesitas?"}]
        st.session_state.user_response = None
        st.rerun()

# --- Render del historial ---
st.title("🖥️ ExpertBot de Hardware")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Entrada del usuario ---
if prompt := st.chat_input("Describe tu necesidad de hardware..."):
    st.session_state.user_response = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Simular petición al sistema multiagente
    st.session_state.blackboard.update("user_input", {"user_input": prompt}, "user_interface")

    with st.chat_message("assistant"):
        with st.spinner("Analizando componentes y generando configuración óptima..."):
            for _ in range(60):
                st.session_state.user_response = st.session_state.blackboard.get("user_response").get("response")
                if st.session_state.user_response or st.session_state.blackboard.state['errors']:
                    break
                sleep(1)

            response = st.session_state.user_response or "⚠️ No se recibió respuesta del sistema. Intenta nuevamente."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
