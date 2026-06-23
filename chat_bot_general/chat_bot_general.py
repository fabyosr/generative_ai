import os
import time
import streamlit as st
import pandas as pd

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# 1. Configuração de Interface e Estilo
st.set_page_config(page_title='Virtual Assistant 🤖', page_icon='🤖', layout='centered')

st.markdown("""
    <style>
        .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
        .stChatInputContainer { border-radius: 20px; }
        h1 {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-weight: 700;
            background: linear-gradient(45deg, #1E90FF, #12005e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 25px;
        }
        [data-testid="stMetricValue"] { font-size: 20px !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title('Virtual Assistant 🤖')

# 2. Inicialização de APIs e Session State Expandido
if "api_configured" not in st.session_state:
    try:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        st.session_state.api_configured = True
    except Exception:
        st.warning("Token do HuggingFace não encontrado.")

if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "last_input_tokens": 0,
        "last_output_tokens": 0,
        "latency": 0.0,
        "tokens_per_sec": 0.0,
        "finish_reason": "N/A",
        "system_fingerprint": "N/A",
        "reasoning_tokens": 0
    }

if "accumulated_cost" not in st.session_state:
    st.session_state.accumulated_cost = 0.0  # Métrica Cost Monitor acumulada

if "latency_history" not in st.session_state:
    st.session_state.latency_history = []

PERSONALITIES = {
    "Técnico 💻": "You are a highly technical assistant. CRITICAL: Reason in English, answer in Brazilian Portuguese.",
    "Comercial 💼": "You are a business-oriented assistant. CRITICAL: Reason in English, answer in Brazilian Portuguese.",
    "Descontraído ☕": "You are a friendly companion. CRITICAL: Reason in English, answer in Brazilian Portuguese."
}

# 3. Sidebar: Configurações e Dashboard Avançado
with st.sidebar:
    st.header("⚙️ Painel de Controle")
    
    selected_personality = st.selectbox("🎭 Personalidade do Bot:", options=list(PERSONALITIES.keys()))
    
    model_class = st.selectbox(
        "Selecione o Provedor de LLM:",
        options=['openai', 'ollama', 'hf_endpoint'],
        format_func=lambda x: "OpenAI (GPT-4o-mini)" if x == 'openai' else "Ollama (Phi-3)" if x == 'ollama' else "HuggingFace (Llama 3)"
    )
    
    temperature = st.slider("Criatividade (Temperature):", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    
    st.markdown("---")
    
    # 💰 MÉTRICA FINANCEIRA EM DESTAQUE (Cost Monitor)
    st.metric(
        label="💰 Custo Acumulado da Sessão", 
        value=f"USD {st.session_state.accumulated_cost:.5f}",
        help="Custo estimado baseado no consumo real de tokens (Tabela GPT-4o-mini)"
    )
    
    st.markdown("---")
    
    with st.expander("📊 Monitor de Metadados Reais", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="📥 Input Tokens", value=st.session_state.metrics["last_input_tokens"])
            st.metric(label="⏱️ Latência", value=f"{st.session_state.metrics['latency']:.2f}s")
            st.metric(label="🧠 Tokens de Raciocínio", value=st.session_state.metrics["reasoning_tokens"])
        with col2:
            st.metric(label="📤 Output Tokens", value=st.session_state.metrics["last_output_tokens"])
            st.metric(label="⚡ Velocidade", value=f"{st.session_state.metrics['tokens_per_sec']:.1f} t/s")
            st.metric(label="🛑 Fim do Stream", value=st.session_state.metrics["finish_reason"])
            
    with st.expander("📈 Histórico de Latência", expanded=True):
        if st.session_state.latency_history:
            df_latency = pd.DataFrame(st.session_state.latency_history, columns=["Latência (s)"])
            st.line_chart(df_latency, height=120)
            st.caption(f"Fingerprint: {st.session_state.metrics['system_fingerprint']}")
        else:
            st.caption("Envie mensagens para mapear a latência.")

    if st.button("Limpar Tudo", use_container_width=True):
        st.session_state.chat_history = [AIMessage(content='Oi! Tudo reiniciado. Como posso ajudar?')]
        st.session_state.metrics = {"last_input_tokens": 0, "last_output_tokens": 0, "latency": 0.0, "tokens_per_sec": 0.0, "finish_reason": "N/A", "system_fingerprint": "N/A", "reasoning_tokens": 0}
        st.session_state.accumulated_cost = 0.0
        st.session_state.latency_history = []
        st.rerun()

# 4. Fábrica de Modelos atualizada com 'stream_usage' habilitado para OpenAI/Ollama
def get_llm_chat(model_class, temperature):
    if model_class == 'openai':
        return ChatOpenAI(model='gpt-4o-mini', temperature=temperature, stream_usage=True)
    elif model_class == 'ollama':
        return ChatOllama(model='phi-3', temperature=temperature, stream_usage=True)
    elif model_class == 'hf_endpoint':
        llm_endpoint = HuggingFaceEndpoint(repo_id='meta-llama/Meta-Llama-3-8B-Instruct', temperature=temperature, return_full_text=False, max_new_tokens=512)
        return ChatHuggingFace(llm=llm_endpoint)

def model_response(user_query, chat_history, model_class, temperature, personality_prompt):
    llm_chat = get_llm_chat(model_class, temperature)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", personality_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    chain = prompt_template | llm_chat  # Removemos o StrOutputParser para capturar os objetos brutos de Chunk
    return chain.stream({"chat_history": chat_history, "input": user_query})

# 5. Renderização do Histórico
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content='Oi, sou seu assistente virtual! Como posso ajudar você?')]

for message in st.session_state.chat_history:
    avatar = '🤖' if isinstance(message, AIMessage) else '👤'
    with st.chat_message('assistant' if isinstance(message, AIMessage) else 'user', avatar=avatar):
        st.markdown(message.content)

# 6. Captura de Inputs e Loop de Stream Manual
user_query = st.chat_input('Digite sua mensagem aqui...')

if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message('user', avatar='👤'):
        st.markdown(user_query)

    with st.chat_message('assistant', avatar='🤖'):
        # Container vazio que será atualizado caractere por caractere
        message_placeholder = st.empty()
        full_response = ""
        
        # Variáveis para capturar os metadados brutos finais
        final_input_tokens = 0
        final_output_tokens = 0
        reasoning_tokens = 0
        finish_reason = "stop"
        system_fingerprint = "N/A"
        
        start_time = time.time()
        
        # Executa o loop do gerador manualmente
        stream = model_response(user_query, st.session_state.chat_history, model_class, temperature, PERSONALITIES[selected_personality])
        
        for chunk in stream:
            # Concatena e exibe o texto na tela
            if chunk.content:
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
            
            # 🔍 INSPEÇÃO DE METADADOS DINÂMICOS
            # Tratamento para OpenAI e provedores que alimentam usage_metadata nativamente no chunk
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                final_input_tokens = chunk.usage_metadata.get("input_tokens", 0)
                final_output_tokens = chunk.usage_metadata.get("output_tokens", 0)
                # Captura tokens de raciocínio se o modelo suportar
                if "input_token_details" in chunk.usage_metadata:
                    reasoning_tokens = chunk.usage_metadata["input_token_details"].get("reasoning", 0)

            # Captura de metadados adicionais do dicionário de resposta
            if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                metadata = chunk.response_metadata
                if "finish_reason" in metadata:
                    finish_reason = metadata.get("finish_reason")
                if "system_fingerprint" in metadata:
                    system_fingerprint = metadata.get("system_fingerprint")

        end_time = time.time()
        message_placeholder.markdown(full_response) # Remove o cursor de digitação '▌'

    # 🧮 Pós-processamento e Cálculo de Métricas Finais
    latency = end_time - start_time
    tokens_per_second = final_output_tokens / latency if latency > 0 and final_output_tokens > 0 else len(full_response.split()) / latency

    # Fallback caso o provedor (como o HuggingFace gratuito) não devolva os metadados nativos no stream
    if final_input_tokens == 0:
        final_input_tokens = (len(user_query) + len(full_response)) // 4
        final_output_tokens = len(full_response) // 4

    # 💸 Cálculo do Cost Monitor (Preços de referência simulados do gpt-4o-mini: Input=$0.15/M, Output=$0.60/M)
    call_cost = ((final_input_tokens / 1_000_000) * 0.15) + ((final_output_tokens / 1_000_000) * 0.60)
    st.session_state.accumulated_cost += call_cost

    # Salva no estado
    st.session_state.latency_history.append(latency)
    st.session_state.metrics = {
        "last_input_tokens": final_input_tokens,
        "last_output_tokens": final_output_tokens,
        "latency": latency,
