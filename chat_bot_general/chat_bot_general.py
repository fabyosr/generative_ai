import os
import time
import streamlit as st

# Usaremos um utilitário leve e rápido para estimar tokens de forma universal
# Se preferir precisão cirúrgica por modelo, pode trocar por tiktoken ou transformers
def count_tokens(text: str) -> int:
    """Estima de forma rápida a contagem de tokens (média de 4 caracteres por token)"""
    if not text:
        return 0
    return max(1, len(text) // 4)

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# 1. Configuração da Página e Estilização Visual Avançada (CSS)
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
        /* Estilização para os blocos de métricas na barra lateral */
        [data-testid="stMetricValue"] { font-size: 20px !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title('Virtual Assistant 🤖')

# 2. Configurações de API e Inicialização de Variáveis na Sessão
if "api_configured" not in st.session_state:
    try:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        st.session_state.api_configured = True
    except Exception:
        st.warning("Token do HuggingFace não encontrado em st.secrets.")

# Inicializadores das métricas no Session State
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "last_input_tokens": 0,
        "last_output_tokens": 0,
        "history_tokens": 0,
        "latency": 0.0,
        "tokens_per_sec": 0.0
    }

# 3. Componente Visual de Configurações & Dashboard (Sidebar)
with st.sidebar:
    st.header("⚙️ Configurações")
    model_class = st.selectbox(
        "Selecione o Provedor de LLM:",
        options=['hf_endpoint', 'openai', 'ollama'],
        format_func=lambda x: "HuggingFace (Llama 3)" if x == 'hf_endpoint' else "OpenAI (GPT-4o)" if x == 'openai' else "Ollama (Phi-3)"
    )
    temperature = st.slider("Criatividade (Temperature):", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    
    st.markdown("---")
    
    # 📊 DASHBOARD DE MÉTRICAS (UI/UX Limpa usando Expander)
    with st.expander("📊 Monitor de Performance (Última Resposta)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="📥 Input Tokens", value=st.session_state.metrics["last_input_tokens"])
            st.metric(label="⏱️ Latência", value=f"{st.session_state.metrics['latency']:.2}s" if st.session_state.metrics['latency'] > 0 else "0.0s")
        with col2:
            st.metric(label="📤 Output Tokens", value=st.session_state.metrics["last_output_tokens"])
            st.metric(label="⚡ Velocidade", value=f"{st.session_state.metrics['tokens_per_sec']:.1f} t/s" if st.session_state.metrics['tokens_per_sec'] > 0 else "0/s")
            
    with st.expander("🧠 Janela de Contexto Total"):
        total_msg = len(st.session_state.get("chat_history", []))
        st.metric(label="💬 Mensagens no Histórico", value=total_msg)
        st.metric(label="📚 Tokens Acumulados no Chat", value=st.session_state.metrics["history_tokens"])

    st.markdown("---")
    if st.button("Limpar Histórico do Chat", use_container_width=True):
        st.session_state.chat_history = [AIMessage(content='Oi! Histórico reiniciado. Como posso ajudar você agora?')]
        st.session_state.metrics = {"last_input_tokens": 0, "last_output_tokens": 0, "history_tokens": 0, "latency": 0.0, "tokens_per_sec": 0.0}
        st.rerun()

# 4. Fábrica de Modelos
def get_llm_chat(model_class, temperature):
    if model_class == 'hf_endpoint':
        llm_endpoint = HuggingFaceEndpoint(
            repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
            temperature=temperature,
            return_full_text=False,
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=llm_endpoint)
    elif model_class == 'openai':
        return ChatOpenAI(model='gpt-4o-mini', temperature=temperature)
    elif model_class == 'ollama':
        return ChatOllama(model='phi-3', temperature=temperature)

def model_response(user_query, chat_history, model_class, temperature):
    llm_chat = get_llm_chat(model_class, temperature)

    system_prompt = (
        "You are a helpful assistant answering general questions. "
        "CRITICAL: Always reason and plan in English, but you MUST write your Final Answer to the user in Brazilian Portuguese."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    chain = prompt_template | llm_chat | StrOutputParser()
    return chain.stream({"chat_history": chat_history, "input": user_query})

# 5. Inicialização da Memória de Conversa
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content='Oi, sou seu assistente virtual! Como posso ajudar você?')]

# 6. Renderização Visual do Histórico
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message('assistant', avatar='🤖'):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message('user', avatar='👤'):
            st.markdown(message.content)

# 7. Fluxo de Entrada e Cálculo Dinâmico de Métricas
user_query = st.chat_input('Digite sua mensagem aqui...')

if user_query and user_query.strip():
    # Mensagem do Usuário
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message('user', avatar='👤'):
        st.markdown(user_query)

    # Preparação para cálculo das métricas de entrada
    history_string = "".join([m.content for m in st.session_state.chat_history])
    input_tokens = count_tokens(user_query) + count_tokens(history_string)

    with st.chat_message('assistant', avatar='🤖'):
        start_time = time.time() # Início do cronômetro
        
        # Consome a stream do modelo em tempo real
        resp = st.write_stream(model_response(user_query, st.session_state.chat_history, model_class, temperature))
        
        end_time = time.time() # Fim do cronômetro

    # Cálculo final das métricas de performance
    latency = end_time - start_time
    output_tokens = count_tokens(resp)
    tokens_per_second = output_tokens / latency if latency > 0 else 0
    total_chat_tokens = count_tokens(history_string) + output_tokens

    # Atualização das métricas salvas na sessão do Streamlit
    st.session_state.metrics = {
        "last_input_tokens": input_tokens,
        "last_output_tokens": output_tokens,
        "history_tokens": total_chat_tokens,
        "latency": latency,
        "tokens_per_sec": tokens_per_second
    }
    
    st.session_state.chat_history.append(AIMessage(content=resp))
    
    # Atualiza a interface da barra lateral de forma reativa para refletir os novos números
    st.rerun()
