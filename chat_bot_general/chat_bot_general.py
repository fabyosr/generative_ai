import os
import time
import streamlit as st
import pandas as pd

# Função utilitária leve para estimar contagem de tokens
def count_tokens(text: str) -> int:
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
        [data-testid="stMetricValue"] { font-size: 20px !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title('Virtual Assistant 🤖')

# 2. Configurações de API
if "api_configured" not in st.session_state:
    try:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        st.session_state.api_configured = True
    except Exception:
        st.warning("Token do HuggingFace não encontrado em st.secrets.")

# Inicializadores do Session State para Métricas e Histórico de Latência
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "last_input_tokens": 0,
        "last_output_tokens": 0,
        "history_tokens": 0,
        "latency": 0.0,
        "tokens_per_sec": 0.0
    }

if "latency_history" not in st.session_state:
    st.session_state.latency_history = []  # Guarda o histórico de tempo para o gráfico

# 3. Definição de Personalidades do Bot
PERSONALITIES = {
    "Técnico 💻": (
        "You are a highly technical, precise, and analytical assistant. Use factual, structured language. "
        "Focus on architecture, logic, and efficiency. CRITICAL: Always reason and plan in English, but you MUST write your Final Answer to the user in Brazilian Portuguese."
    ),
    "Comercial 💼": (
        "You are a persuasive, professional, and business-oriented sales assistant. Focus on value proposition, benefits, "
        "and professional alignment. Use polished business terminology. CRITICAL: Always reason and plan in English, but you MUST write your Final Answer to the user in Brazilian Portuguese."
    ),
    "Descontraído ☕": (
        "You are a casual, friendly, and enthusiastic AI companion. Use light humor, analogies, and a very approachable tone "
        "while remaining helpful. CRITICAL: Always reason and plan in English, but you MUST write your Final Answer to the user in Brazilian Portuguese."
    )
}

# 4. Componente Visual de Configurações & Dashboard (Sidebar)
with st.sidebar:
    st.header("⚙️ Painel de Controle")
    
    # Seletor Visual de Personalidade
    selected_personality = st.selectbox(
        "🎭 Personalidade do Bot:",
        options=list(PERSONALITIES.keys())
    )
    
    model_class = st.selectbox(
        "Selecione o Provedor de LLM:",
        options=['hf_endpoint', 'openai', 'ollama'],
        format_func=lambda x: "HuggingFace (Llama 3)" if x == 'hf_endpoint' else "OpenAI (GPT-4o)" if x == 'openai' else "Ollama (Phi-3)"
    )
    
    temperature = st.slider("Criatividade (Temperature):", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    
    st.markdown("---")
    
    # 📊 DASHBOARD DE MÉTRICAS
    with st.expander("📊 Monitor de Performance", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="📥 Input Tokens", value=st.session_state.metrics["last_input_tokens"])
            st.metric(label="⏱️ Latência Atual", value=f"{st.session_state.metrics['latency']:.2f}s" if st.session_state.metrics['latency'] > 0 else "0.0s")
        with col2:
            st.metric(label="📤 Output Tokens", value=st.session_state.metrics["last_output_tokens"])
            st.metric(label="⚡ Velocidade", value=f"{st.session_state.metrics['tokens_per_sec']:.1f} t/s" if st.session_state.metrics['tokens_per_sec'] > 0 else "0/s")
            
    # 📈 GRÁFICO DINÂMICO DE HISTÓRICO DE LATÊNCIA
    with st.expander("📈 Histórico de Latência", expanded=True):
        if st.session_state.latency_history:
            # Transforma a lista em um DataFrame indexado para uma exibição bonita
            df_latency = pd.DataFrame(st.session_state.latency_history, columns=["Latência (s)"])
            st.line_chart(df_latency, height=150)
        else:
            st.caption("Envie mensagens para começar a mapear o desempenho do modelo.")

    with st.expander("🧠 Janela de Contexto Total"):
        total_msg = len(st.session_state.get("chat_history", []))
        st.metric(label="💬 Mensagens no Histórico", value=total_msg)
        st.metric(label="📚 Tokens Acumulados", value=st.session_state.metrics["history_tokens"])

    st.markdown("---")
    if st.button("Limpar Histórico do Chat", use_container_width=True):
        st.session_state.chat_history = [AIMessage(content='Oi! Histórico e métricas reiniciados. Como posso ajudar você agora?')]
        st.session_state.metrics = {"last_input_tokens": 0, "last_output_tokens": 0, "history_tokens": 0, "latency": 0.0, "tokens_per_sec": 0.0}
        st.session_state.latency_history = []
        st.rerun()

# 5. Fábrica de Modelos
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

def model_response(user_query, chat_history, model_class, temperature, personality_prompt):
    llm_chat = get_llm_chat(model_class, temperature)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", personality_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    chain = prompt_template | llm_chat | StrOutputParser()
    return chain.stream({"chat_history": chat_history, "input": user_query})

# 6. Inicialização da Memória de Conversa
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content='Oi, sou seu assistente virtual! Como posso ajudar você?')]

# 7. Renderização Visual do Histórico
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message('assistant', avatar='🤖'):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message('user', avatar='👤'):
            st.markdown(message.content)

# 8. Fluxo de Entrada e Processamento Reativo
user_query = st.chat_input('Digite sua mensagem aqui...')

if user_query and user_query.strip():
    # Registrar mensagem do Usuário
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message('user', avatar='👤'):
        st.markdown(user_query)

    # Cálculo dos tokens de entrada
    history_string = "".join([m.content for m in st.session_state.chat_history])
    input_tokens = count_tokens(user_query) + count_tokens(history_string)

    with st.chat_message('assistant', avatar='🤖'):
        start_time = time.time()  # Início do cronômetro
        
        # Resgata a instrução de sistema baseada na escolha da interface
        current_system_prompt = PERSONALITIES[selected_personality]
        
        # Processamento via Stream
        resp = st.write_stream(
            model_response(user_query, st.session_state.chat_history, model_class, temperature, current_system_prompt)
        )
        
        end_time = time.time()  # Fim do cronômetro

    # Cálculos pós-resposta
    latency = end_time - start_time
    output_tokens = count_tokens(resp)
    tokens_per_second = output_tokens / latency if latency > 0 else 0
    total_chat_tokens = count_tokens(history_string) + output_tokens

    # Atualiza histórico de latência para o gráfico de linhas
    st.session_state.latency_history.append(latency)

    # Atualização do dicionário de métricas gerais
    st.session_state.metrics = {
        "last_input_tokens": input_tokens,
        "last_output_tokens": output_tokens,
        "history_tokens": total_chat_tokens,
        "latency": latency,
        "tokens_per_sec": tokens_per_second
    }
    
    st.session_state.chat_history.append(AIMessage(content=resp))
    
    # Força a atualização da interface para renderizar o gráfico e métricas atualizados
    st.rerun()
