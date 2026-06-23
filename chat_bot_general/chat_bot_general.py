import os
import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# 1. Configuração da Página e Estilização Visual Avançada (CSS)
st.set_page_config(page_title='Virtual Assistant 🤖', page_icon='🤖', layout='centered')

# Injeção de CSS para melhorar o visual do chat e das caixas de mensagem
st.markdown("""
    <style>
        .stChatMessage {
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .stChatInputContainer {
            border-radius: 20px;
        }
        h1 {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-weight: 700;
            background: linear-gradient(45deg, #1E90FF, #12005e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.title('Virtual Assistant 🤖')

# 2. Configurações de API e Inicialização de Variáveis na Sessão
if "api_configured" not in st.session_state:
    try:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        st.session_state.api_configured = True
    except Exception:
        st.warning("Token do HuggingFace não encontrado em st.secrets. Verifique suas configurações.")

# 3. Componente Visual de Configurações (Sidebar)
with st.sidebar:
    st.header("⚙️ Configurações do Assistente")
    model_class = st.selectbox(
        "Selecione o Provedor de LLM:",
        options=['hf_endpoint', 'openai', 'ollama'],
        format_func=lambda x: "HuggingFace (Llama 3)" if x == 'hf_endpoint' else "OpenAI (GPT-4o)" if x == 'openai' else "Ollama (Phi-3)"
    )
    temperature = st.slider("Criatividade (Temperature):", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    
    if st.button("Limpar Histórico do Chat", use_container_width=True):
        st.session_state.chat_history = [AIMessage(content='Oi! Histórico reiniciado. Como posso ajudar você agora?')]
        st.rerun()

# 4. Fábrica de Modelos Otimizada (Evita recriar instâncias sem necessidade)
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

    # Definição limpa dos prompts. O ChatHuggingFace cuida dos tokens estruturais internamente.
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

# 6. Renderização Visual do Histórico (Com Avatares Customizados)
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message('assistant', avatar='🤖'):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message('user', avatar='👤'):
            st.markdown(message.content)

# 7. Fluxo de Entrada e Resposta (User Interaction)
user_query = st.chat_input('Digite sua mensagem aqui...')

if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message('user', avatar='👤'):
        st.markdown(user_query)

    with st.chat_message('assistant', avatar='🤖'):
        # st.write_stream consome o generator e renderiza em tempo real na tela
        resp = st.write_stream(model_response(user_query, st.session_state.chat_history, model_class, temperature))
        
    st.session_state.chat_history.append(AIMessage(content=resp))
