"""
=============================================================================
core/rag.py — Pipeline RAG: Indexação, Recuperação e Chain
=============================================================================
Responsabilidade:
    Encapsula toda a lógica de Retrieval-Augmented Generation:
      1. Carregamento e parsing de PDFs
      2. Divisão em chunks com RecursiveCharacterTextSplitter
      3. Geração de embeddings (HuggingFace BGE-M3)
      4. Armazenamento e busca vetorial com FAISS
      5. Criação da RAG chain completa com histórico de conversa

Fluxo de dados:
    PDF(s) → chunks → embeddings → FAISS → retriever
    retriever + LLM → history_aware_retriever → rag_chain

Todas as funções são puras (sem st.*), facilitando testes e reuso.
=============================================================================
"""

from __future__ import annotations

import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# ---------------------------------------------------------------------------
# Constantes de configuração do pipeline
# ---------------------------------------------------------------------------

# Modelo de embeddings multilíngue de alta qualidade (BAAI/bge-m3)
EMBEDDING_MODEL = "BAAI/bge-m3"

# Configurações de chunking
CHUNK_SIZE    = 1000   # caracteres por chunk
CHUNK_OVERLAP = 200    # sobreposição para manter contexto entre chunks

# Configurações do retriever MMR
# MMR (Maximal Marginal Relevance) equilibra relevância e diversidade
RETRIEVER_K       = 3   # chunks retornados ao LLM
RETRIEVER_FETCH_K = 4   # chunks candidatos antes da diversificação MMR

# Diretório de persistência do índice FAISS
VECTORSTORE_PATH = "vectorstore/db_faiss"


# ---------------------------------------------------------------------------
# Etapa 1 — Carregamento e indexação de documentos
# ---------------------------------------------------------------------------

def build_retriever(uploaded_files: list) -> FAISS:
    """
    Constrói o retriever a partir de uma lista de arquivos PDF enviados.

    Etapas internas:
        1. Salva cada arquivo em diretório temporário
        2. Carrega páginas via PyPDFLoader
        3. Divide em chunks via RecursiveCharacterTextSplitter
        4. Gera embeddings com BGE-M3
        5. Indexa no FAISS e persiste localmente
        6. Retorna retriever configurado com busca MMR

    Args:
        uploaded_files: Lista de UploadedFile (Streamlit) com PDFs.

    Returns:
        VectorStoreRetriever: Retriever FAISS configurado para MMR.
    """
    docs     = []
    temp_dir = tempfile.TemporaryDirectory()

    # --- 1. Salvar arquivos temporariamente e carregar páginas ---
    for file in uploaded_files:
        temp_path = os.path.join(temp_dir.name, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())

        loader = PyPDFLoader(temp_path)
        docs.extend(loader.load())

    # --- 2. Dividir em chunks ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = splitter.split_documents(docs)

    # --- 3. Gerar embeddings ---
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # --- 4. Criar índice FAISS e persistir ---
    vectorstore = FAISS.from_documents(splits, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    # --- 5. Retornar retriever com MMR ---
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K},
    )

    return retriever


# ---------------------------------------------------------------------------
# Etapa 2 — Construção dos prompts
# ---------------------------------------------------------------------------

def _build_prompts() -> tuple[ChatPromptTemplate, ChatPromptTemplate]:
    """
    Constrói os prompts de contextualização e Q&A como ChatPromptTemplate.

    Por que não há mais diferenciação por provedor aqui:
        O wrapper ChatHuggingFace aplica automaticamente o chat_template
        do tokenizer do modelo via apply_chat_template(), inserindo os tokens
        de controle corretos (<|system|>, <|end|>, <|assistant|>, etc.)
        para cada arquitetura (Phi-3, LLaMA-3, Mistral…).

        Injetar essas tags manualmente no conteúdo das mensagens causaria
        duplicação ou formatação incorreta no prompt final. Todos os
        provedores (OpenAI, Groq, HuggingFace) recebem o mesmo
        ChatPromptTemplate estruturado com roles — cada SDK se encarrega
        da serialização correta para sua API.

    Returns:
        tuple: (context_prompt, qa_prompt)
            - context_prompt: Reformula a pergunta para ser autocontida
                              (necessário para a busca vetorial funcionar
                               sem depender do histórico).
            - qa_prompt:      Responde usando os chunks recuperados,
                              mantendo coerência com o histórico.
    """
    # --- Prompt de contextualização ---
    # O MessagesPlaceholder injeta o chat_history completo entre system e human,
    # permitindo que o LLM reformule a pergunta atual com contexto acumulado.
    context_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the following chat history and the follow-up question, "
         "formulate a standalone question that can be understood without "
         "the chat history. Do NOT answer — only reformulate if needed, "
         "otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # --- Prompt de Q&A ---
    # Usa ChatPromptTemplate (não PromptTemplate de texto puro) para manter
    # a estrutura de roles em todos os provedores.
    # O chat_history aqui garante coerência em conversas longas — o LLM
    # pode referenciar trocas anteriores ao formular a resposta.
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um assistente prestativo respondendo perguntas sobre documentos. "
         "Use o contexto recuperado abaixo para responder. "
         "Se não souber a resposta, diga que não sabe. Seja conciso. "
         "Responda sempre em português.\n\n"
         "Contexto: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    return context_prompt, qa_prompt


# ---------------------------------------------------------------------------
# Etapa 3 — Montagem da RAG Chain completa
# ---------------------------------------------------------------------------

def build_rag_chain(llm, retriever):
    """
    Monta a RAG chain completa com suporte a histórico de conversa.

    Arquitetura da chain:
        [usuário] → history_aware_retriever → reformula pergunta
                  → FAISS retriever         → busca chunks relevantes (MMR)
                  → qa_chain                → gera resposta com contexto

    O argumento `provider` foi removido: os prompts são agora idênticos
    para todos os provedores, pois a serialização de tokens especiais
    é responsabilidade do SDK de cada provedor (ChatHuggingFace,
    ChatOpenAI, ChatGroq), não do prompt template.

    O resultado final expõe as chaves:
        - "answer":  resposta gerada pelo LLM
        - "context": lista de Document com chunks recuperados

    Args:
        llm:       Instância de LLM (qualquer BaseChatModel do LangChain).
        retriever: Retriever FAISS configurado por build_retriever().

    Returns:
        Runnable: Chain invocável com {"input": str, "chat_history": list}.
    """
    context_prompt, qa_prompt = _build_prompts()

    st.markdown(f'context_prompt: {context_prompt}')
    st.markdown(f'qa_prompt: {qa_prompt}')

    # Chain 1: recupera documentos consciente do histórico
    # create_history_aware_retriever: Este módulo ajusta as perguntas do usuário para o contexto da conversa. Se você perguntou "Onde fica?" e depois "Qual é o telefone?", o módulo usa o histórico para reescrever a segunda pergunta como "Qual é o telefone do 'Onde fica'?" antes de buscar nos seus arquivos.
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=context_prompt,
    )

    # Chain 2: gera resposta usando os documentos recuperados
    # create_stuff_documents_chain: Este módulo pega os documentos encontrados na busca e os "empacota" junto com a pergunta do usuário. Ele envia todo esse conteúdo de uma vez para o modelo de linguagem (LLM) gerar a resposta final com base nas suas informações.
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Chain completa: combina recuperação + geração
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain
