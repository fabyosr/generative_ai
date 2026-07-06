# 📚 RAG Document Chat

Chatbot com Retrieval-Augmented Generation (RAG) para conversar com documentos PDF.  
Construído com **LangChain ≥ 0.2**, **FAISS**, **Streamlit** e suporte a múltiplos provedores LLM.

---

## ✨ Funcionalidades

| Feature | Descrição |
|---|---|
| 📂 Upload de PDFs | Múltiplos arquivos, reindexação automática ao detectar mudança |
| 🤖 Multi-provedor | OpenAI, Groq e HuggingFace Hub — selecionável em tempo real |
| 🧠 Histórico de conversa | Contexto acumulado via `MessagesPlaceholder` |
| 📡 RAG com MMR | Recuperação semântica com diversidade via Maximal Marginal Relevance |
| 🔍 Observabilidade | Tokens, métricas RAG, metadados do modelo, latência — tudo em tempo real |
| 🔐 Secrets seguro | Chaves de API via `st.secrets` (local ou Streamlit Cloud) |

---

## 🗂 Estrutura do Projeto

```
rag_chatbot/
│
├── app.py                        # Orquestrador principal (entry point)
│
├── core/                         # Lógica de negócio — sem dependência Streamlit
│   ├── __init__.py
│   ├── models.py                 # Fábrica de LLMs (Factory Method)
│   ├── rag.py                    # Indexação, retriever e RAG chain
│   ├── metrics.py                # Cálculo de tokens, métricas RAG e latência
│   └── secrets.py                # Injeção de st.secrets → os.environ
│
├── ui/                           # Renderização Streamlit — sem lógica de negócio
│   ├── __init__.py
│   ├── sidebar.py                # Upload, seletor de modelo, temperatura
│   ├── chat.py                   # Histórico, mensagens, popovers de fontes
│   └── observability.py          # Painel de métricas com st.tabs
│
├── vectorstore/                  # Índice FAISS (gerado em runtime, ignorado pelo git)
│
├── .streamlit/
│   ├── config.toml               # Tema e configurações do servidor
│   └── secrets.toml              # Chaves de API (NÃO commitar)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Instalação

### 1. Clone e crie o ambiente

```bash
git clone <seu-repo>
cd rag_chatbot

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

> **Nota:** A instalação do `torch` + `sentence-transformers` pode demorar alguns minutos.  
> Para usar apenas CPU (sem GPU), o `faiss-cpu` já está configurado.

### 3. Configure as chaves de API

Edite `.streamlit/secrets.toml` com suas chaves:

```toml
OPENAI_API_KEY           = "sk-..."
GROQ_API_KEY             = "gsk_..."
HUGGINGFACEHUB_API_TOKEN = "hf_..."
```

> Você só precisa preencher as chaves dos provedores que for usar.  
> O seletor de provedor na sidebar exibirá apenas os que têm chave configurada.

### 4. Execute

```bash
streamlit run app.py
```

---

## 🔍 Painel de Observabilidade

Após enviar uma mensagem, acesse a aba **🔍 Observabilidade** para ver:

### 🧠 Contexto
- Tokens acumulados no histórico (com barra de progresso)
- Distribuição de tokens: Human vs AI
- Contagem de mensagens por papel

### 📡 RAG
- Chunks recuperados na última query
- Fontes únicas (arquivos distintos)
- Tokens de contexto injetados no prompt
- Detalhes de cada chunk: arquivo, página, tokens, prévia

### 🤖 Modelo
- Provedor e modelo ativos
- Temperatura configurada
- Janela de contexto estimada
- Interpretação qualitativa da temperatura

### ⏱ Desempenho
- Latência da última resposta
- Avaliação qualitativa (Excelente / Boa / Moderada / Lenta)
- Histórico de latências da sessão (média, mín, máx)

---

## 🤖 Provedores Suportados

| Provedor | Modelos incluídos | Chave necessária |
|---|---|---|
| 🟢 OpenAI | gpt-4o-mini, gpt-4o, gpt-3.5-turbo | `OPENAI_API_KEY` |
| ⚡ Groq | llama3-70b, llama3-8b, mixtral-8x7b, gemma2-9b | `GROQ_API_KEY` |
| 🤗 HuggingFace | Phi-3-mini, Llama-3-8B, Mistral-7B | `HUGGINGFACEHUB_API_TOKEN` |

---

## 🏗 Arquitetura

```
Upload PDF(s)
     │
     ▼
build_retriever()          ← core/rag.py
  PyPDFLoader → chunks → BGE-M3 embeddings → FAISS index
     │
     ▼
[Usuário digita pergunta]
     │
     ▼
build_rag_chain()          ← core/rag.py
  history_aware_retriever  → reformula pergunta c/ histórico
  FAISS retriever          → busca chunks relevantes (MMR)
  qa_chain                 → gera resposta com contexto
     │
     ▼
compute_*_metrics()        ← core/metrics.py
  tokens, RAG stats, latência
     │
     ▼
render_*()                 ← ui/chat.py + ui/observability.py
  Chat tab + Observabilidade tab
```

---

## 🛠 Decisões Técnicas

**Por que FAISS + BGE-M3?**  
O BGE-M3 é um dos melhores modelos de embedding multilíngue disponíveis gratuitamente, com excelente performance para português. O FAISS oferece busca vetorial rápida localmente sem necessidade de serviços externos.

**Por que MMR no retriever?**  
Maximal Marginal Relevance equilibra relevância com diversidade: evita retornar chunks muito similares entre si, trazendo mais cobertura informacional por query.

**Por que `tiktoken` para métricas de tokens?**  
Para OpenAI e Groq (que usa modelos compatíveis), `tiktoken` fornece contagem precisa. Para HuggingFace, onde cada modelo tem seu próprio tokenizador, usamos a heurística de 4 chars/token como estimativa razoável.

**Por que `st.secrets` ao invés de `.env`?**  
`st.secrets` funciona nativamente tanto localmente (via `.streamlit/secrets.toml`) quanto no Streamlit Cloud, sem dependência de `python-dotenv` em produção.
