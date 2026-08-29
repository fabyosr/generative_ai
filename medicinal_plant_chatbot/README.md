# Agente Conversacional para Identificação e Educação sobre Plantas Medicinais

Projeto final de Pós-Graduação em IA Generativa e Large Language Models.

Agente conversacional multimodal (texto, imagem e áudio) que identifica
plantas medicinais por busca semântica (dual-encoder) e fornece
informações educacionais fundamentadas por RAG, com fallback hierárquico
(Wikipedia → Tavily) para o que estiver fora da base curada, e guardrails
específicos para o domínio de saúde.

## Status

Esqueleto de arquitetura — módulos com assinaturas, tipagem e docstrings
definidas, lógica de negócio ainda não implementada (`NotImplementedError`
nos pontos a preencher).

## Estrutura do projeto

```
src/
├── app.py                 # Ponto de entrada (Streamlit)
├── core/                  # Modelos de domínio e casos de uso (sem
│                           # dependência de infraestrutura)
├── agents/                # Grafo LangGraph (orquestração)
├── tools/                 # Integrações externas: dual-encoder, RAG,
│                           # busca (Wikipedia/Tavily), TTS
├── guardrails/             # Guardrails de entrada, saída e de escopo
│                           # (camada separada, por decisão de projeto)
├── prompts/                # System prompts como arquivos .md,
│                           # separados do código
├── observability/          # Logging estruturado
└── config/
    ├── settings.py          # Segredos e config de runtime (híbrido:
    │                         # .env local / st.secrets / env vars GCP)
    └── constants.py          # Lista canônica das 6 plantas, limiares
                              # padrão

tests/
├── unit/          # core/ e agents/ isolados, com mocks das dependências
├── integration/   # tools/ com integrações reais
└── evaluation/     # scripts de avaliação (groundedness, alucinação,
                    # acurácia de identificação — Fase 6)

docs/
├── architecture/   # diagramas e descrição da arquitetura
└── adr/             # Architecture Decision Records
```

## Decisões arquiteturais relevantes

Ver `docs/adr/` para o racional completo de cada decisão. Resumo:

- **RAG restrito às 6 plantas** cobertas pelo dual-encoder — Wikipedia e
  Tavily cobrem o restante, por decisão de foco acadêmico.
- **Extração de intenção determinística** (saída estruturada, uma única
  chamada LLM) decide o roteamento antes de qualquer consulta ao
  dual-encoder ou ao RAG — não é um agente ReAct com loop.
- **Guardrail de escopo separado**, posterior no pipeline, para
  demonstrar guardrails como componente independente e testável.
- **Limiares de confiança** (dual-encoder e RAG) são parâmetros
  ajustáveis pelo usuário via sidebar, registrados por turno nos logs
  de observabilidade para preservar rastreabilidade.
- **Aviso de baixa confiança** é injetado via template determinístico no
  próprio texto da resposta — não gerado livremente pelo LLM — para que
  seja automaticamente narrado pelo TTS sem lógica duplicada.
- **Estrutura modular simplificada**, sem Clean Architecture estrita
  (sem camada de "ports" formal para cada integração) — única exceção é
  `tools/search.py`, que tem um `Protocol` real por ter duas
  implementações intercambiáveis (Wikipedia/Tavily).

## Configuração

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# preencha .env com suas chaves (LLM_API_KEY, TAVILY_API_KEY, caminhos
# de modelos locais)
```

Em deploy no **Streamlit Cloud**, configure as mesmas chaves via painel
de `st.secrets` em vez de `.env`. Em deploy no **Google Cloud**,
configure como variáveis de ambiente nativas do serviço. `settings.py`
resolve automaticamente a origem correta — nenhuma alteração de código
é necessária entre os cenários.

## Como executar

```bash
streamlit run src/app.py
```

## Como validar

```bash
# testes unitários (core/ e agents/, com mocks — não requer credenciais)
pytest tests/unit

# testes de integração (requer .env configurado e serviços reais)
pytest tests/integration

# avaliação (Fase 6 — groundedness, alucinação, acurácia de
# identificação, comparação com baseline)
pytest tests/evaluation
```

## Próximos passos de implementação

1. Preencher `config/constants.py` com os dados reais das 6 plantas.
2. Implementar `tools/dual_encoder.py` conectando ao modelo Two-Tower
   já treinado.
3. Curar a base textual das 6 plantas e implementar `tools/rag.py`
   (ingestão + busca).
4. Escrever os prompts em `prompts/*.md`.
5. Implementar os casos de uso em `core/use_cases.py`.
6. Implementar o grafo em `agents/graph.py`.
7. Implementar `guardrails/` (entrada, saída, escopo).
8. Adaptar o TTS Kokoro ONNX já existente em `tools/tts.py`.
9. Construir `app.py` (interface Streamlit).
10. Calibrar empiricamente os limiares padrão (Fase 4 — prova de conceito).
