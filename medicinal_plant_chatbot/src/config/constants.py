"""
Constantes de domínio: dados fixos e estáveis do projeto.

Diferente de `config/settings.py`, nada aqui é segredo nem varia por
ambiente — são dados do domínio (a base de conhecimento fechada do
projeto), então ficam versionados normalmente no código.

A lista de plantas abaixo é a fonte única de verdade para:
    - matching determinístico no roteamento (core/use_cases.py::identificar_planta)
    - escopo da base RAG (restrita a estas 6 plantas — decisão de projeto)
    - conjunto de teste da avaliação (Fase 6)

TODO: preencher nomes populares, científicos e sinônimos reais conforme
a curadoria for concluída. Os valores abaixo são placeholders de
estrutura, não dados finais.
"""

from __future__ import annotations

from core.models import ClasseTerapeutica, Planta

# --- Lista canônica das 6 plantas cobertas pelo dual-encoder e pelo RAG ---
#
# Distribuição REAL por classe terapêutica (não uniforme — 3 plantas em
# Cardiovasculares, 1 em cada uma das demais). Isso é um dado do domínio,
# não um erro de curadoria, e deve ser refletido tal como é na avaliação
# (ex.: uma eventual análise de acurácia por classe terá amostra desigual
# entre classes — registrar isso como limitação metodológica na Fase 6).
#
# `sinonimos` está vazio propositalmente: nenhuma variação de nome foi
# fornecida ainda. O matching determinístico (core/use_cases.py::
# identificar_planta) deve normalizar nome_popular e nome_cientifico
# (case-insensitive, sem acentos, hífen/espaço equivalentes) antes de
# comparar, para cobrir variações de digitação sem depender só desta lista.
PLANTAS_CONHECIDAS: tuple[Planta, ...] = (
    Planta(
        nome_popular="Alho",
        nome_cientifico="Allium sativum",
        classe_terapeutica=ClasseTerapeutica.CARDIOVASCULAR,
        sinonimos=(),
    ),
    Planta(
        nome_popular="Sete-sangrias",
        nome_cientifico="Cuphea carthagenensis",
        classe_terapeutica=ClasseTerapeutica.CARDIOVASCULAR,
        sinonimos=(),
    ),
    Planta(
        nome_popular="Hibisco",
        nome_cientifico="Hibiscus sabdariffa",
        classe_terapeutica=ClasseTerapeutica.CARDIOVASCULAR,
        sinonimos=(),
    ),
    Planta(
        nome_popular="Gengibre",
        nome_cientifico="Zingiber officinale Roscoe",
        classe_terapeutica=ClasseTerapeutica.ANTI_INFLAMATORIA,
        sinonimos=(),
    ),
    Planta(
        nome_popular="Carqueja",
        nome_cientifico="Baccharis trimera",
        classe_terapeutica=ClasseTerapeutica.DIGESTIVA,
        sinonimos=(),
    ),
    Planta(
        nome_popular="Camomila",
        nome_cientifico="Matricaria recutita",
        classe_terapeutica=ClasseTerapeutica.CALMANTE,
        sinonimos=(),
    ),
)

# --- Limiar de confiança padrão (ajustável pelo usuário via sidebar em
#     runtime; este é apenas o valor inicial sugerido, a calibrar
#     empiricamente na Fase 4 com os dados de validação do dual-encoder) ---
# Nota: só existe limiar para o dual-encoder. Não há limiar de RAG — a
# consulta ao RAG é uma busca estruturada por planta já identificada
# (ver tools/rag.py), não uma busca por similaridade; um score de RAG
# deixou de fazer sentido depois que o roteamento passou a ser 100%
# determinístico via identificar_planta (ver ADR-002 e ADR-004).
LIMIAR_DUAL_ENCODER_PADRAO: float = 0.5

# --- Limites operacionais ---
MAX_SOLICITACOES_POR_MENSAGEM: int = 5  # cada solicitação extraída gera
                                          # ao menos uma consulta a
                                          # ferramenta (dual-encoder ou
                                          # RAG/fallback) — protege contra
                                          # fan-out excessivo em um turno.
                                          # Ver prompts/extrator_intencao.md

MAX_CARACTERES_MENSAGEM_USUARIO: int = 2000  # guardrail de entrada
                                               # determinístico — ver
                                               # guardrails/input.py

# --- Estágio da conversa (ver core/use_cases.py::calcular_estagio_conversa
#     e prompts/resposta_final.md) — placeholders, a calibrar com uso real ---
LIMITE_TURNOS_DESENVOLVIMENTO: int = 4  # a partir daqui, tende a "fechamento"
LIMITE_TURNOS_FORCADO: int = 6          # teto rígido — força "fechamento_forcado"
                                          # independente de sinal_encerramento
